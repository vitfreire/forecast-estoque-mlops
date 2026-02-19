import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from prefect import flow, task
from dotenv import load_dotenv

from src.config import settings
from src.io import read_raw, write_parquet, read_parquet, ensure_dir
from src.features import merge_walmart, add_time_features, add_lag_features, finalize_features
from src.split import temporal_split
from src.baseline import seasonal_naive_week
from src.metrics import mae, rmse, mape
from src.cost import cost_error
from src.train_lgbm import train_lgbm
from src.predict import predict


# =========================
# CONFIG
# =========================
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "forecast_estoque_walmart")
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "walmart_forecast_lgbm_cost")
ARTIFACTS_DIR = os.getenv("MLFLOW_ARTIFACTS_DIR", "artifacts")
MODEL_NAME_LOCAL = "lgbm_walmart_cost"


# =========================
# HELPERS
# =========================
def should_promote_to_staging(baseline_cost: float, model_cost: float) -> bool:
    return float(model_cost) < float(baseline_cost)


def _safe_end_active_run():
    if mlflow.active_run() is not None:
        mlflow.end_run()


def _log_dict(prefix: str, d: Dict[str, Any]):
    for k, v in d.items():
        try:
            mlflow.log_metric(f"{prefix}{k}", float(v))
        except Exception:
            pass


# =========================
# DRIFT REFERENCE
# =========================
def build_drift_reference(X_ref: pd.DataFrame, feature_names: list, cat_cols: list) -> Dict[str, Any]:

    ref = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "n_rows": int(len(X_ref)),
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "features": {}
    }

    for col in feature_names:

        s = X_ref[col] if col in X_ref.columns else pd.Series(
            [pd.NA] * len(X_ref))
        missing_rate = float(pd.isna(s).mean())

        if col in cat_cols:
            vc = s.astype("string").fillna(
                "__MISSING__").value_counts(normalize=True)
            ref["features"][col] = {
                "type": "categorical",
                "missing_rate": missing_rate,
                "top_values": vc.head(30).to_dict()
            }
        else:
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(s_num) == 0:
                continue

            counts, bins = pd.cut(
                s_num, bins=10, retbins=True, include_lowest=True)
            hist_counts = counts.value_counts(
                sort=False).astype("int64").tolist()

            ref["features"][col] = {
                "type": "numeric",
                "missing_rate": missing_rate,
                "stats": {
                    "mean": float(s_num.mean()),
                    "std": float(s_num.std()),
                    "min": float(s_num.min()),
                    "max": float(s_num.max())
                },
                "hist": {
                    "bins": [float(b) for b in bins.tolist()],
                    "counts": hist_counts
                }
            }

    return ref


@task
def build_and_log_drift_reference(dataset_path: str, cutoff: str):

    preprocess_path = os.path.join(
        ARTIFACTS_DIR, f"{MODEL_NAME_LOCAL}_preprocess.json")

    if not os.path.exists(preprocess_path):
        mlflow.set_tag("drift_reference", "not_created")
        return None

    with open(preprocess_path, "r", encoding="utf-8") as f:
        pre = json.load(f)

    feature_names = pre["feature_names"]
    cat_cols = pre.get("cat_cols", [])

    df = read_parquet(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    cutoff_dt = pd.to_datetime(cutoff)
    ref_df = df[df["Date"] <= cutoff_dt].copy()

    X_ref = ref_df.drop(columns=["Weekly_Sales"], errors="ignore")

    missing_cols = [c for c in feature_names if c not in X_ref.columns]
    for c in missing_cols:
        X_ref[c] = pd.NA

    X_ref = X_ref[feature_names]

    ref = build_drift_reference(X_ref, feature_names, cat_cols)

    ensure_dir(ARTIFACTS_DIR)
    path = os.path.join(ARTIFACTS_DIR, "drift_reference.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)

    mlflow.log_artifact(path, artifact_path="monitoring")

    return path


# =========================
# TASKS
# =========================
@task
def build_dataset():
    raw = read_raw(settings.DATA_RAW_DIR)

    df = merge_walmart(raw["train"], raw["features"], raw["stores"])
    df = add_time_features(df)
    df = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
    df = finalize_features(df)

    ensure_dir(settings.DATA_PROCESSED_DIR)
    path = os.path.join(settings.DATA_PROCESSED_DIR, "dataset.parquet")
    write_parquet(df, path)

    return path


@task
def split_dataset(path):
    df = read_parquet(path)
    train_df, valid_df, cutoff = temporal_split(
        df, "Date", settings.HORIZON_DAYS)
    return train_df, valid_df, str(cutoff.date())


@task
def eval_baseline(train_df, valid_df):

    y_true = valid_df["Weekly_Sales"].values
    y_pred = seasonal_naive_week(train_df, valid_df)

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "cost_total": cost_error(y_true, y_pred, settings.COST_UNDER, settings.COST_OVER)
    }


@task
def train_model(train_df, valid_df):
    return train_lgbm(train_df, valid_df, model_name=MODEL_NAME_LOCAL)


@task
def eval_model(model, valid_df):

    pred_df = predict(model, valid_df, MODEL_NAME_LOCAL)

    ensure_dir(settings.REPORTS_DIR)
    path = os.path.join(settings.REPORTS_DIR, "valid_predictions.parquet")
    pred_df.to_parquet(path, index=False)

    y_true = pred_df["Weekly_Sales"].values
    y_pred = pred_df["y_pred"].values

    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "cost_total": cost_error(y_true, y_pred, settings.COST_UNDER, settings.COST_OVER)
    }

    return metrics, path


@task
def register_model(model):

    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id

    for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
        if mv.run_id == run_id:
            return int(mv.version)

    raise RuntimeError("Model version not found.")


# =========================
# FLOW
# =========================
@flow(name="forecast_estoque_walmart_flow")
def main():

    load_dotenv()
    _safe_end_active_run()

    exp = mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"prefect_{datetime.now().strftime('%Y%m%d_%H%M%S')}", experiment_id=exp.experiment_id) as run:

        mlflow.set_tag("pipeline", "forecast")
        mlflow.log_param("horizon_days", settings.HORIZON_DAYS)

        dataset_path = build_dataset()
        train_df, valid_df, cutoff = split_dataset(dataset_path)

        mlflow.log_artifact(dataset_path, artifact_path="data")

        baseline_metrics = eval_baseline(train_df, valid_df)
        _log_dict("baseline_", baseline_metrics)

        model = train_model(train_df, valid_df)

        model_metrics, pred_path = eval_model(model, valid_df)
        _log_dict("model_", model_metrics)

        mlflow.log_artifact(pred_path, artifact_path="reports")

        drift_path = build_and_log_drift_reference(dataset_path, cutoff)

        version = register_model(model)

        if should_promote_to_staging(baseline_metrics["cost_total"], model_metrics["cost_total"]):
            client = MlflowClient()
            client.transition_model_version_stage(
                REGISTERED_MODEL_NAME,
                str(version),
                "Staging",
                archive_existing_versions=True
            )

        print("RUN_ID:", run.info.run_id)

        return {
            "run_id": run.info.run_id,
            "model_version": version,
            "drift_reference": drift_path
        }


if __name__ == "__main__":
    main()
