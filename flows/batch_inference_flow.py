import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from prefect import flow, task
from dotenv import load_dotenv

from src.config import settings
from src.io import read_parquet, ensure_dir
from src.predict import predict


# =========================
# CONFIG
# =========================
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "walmart_forecast_lgbm_cost"
)

ARTIFACTS_DIR = os.getenv("MLFLOW_ARTIFACTS_DIR", "artifacts")
MODEL_NAME_LOCAL = "lgbm_walmart_cost"


# =========================
# PSI (corrigido)
# =========================
def _psi_from_hist(ref_counts: pd.Series, batch_counts: pd.Series) -> float:
    r = ref_counts.astype("float64")
    b = batch_counts.astype("float64")

    r = r / r.sum()
    b = b / b.sum()

    r = r.replace(0, 1e-8)
    b = b.replace(0, 1e-8)

    psi = np.sum((b - r) * np.log(b / r))

    return float(psi)


# =========================
# Drift numérico
# =========================
def _calc_numeric_drift(ref_feat: Dict[str, Any], batch_series: pd.Series) -> Dict[str, Any]:

    s = pd.to_numeric(batch_series, errors="coerce")
    x = s.dropna()

    ref_counts = pd.Series(ref_feat["hist"]["counts"])
    bins = ref_feat["hist"]["bins"]

    if len(x) == 0:
        return {"psi": None}

    batch_counts = pd.cut(
        x,
        bins=bins,
        include_lowest=True
    ).value_counts(sort=False)

    batch_counts = batch_counts.reindex(range(len(ref_counts)), fill_value=0)

    psi = _psi_from_hist(ref_counts, batch_counts)

    return {
        "psi": psi,
        "mean_batch": float(x.mean()),
        "std_batch": float(x.std(ddof=0)),
    }


# =========================
# Drift categórico
# =========================
def _calc_categorical_drift(ref_feat: Dict[str, Any], batch_series: pd.Series) -> Dict[str, Any]:

    top_values = ref_feat.get("top_values", {})
    total_batch = len(batch_series)

    if total_batch == 0:
        return {"missing_rate": None}

    batch_freq = (
        batch_series
        .astype("string")
        .fillna("__MISSING__")
        .value_counts()
        .to_dict()
    )

    drift = {}
    for k, ref_freq in top_values.items():
        batch_freq_val = batch_freq.get(k, 0) / total_batch
        drift[k] = float(batch_freq_val - ref_freq)

    return {
        "top_value_drift": drift,
        "missing_rate": float(pd.isna(batch_series).mean())
    }


# =========================
# Drift completo
# =========================
def compute_drift_report(reference: Dict[str, Any], batch_X: pd.DataFrame) -> Dict[str, Any]:

    report = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "n_batch_rows": int(len(batch_X)),
        "features": {}
    }

    feature_names = reference.get("feature_names", [])
    cat_cols = reference.get("cat_cols", [])

    for col in feature_names:

        ref_feat = reference["features"].get(col)
        if ref_feat is None:
            continue

        s = batch_X[col] if col in batch_X.columns else pd.Series(
            [pd.NA] * len(batch_X))

        if col in cat_cols:
            out = _calc_categorical_drift(ref_feat, s)
        else:
            out = _calc_numeric_drift(ref_feat, s)

        report["features"][col] = out

    return report


# =========================
# TASKS
# =========================
@task
def load_batch_dataset() -> pd.DataFrame:
    df = read_parquet(os.path.join(
        settings.DATA_PROCESSED_DIR, "dataset.parquet"))
    return df


@task
def select_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - pd.Timedelta(days=settings.HORIZON_DAYS)
    return df[df["Date"] > cutoff]


@task
def load_model_from_registry():
    client = MlflowClient()

    latest = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not latest:
        raise RuntimeError("Modelo não encontrado no registry.")

    latest_version = max(latest, key=lambda mv: int(mv.version))

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version.version}"

    model = mlflow.lightgbm.load_model(model_uri)

    return model


@task
def run_inference_and_save(model, batch_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = predict(model, batch_df, model_name=MODEL_NAME_LOCAL)

    ensure_dir(settings.REPORTS_DIR)
    path = os.path.join(settings.REPORTS_DIR, "batch_predictions.parquet")
    pred_df.to_parquet(path, index=False)

    mlflow.log_artifact(path, artifact_path="batch")

    return pred_df


@task
def load_drift_reference_from_latest_run() -> Optional[Dict[str, Any]]:

    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=["1"],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        return None

    run_id = runs[0].info.run_id

    try:
        local_path = client.download_artifacts(
            run_id,
            "monitoring/drift_reference.json"
        )

        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return None


@task
def compute_and_log_drift(reference: Dict[str, Any], batch_df: pd.DataFrame) -> Optional[str]:

    if reference is None:
        mlflow.set_tag("drift_status", "no_reference_found")
        return None

    batch_X = batch_df.drop(columns=["Weekly_Sales"], errors="ignore")

    report = compute_drift_report(reference, batch_X)

    ensure_dir(ARTIFACTS_DIR)
    path = os.path.join(ARTIFACTS_DIR, "drift_report.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    mlflow.log_artifact(path, artifact_path="monitoring")
    mlflow.set_tag("drift_status", "computed")

    return path


# =========================
# FLOW
# =========================
@flow(name="batch_inference_flow")
def main():

    load_dotenv()

    df = load_batch_dataset()
    batch_df = select_batch(df)

    model = load_model_from_registry()

    pred_df = run_inference_and_save(model, batch_df)

    drift_ref = load_drift_reference_from_latest_run()

    drift_report_path = compute_and_log_drift(drift_ref, batch_df)

    return {
        "n_batch_rows": len(batch_df),
        "drift_report": drift_report_path
    }


if __name__ == "__main__":
    main()
