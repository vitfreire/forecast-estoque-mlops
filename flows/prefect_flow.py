from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from prefect import flow, task, get_run_logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import joblib


# =============================
# Modelos (import opcional)
# =============================
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None


# =============================
# Utils
# =============================
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_path(*parts: str) -> Path:
    return project_root() / "data" / Path(*parts)


def artifacts_path(*parts: str) -> Path:
    return project_root() / "artifacts" / Path(*parts)


def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v else default


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================
# MLflow Setup
# =============================
def setup_mlflow() -> Path:
    root = project_root()
    tracking_dir = (root / "mlruns").resolve()
    ensure_dir(tracking_dir)

    # Importante: usar o MESMO caminho que você usa no `mlflow ui --backend-store-uri ./mlruns`
    mlflow.set_tracking_uri(str(tracking_dir))
    mlflow.set_experiment(env("MLFLOW_EXPERIMENT", "forecast_walmart"))
    return tracking_dir


# =============================
# Métricas
# =============================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


# =============================
# Config
# =============================
@dataclass
class DatasetConfig:
    raw_dir: str = "raw"
    processed_dir: str = "processed"

    train_file: str = "train.csv"
    features_file: str = "features.csv"
    stores_file: str = "stores.csv"

    target_col: str = "Weekly_Sales"
    date_col: str = "Date"
    store_col: str = "Store"
    dept_col: str = "Dept"
    holiday_col: str = "IsHoliday"

    valid_frac: float = 0.2

    # Feature engineering (lags/rolling sem “ver futuro”: sempre shift(1))
    lags: Tuple[int, ...] = (1, 2, 3, 4)
    windows: Tuple[int, ...] = (4, 8, 12)

    # DEV speed (você pode sobrescrever por env)
    dev_max_rows: int = 20_000
    dev_max_stores: int = 5
    dev_max_depts: int = 5


@dataclass
class TrainConfig:
    random_state: int = 42


# =============================
# Feature Engineering (estilo “antes”, rápido e robusto)
# =============================
def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    dt = pd.to_datetime(d[date_col], errors="coerce")
    d["year"] = dt.dt.year.astype("int32")
    d["month"] = dt.dt.month.astype("int8")
    d["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    d["dayofweek"] = dt.dt.dayofweek.astype("int8")
    return d


def add_lag_roll_features(
    df: pd.DataFrame,
    group_cols: Tuple[str, str],
    date_col: str,
    target_col: str,
    lags: Tuple[int, ...],
    windows: Tuple[int, ...],
) -> pd.DataFrame:
    """
    Lags/Rolling com shift(1): não usa o valor do próprio timestamp para formar features.
    Em forecasting 1-step, isso é correto e não vaza futuro.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.sort_values(list(group_cols) + [date_col]).reset_index(drop=True)

    g = d.groupby(list(group_cols), sort=False)[target_col]

    for lag in lags:
        d[f"lag_{lag}"] = g.shift(lag)

    for w in windows:
        d[f"roll_mean_{w}"] = g.shift(1).transform(lambda s: s.rolling(w).mean())
        d[f"roll_std_{w}"] = g.shift(1).transform(lambda s: s.rolling(w).std())

    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def safe_macro_impute(df: pd.DataFrame, store_col: str) -> pd.DataFrame:
    d = df.copy()
    macro_cols = [
        "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment",
    ]
    for c in macro_cols:
        if c in d.columns:
            d[c] = d.groupby(store_col)[c].transform(lambda s: s.fillna(s.median()))
            d[c] = d[c].fillna(d[c].median())
    return d


def build_design_matrix(df: pd.DataFrame, cfg: DatasetConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    d = df.copy()

    # target
    _y = d[cfg.target_col].astype(float).values  # noqa: F841

    # date -> time features
    d = add_time_features(d, cfg.date_col)
    d = d.drop(columns=[cfg.date_col])

    # bool -> int
    for col in d.columns:
        if col == cfg.target_col:
            continue
        if d[col].dtype == "bool":
            d[col] = d[col].astype("int8")

    # object -> numérico ou category
    obj_cols = d.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        coerced = pd.to_numeric(d[col], errors="coerce")
        if coerced.notna().mean() > 0.95:
            d[col] = coerced
        else:
            d[col] = d[col].astype("category")

    # drop target
    d = d.drop(columns=[cfg.target_col])

    # one-hot para category
    cat_cols = d.select_dtypes(include=["category"]).columns.tolist()
    if cat_cols:
        d = pd.get_dummies(d, columns=cat_cols, drop_first=False)

    # numérico final
    _X = d.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)


def parse_run_models() -> List[str]:
    raw = env("RUN_MODELS", "lightgbm,xgboost,random_forest").strip().lower()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    allowed = {"lightgbm", "xgboost", "random_forest"}
    out = [p for p in parts if p in allowed]
    return out if out else ["lightgbm"]


def get_model(model_name: str, tcfg: TrainConfig):
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            return None
        return LGBMRegressor(
            n_estimators=int(env("LGBM_N_ESTIMATORS", "1500")),
            learning_rate=float(env("LGBM_LR", "0.03")),
            subsample=float(env("LGBM_SUBSAMPLE", "0.8")),
            colsample_bytree=float(env("LGBM_COLSAMPLE", "0.8")),
            random_state=tcfg.random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        if XGBRegressor is None:
            return None
        return XGBRegressor(
            n_estimators=int(env("XGB_N_ESTIMATORS", "1200")),
            learning_rate=float(env("XGB_LR", "0.03")),
            max_depth=int(env("XGB_MAX_DEPTH", "8")),
            subsample=float(env("XGB_SUBSAMPLE", "0.8")),
            colsample_bytree=float(env("XGB_COLSAMPLE", "0.8")),
            reg_lambda=float(env("XGB_REG_LAMBDA", "1.0")),
            random_state=tcfg.random_state,
            n_jobs=-1,
        )

    if model_name == "random_forest":
        if RandomForestRegressor is None:
            return None
        return RandomForestRegressor(
            n_estimators=int(env("RF_N_ESTIMATORS", "300")),
            random_state=tcfg.random_state,
            n_jobs=-1,
            max_depth=None,
        )

    return None


def log_model_to_mlflow(model_name: str, model, X_example: pd.DataFrame) -> None:
    """
    Log do modelo no MLflow com o flavor correto.
    """
    input_example = X_example.head(5)

    if model_name == "lightgbm":
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(model, artifact_path="model", input_example=input_example)

    elif model_name == "xgboost":
        import mlflow.xgboost
        mlflow.xgboost.log_model(model, artifact_path="model", input_example=input_example)

    else:
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)


# =============================
# Tasks
# =============================
@task
def build_base_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    logger = get_run_logger()
    app_env = env("APP_ENV", "prod").lower()

    train = pd.read_csv(data_path(cfg.raw_dir, cfg.train_file))
    features = pd.read_csv(data_path(cfg.raw_dir, cfg.features_file))
    stores = pd.read_csv(data_path(cfg.raw_dir, cfg.stores_file))

    train[cfg.date_col] = pd.to_datetime(train[cfg.date_col], errors="coerce")
    features[cfg.date_col] = pd.to_datetime(features[cfg.date_col], errors="coerce")

    merge_keys = [cfg.store_col, cfg.date_col]
    if cfg.holiday_col in train.columns and cfg.holiday_col in features.columns:
        merge_keys.append(cfg.holiday_col)

    df = train.merge(features, on=merge_keys, how="left")
    df = df.merge(stores, on=cfg.store_col, how="left")

    df = df.dropna(subset=[cfg.target_col, cfg.date_col]).copy()

    if cfg.holiday_col in df.columns:
        df[cfg.holiday_col] = df[cfg.holiday_col].fillna(0).astype(int)

    df = safe_macro_impute(df, cfg.store_col)

    # DEV real (rápido)
    if app_env == "dev":
        dev_max_rows = int(env("DEV_MAX_ROWS", str(cfg.dev_max_rows)))
        dev_max_stores = int(env("DEV_MAX_STORES", str(cfg.dev_max_stores)))
        dev_max_depts = int(env("DEV_MAX_DEPTS", str(cfg.dev_max_depts)))

        df = df.sort_values(cfg.date_col).reset_index(drop=True)

        stores_sample = df[cfg.store_col].dropna().unique().tolist()[:dev_max_stores]
        df = df[df[cfg.store_col].isin(stores_sample)]

        depts_sample = df[cfg.dept_col].dropna().unique().tolist()[:dev_max_depts]
        df = df[df[cfg.dept_col].isin(depts_sample)]

        if len(df) > dev_max_rows:
            df = df.tail(dev_max_rows).copy()

    logger.info(f"Base dataset: {df.shape} | ENV={app_env}")
    return df


@task
def add_features(cfg: DatasetConfig, base_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    # Lags/Rolling no dataset inteiro com shift(1)
    df = add_lag_roll_features(
        df=base_df,
        group_cols=(cfg.store_col, cfg.dept_col),
        date_col=cfg.date_col,
        target_col=cfg.target_col,
        lags=cfg.lags,
        windows=cfg.windows,
    )

    # remove linhas sem histórico mínimo (lags/rolling NaN)
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]
    df = df.dropna(subset=lag_cols).copy()

    # segurança: target não-negativo
    df[cfg.target_col] = df[cfg.target_col].clip(lower=0.0)

    logger.info(f"Feature dataset: {df.shape}")
    return df


@task
def split_dataset(cfg: DatasetConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()

    d = df.sort_values(cfg.date_col).reset_index(drop=True)

    dates = d[cfg.date_col].dropna().sort_values().unique()
    n_dates = len(dates)
    n_valid_dates = int(np.ceil(n_dates * cfg.valid_frac))
    n_train_dates = n_dates - n_valid_dates

    train_cut = dates[n_train_dates - 1]
    train_df = d[d[cfg.date_col] <= train_cut].copy()
    valid_df = d[d[cfg.date_col] > train_cut].copy()

    logger.info(f"Split: train={train_df.shape}, valid={valid_df.shape} | cut={train_cut}")
    return train_df, valid_df


@task
def train_and_track_mlflow(cfg: DatasetConfig, tcfg: TrainConfig, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    setup_mlflow()

    app_env = env("APP_ENV", "prod").lower()
    models_to_run = parse_run_models()

    # design matrix
    X_train, y_train = build_design_matrix(train_df, cfg)
    X_valid, y_valid = build_design_matrix(valid_df, cfg)

    # alinhar colunas
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)

    # outputs locais (deploy)
    out_models_dir = artifacts_path("models")
    ensure_dir(out_models_dir)

    results: List[Dict] = []
    best: Dict[str, object] = {"model_name": None, "rmse": float("inf"), "model": None}

    with mlflow.start_run(run_name=f"prefect_{app_env}"):
        mlflow.set_tag("app_env", app_env)
        mlflow.set_tag("run_type", "train_compare")
        mlflow.log_param("valid_frac", cfg.valid_frac)
        mlflow.log_param("lags", list(cfg.lags))
        mlflow.log_param("windows", list(cfg.windows))
        mlflow.log_param("models", ",".join(models_to_run))
        mlflow.log_param("n_train_rows", int(len(train_df)))
        mlflow.log_param("n_valid_rows", int(len(valid_df)))
        mlflow.log_param("n_features", int(X_train.shape[1]))

        for model_name in models_to_run:
            model = get_model(model_name, tcfg)
            if model is None:
                logger.warning(f"Modelo '{model_name}' não disponível no ambiente.")
                continue

            with mlflow.start_run(run_name=model_name, nested=True):
                # params relevantes
                mlflow.log_param("model_name", model_name)

                # fit/predict
                model.fit(X_train, y_train)
                pred = model.predict(X_valid)

                mae_v = float(mean_absolute_error(y_valid, pred))
                rmse_v = rmse(y_valid, pred)
                smape_v = smape(y_valid, pred)

                # métricas
                mlflow.log_metric("mae", mae_v)
                mlflow.log_metric("rmse", rmse_v)
                mlflow.log_metric("smape", smape_v)

                # log do modelo
                log_model_to_mlflow(model_name, model, X_train)

                results.append({
                    "model": model_name,
                    "mae": mae_v,
                    "rmse": rmse_v,
                    "smape": smape_v,
                })

                # best
                if rmse_v < float(best["rmse"]):
                    best["model_name"] = model_name
                    best["rmse"] = rmse_v
                    best["model"] = model

        if not results:
            raise RuntimeError("Nenhum modelo treinado. Verifique dependências e RUN_MODELS.")

        leaderboard = (
            pd.DataFrame(results)
            .sort_values(["rmse", "mae", "smape"], ascending=True)
            .reset_index(drop=True)
        )

        # (A) registra melhor modelo automaticamente (no run pai)
        mlflow.log_param("best_model", str(best["model_name"]))
        mlflow.log_metric("best_rmse", float(best["rmse"]))

        # (B) salva melhor modelo como artifact local + MLflow artifact
        best_model_path = out_models_dir / "best_model.joblib"
        joblib.dump(best["model"], best_model_path)
        mlflow.log_artifact(str(best_model_path), artifact_path="best_model")

        meta = {
            "best_model": best["model_name"],
            "best_rmse": float(best["rmse"]),
            "features": list(X_train.columns),
            "app_env": app_env,
            "valid_frac": cfg.valid_frac,
            "lags": list(cfg.lags),
            "windows": list(cfg.windows),
        }
        meta_path = out_models_dir / "best_model_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(meta_path), artifact_path="best_model")

        # (C) pronto para deploy local: artifacts/models/*
        logger.info("Best model salvo em: " + str(best_model_path))
        logger.info("Meta salvo em: " + str(meta_path))

        logger.info("\n" + leaderboard.to_string(index=False))
        return leaderboard


# =============================
# Flow
# =============================
@flow(name="forecast_estoque_walmart_flow")
def main():
    cfg = DatasetConfig()
    tcfg = TrainConfig()

    base = build_base_dataset(cfg)
    fe = add_features(cfg, base)
    train_df, valid_df = split_dataset(cfg, fe)
    leaderboard = train_and_track_mlflow(cfg, tcfg, train_df, valid_df)
    return leaderboard


if __name__ == "__main__":
    main()