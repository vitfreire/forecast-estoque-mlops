from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prefect import flow, task, get_run_logger

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Modelos (carrega se existir no ambiente)
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


# -----------------------------
# Utils de caminho / IO
# -----------------------------
def project_root() -> str:
    # flows/prefect_flow.py -> volta 1 nível (flows) e mais 1 (raiz)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_path(*parts: str) -> str:
    return os.path.join(project_root(), "data", *parts)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Métricas
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


# -----------------------------
# Config
# -----------------------------
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
    # split temporal: % final para validação
    valid_frac: float = 0.2


@dataclass
class TrainConfig:
    random_state: int = 42


# -----------------------------
# Tasks
# -----------------------------
@task
def build_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    logger = get_run_logger()

    train_path = data_path(cfg.raw_dir, cfg.train_file)
    feat_path = data_path(cfg.raw_dir, cfg.features_file)
    stores_path = data_path(cfg.raw_dir, cfg.stores_file)

    # valida presença
    missing = [p for p in [train_path, feat_path,
                           stores_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Arquivos não encontrados:\n" + "\n".join(missing) +
            "\n\nEsperado em: data/raw/{train.csv, features.csv, stores.csv}"
        )

    df_train = pd.read_csv(train_path)
    df_feat = pd.read_csv(feat_path)
    df_stores = pd.read_csv(stores_path)

    # normaliza Date
    for df in (df_train, df_feat):
        if cfg.date_col in df.columns:
            df[cfg.date_col] = pd.to_datetime(
                df[cfg.date_col], errors="coerce")

    # merge train + features
    # Alguns datasets têm IsHoliday em ambos; usamos join robusto.
    merge_keys = [cfg.store_col, cfg.date_col]
    if cfg.holiday_col in df_train.columns and cfg.holiday_col in df_feat.columns:
        merge_keys.append(cfg.holiday_col)

    df = df_train.merge(df_feat, on=merge_keys, how="left")

    # merge stores (Store é a chave)
    if cfg.store_col in df_stores.columns:
        df = df.merge(df_stores, on=cfg.store_col, how="left")
    else:
        logger.warning("stores.csv não tem coluna 'Store'. Merge ignorado.")

    # valida target
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Coluna target '{cfg.target_col}' não encontrada no train.csv.")

    # remove linhas sem Date/target
    df = df.dropna(subset=[cfg.date_col, cfg.target_col]).copy()

    # features temporais
    dt = df[cfg.date_col]
    df["year"] = dt.dt.year.astype("int32")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["dayofweek"] = dt.dt.dayofweek.astype("int8")
    df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")

    # drop Date (evita dtype datetime nos modelos)
    df = df.drop(columns=[cfg.date_col])

    # Tipos: garante numérico onde faz sentido
    # Mantém bool/categ como int (modelos lidam melhor assim sem encoding pesado)
    for col in df.columns:
        if col == cfg.target_col:
            continue
        if df[col].dtype == "bool":
            df[col] = df[col].astype("int8")

    # Converte objetos para categoria ou numérico quando possível
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        # tenta numérico
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() > 0.95:
            df[col] = coerced
        else:
            df[col] = df[col].astype("category")

    logger.info(f"Dataset final: shape={df.shape}")
    return df


@task
def split_dataset(cfg: DatasetConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal: ordena por year/month/week/day (criados) e pega o final como validação.
    Evita vazamento em séries temporais.
    """
    logger = get_run_logger()

    sort_cols = [c for c in ["year", "month",
                             "weekofyear", "day"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    n = len(df_sorted)
    n_valid = int(np.ceil(n * cfg.valid_frac))
    n_train = n - n_valid

    train_df = df_sorted.iloc[:n_train].copy()
    valid_df = df_sorted.iloc[n_train:].copy()

    logger.info(
        f"Split temporal: train={train_df.shape}, valid={valid_df.shape}")
    return train_df, valid_df


def make_X_y(cfg: DatasetConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[cfg.target_col].values
    X = df.drop(columns=[cfg.target_col]).copy()

    # One-hot para categorias (leve e robusto)
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # garante numérico puro
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y


@task
def train_and_compare(cfg: DatasetConfig, tcfg: TrainConfig, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    X_train, y_train = make_X_y(cfg, train_df)
    X_valid, y_valid = make_X_y(cfg, valid_df)

    # alinha colunas (caso dummies difiram entre train/valid)
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)

    results: List[Dict] = []

    # 1) LightGBM
    if LGBMRegressor is not None:
        model = LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=tcfg.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        results.append({
            "model": "lightgbm",
            "mae": float(mean_absolute_error(y_valid, pred)),
            "rmse": rmse(y_valid, pred),
            "mape": mape(y_valid, pred),
            "smape": smape(y_valid, pred),
        })
    else:
        logger.warning(
            "LightGBM não disponível no ambiente (pip install lightgbm).")

    # 2) XGBoost
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=tcfg.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        results.append({
            "model": "xgboost",
            "mae": float(mean_absolute_error(y_valid, pred)),
            "rmse": rmse(y_valid, pred),
            "mape": mape(y_valid, pred),
            "smape": smape(y_valid, pred),
        })
    else:
        logger.warning(
            "XGBoost não disponível no ambiente (pip install xgboost).")

    # 3) RandomForest (baseline)
    if RandomForestRegressor is not None:
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=tcfg.random_state,
            n_jobs=-1,
            max_depth=None,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        results.append({
            "model": "random_forest",
            "mae": float(mean_absolute_error(y_valid, pred)),
            "rmse": rmse(y_valid, pred),
            "mape": mape(y_valid, pred),
            "smape": smape(y_valid, pred),
        })
    else:
        logger.warning(
            "RandomForest não disponível (estranho; vem no scikit-learn).")

    if not results:
        raise RuntimeError(
            "Nenhum modelo foi treinado. Verifique dependências do ambiente.")

    leaderboard = pd.DataFrame(results).sort_values(
        ["rmse", "mae"], ascending=True).reset_index(drop=True)
    logger.info("Leaderboard:\n" + leaderboard.to_string(index=False))
    return leaderboard


@task
def persist_processed(cfg: DatasetConfig, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    out_dir = data_path(cfg.processed_dir)
    ensure_dir(out_dir)

    train_out = os.path.join(out_dir, "train.parquet")
    valid_out = os.path.join(out_dir, "valid.parquet")

    train_df.to_parquet(train_out, index=False)
    valid_df.to_parquet(valid_out, index=False)


# -----------------------------
# Flow
# -----------------------------
@flow(name="forecast_estoque_walmart_flow")
def main():
    cfg = DatasetConfig()
    tcfg = TrainConfig()

    df = build_dataset(cfg)
    train_df, valid_df = split_dataset(cfg, df)
    persist_processed(cfg, train_df, valid_df)
    leaderboard = train_and_compare(cfg, tcfg, train_df, valid_df)

    return leaderboard


if __name__ == "__main__":
    main()
