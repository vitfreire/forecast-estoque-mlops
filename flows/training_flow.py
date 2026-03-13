from __future__ import annotations

import os
from typing import Dict, Any, List

import pandas as pd
import mlflow
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv

from src.config import settings
from src.io import read_raw, write_parquet, ensure_dir, read_parquet
from src.features import merge_walmart, add_time_features, add_lag_features, finalize_features
from src.split import temporal_split, rolling_splits
from src.models.trainer import train_compare_and_log
from src.mlflow_utils import setup_mlflow


def _is_dev() -> bool:
    return os.getenv("APP_ENV", "prod").lower() == "dev"


@task
def build_dataset_task() -> str:
    logger = get_run_logger()

    raw = read_raw(settings.DATA_RAW_DIR)

    df = merge_walmart(raw["train"], raw["features"], raw["stores"])
    df = add_time_features(df)
    df = add_lag_features(df, ["Store", "Dept"], "Weekly_Sales")
    df = finalize_features(df)

    # DEV speed-up
    if _is_dev():
        max_rows = int(os.getenv("DEV_MAX_ROWS", "20000"))
        max_stores = int(os.getenv("DEV_MAX_STORES", "5"))
        max_depts = int(os.getenv("DEV_MAX_DEPTS", "5"))

        df = df.sort_values("Date").copy()
        stores = df["Store"].dropna().unique().tolist()[:max_stores]
        df = df[df["Store"].isin(stores)]
        depts = df["Dept"].dropna().unique().tolist()[:max_depts]
        df = df[df["Dept"].isin(depts)]
        if len(df) > max_rows:
            df = df.iloc[-max_rows:].copy()

        logger.info(f"[DEV] dataset reduzido: rows={len(df)} stores={len(stores)} depts={len(depts)}")

    ensure_dir(settings.DATA_PROCESSED_DIR)
    path = os.path.join(settings.DATA_PROCESSED_DIR, "dataset.parquet")
    write_parquet(df, path)

    logger.info(f"Dataset salvo: {path} | shape={df.shape} | ENV={os.getenv('APP_ENV','prod')}")
    return path


@task
def validate_split_task(dataset_path: str) -> Dict[str, Any]:
    df = read_parquet(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"])

    horizon = int(os.getenv("HORIZON_DAYS", str(settings.HORIZON_DAYS)))
    train_df, valid_df, cutoff = temporal_split(df, "Date", horizon)

    return {
        "min_date": str(df["Date"].min().date()),
        "max_date": str(df["Date"].max().date()),
        "cutoff": str(cutoff.date()),
        "train_max": str(train_df["Date"].max().date()),
        "valid_min": str(valid_df["Date"].min().date()),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "horizon_days": horizon,
    }


@task
def train_holdout_task(dataset_path: str) -> Dict[str, Any]:
    df = read_parquet(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"])

    horizon = int(os.getenv("HORIZON_DAYS", str(settings.HORIZON_DAYS)))
    train_df, valid_df, cutoff = temporal_split(df, "Date", horizon)

    res = train_compare_and_log(
        train_df=train_df,
        valid_df=valid_df,
        target_col="Weekly_Sales",
        fold_info={"type": "holdout", "cutoff": str(cutoff.date())},
    )

    return {
        "best_model": res.best_key,
        "best_run_id": res.best_run_id,
        "best_model_uri": res.best_model_uri,
        "leaderboard": res.leaderboard.to_dict(orient="records"),
    }


@task
def train_rolling_task(dataset_path: str) -> Dict[str, Any]:
    """
    Rolling Time Series Cross Validation.
    Cria um run global no MLflow e dentro dele executa folds.
    """
    logger = get_run_logger()

    setup_mlflow()

    df = read_parquet(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"])

    horizon = int(os.getenv("HORIZON_DAYS", str(settings.HORIZON_DAYS)))
    n_folds = int(os.getenv("N_FOLDS", "3"))

    logger.info(f"Rolling CV iniciado | horizon={horizon} | folds={n_folds}")

    splits = rolling_splits(df, "Date", horizon_days=horizon, n_folds=n_folds)

    fold_summaries = []
    best_overall = None

    for train_df, valid_df, fold in splits:

        logger.info(f"Treinando fold {fold.fold}")

        res = train_compare_and_log(
            train_df=train_df,
            valid_df=valid_df,
            target_col="Weekly_Sales",
            fold_info={
                "type": "rolling",
                "fold": fold.fold,
                "train_end": str(fold.train_end.date()),
                "valid_start": str(fold.valid_start.date()),
                "valid_end": str(fold.valid_end.date()),
            },
        )

        fold_summaries.append({
            "fold": fold.fold,
            "best_model": res.best_key,
            "best_model_uri": res.best_model_uri,
            "leaderboard": res.leaderboard.to_dict(orient="records"),
        })

        best_row = res.leaderboard.iloc[0].to_dict()
        best_row["fold"] = fold.fold

        if best_overall is None or best_row["rmse"] < best_overall["rmse"]:
            best_overall = best_row

    logger.info(f"Melhor modelo global: {best_overall}")

    return {
        "horizon_days": horizon,
        "n_folds": n_folds,
        "best_overall": best_overall,
        "folds": fold_summaries,
    }


@flow(name="training_flow_walmart_multimodel")
def main():
    load_dotenv()
    logger = get_run_logger()

    dataset_path = build_dataset_task()

    split_info = validate_split_task(dataset_path)
    logger.info(f"Split check: {split_info}")

    mode = os.getenv("EVAL_MODE", "holdout").lower().strip()

    if mode == "rolling":
        logger.info("Modo de avaliação: rolling time series CV")
        out = train_rolling_task(dataset_path)
    else:
        logger.info("Modo de avaliação: holdout")
        out = train_holdout_task(dataset_path)

    return {"split": split_info, "result": out}


if __name__ == "__main__":
    main()