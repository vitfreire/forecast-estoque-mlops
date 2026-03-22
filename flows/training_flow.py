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
    Seleciona o melhor modelo pela média do RMSE em todos os folds (não pelo melhor fold isolado).
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
    all_fold_records: List[Dict[str, Any]] = []

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
            "leaderboard": res.leaderboard.to_dict(orient="records"),
        })

        for row in res.leaderboard.to_dict(orient="records"):
            all_fold_records.append({**row, "fold": fold.fold})

    # -------------------------------------------------------
    # Seleção do melhor modelo pela MÉDIA do RMSE nos folds
    # Evita escolher modelo que venceu apenas um fold fácil
    # -------------------------------------------------------
    df_all = pd.DataFrame(all_fold_records)
    avg_lb = (
        df_all.groupby("model", as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), smape=("smape", "mean"),
             train_seconds=("train_seconds", "mean"))
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    best_model_name = avg_lb.iloc[0]["model"]

    # URI do modelo mais recente (fold com maior índice = validação mais recente)
    recent_fold = max(fold_summaries, key=lambda s: s["fold"])
    recent_lb = pd.DataFrame(recent_fold["leaderboard"])
    best_row_recent = recent_lb[recent_lb["model"] == best_model_name]
    if best_row_recent.empty:
        best_row_recent = recent_lb.iloc[[0]]
    best_model_uri = best_row_recent.iloc[0]["model_uri"]
    best_run_id = best_row_recent.iloc[0]["run_id"]

    # Anota run_id / model_uri do fold mais recente para cada modelo (para o dashboard)
    uri_map = {r["model"]: r["model_uri"] for r in recent_fold["leaderboard"]}
    run_map = {r["model"]: r["run_id"] for r in recent_fold["leaderboard"]}
    avg_lb["model_uri"] = avg_lb["model"].map(uri_map)
    avg_lb["run_id"] = avg_lb["model"].map(run_map)

    # Grava leaderboard final com métricas médias (substitui o leaderboard por fold)
    from src.io import ensure_dir
    lb_path = "artifacts/reports/leaderboard.csv"
    ensure_dir(os.path.dirname(lb_path))
    avg_lb.to_csv(lb_path, index=False)
    logger.info(f"Leaderboard médio salvo: {lb_path}")

    best_overall = avg_lb.iloc[0].to_dict()
    best_overall["model_uri"] = best_model_uri
    best_overall["run_id"] = best_run_id

    logger.info(
        f"Melhor modelo (média de {n_folds} folds): {best_model_name} "
        f"| RMSE={best_overall['rmse']:.1f} | MAE={best_overall['mae']:.1f} "
        f"| SMAPE={best_overall['smape']:.2f}%"
    )

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