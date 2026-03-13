from __future__ import annotations

from typing import List, Optional
import mlflow
import pandas as pd


def list_experiments():
    """
    Lista todos os experimentos do MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    return client.search_experiments()


def search_runs(
    experiment_ids: List[str],
    filter_string: str = "",
    max_results: int = 1000,
):
    """
    Busca runs no MLflow.
    """
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    return runs


def runs_to_dataframe(runs):
    """
    Converte runs MLflow em DataFrame.
    """

    rows = []

    for run in runs:

        row = {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

        # metrics
        for k, v in run.data.metrics.items():
            row[f"metrics.{k}"] = v

        # params
        for k, v in run.data.params.items():
            row[f"params.{k}"] = v

        # tags
        for k, v in run.data.tags.items():
            row[f"tags.{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def pick_best_run(
    df: pd.DataFrame,
    metric: str,
):
    """
    Seleciona melhor run baseado em métrica.
    """

    if metric not in df.columns:
        return None

    df = df.dropna(subset=[metric])

    if df.empty:
        return None

    best = df.sort_values(metric).iloc[0]

    return best


def get_run_detail(run_id: str):
    """
    Retorna detalhes de um run específico.
    """

    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)

    return run