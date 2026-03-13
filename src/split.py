import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


def temporal_split(df: pd.DataFrame, date_col: str, horizon_days: int):
    df = df.sort_values(date_col)
    last_date = pd.to_datetime(df[date_col]).max()

    cutoff = last_date - pd.Timedelta(days=horizon_days)
    train_df = df[pd.to_datetime(df[date_col]) <= cutoff].copy()
    valid_df = df[pd.to_datetime(df[date_col]) > cutoff].copy()

    return train_df, valid_df, cutoff


@dataclass(frozen=True)
class RollingFold:
    fold: int
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def rolling_splits(
    df: pd.DataFrame,
    date_col: str,
    horizon_days: int,
    n_folds: int = 3,
    step_days: int | None = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, RollingFold]]:
    """
    Rolling backtest:
    - Usa as últimas janelas de validação
    - Cada fold valida um intervalo de horizon_days
    - step_days define quanto anda para trás entre folds (default = horizon_days)
    """
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    last_date = df[date_col].max()
    if step_days is None:
        step_days = horizon_days

    splits = []
    for i in range(n_folds):
        valid_end = last_date - pd.Timedelta(days=i * step_days)
        valid_start = valid_end - pd.Timedelta(days=horizon_days)
        train_end = valid_start

        train_df = df[df[date_col] <= train_end].copy()
        valid_df = df[(df[date_col] > valid_start) & (df[date_col] <= valid_end)].copy()

        fold_info = RollingFold(
            fold=i + 1,
            train_end=train_end,
            valid_start=valid_start + pd.Timedelta(days=1),
            valid_end=valid_end,
        )

        splits.append((train_df, valid_df, fold_info))

    splits.reverse()  # fold 1 mais antigo primeiro
    return splits