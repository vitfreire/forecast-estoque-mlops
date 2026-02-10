import pandas as pd
import numpy as np


def seasonal_naive_week(df_train: pd.DataFrame, df_valid: pd.DataFrame, group_cols=("Store", "Dept"), date_col="Date", target_col="Weekly_Sales"):
    """
    Baseline: usar o valor da mesma semana do ano anterior (52 semanas).
    Se não houver, usar mediana do grupo.
    """
    tr = df_train.sort_values(list(group_cols) + [date_col]).copy()
    tr["key_date"] = tr[date_col] + pd.Timedelta(weeks=52)

    lookup = tr[list(group_cols) + ["key_date", target_col]
                ].rename(columns={"key_date": date_col, target_col: "pred"})
    dv = df_valid.merge(lookup, on=list(group_cols) + [date_col], how="left")

    group_median = tr.groupby(list(group_cols))[target_col].median(
    ).reset_index().rename(columns={target_col: "median_group"})
    dv = dv.merge(group_median, on=list(group_cols), how="left")

    dv["pred"] = dv["pred"].fillna(
        dv["median_group"]).fillna(tr[target_col].median())
    return dv["pred"].values
