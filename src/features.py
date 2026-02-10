import pandas as pd
import numpy as np


def _to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col])
    return df


def merge_walmart(train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on=["Store"], how="left")

    # Resolver IsHoliday de forma robusta
    if "IsHoliday_y" in df.columns:
        df["IsHoliday"] = df["IsHoliday_y"]
    elif "IsHoliday_x" in df.columns:
        df["IsHoliday"] = df["IsHoliday_x"]
    elif "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"]
    else:
        df["IsHoliday"] = 0  # fallback seguro

    df["IsHoliday"] = df["IsHoliday"].fillna(0).astype(int)

    # limpar colunas duplicadas
    drop_cols = [c for c in ["IsHoliday_x", "IsHoliday_y"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.month
    d["weekofyear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["dayofweek"] = d["Date"].dt.dayofweek
    d["is_month_start"] = d["Date"].dt.is_month_start.astype(int)
    d["is_month_end"] = d["Date"].dt.is_month_end.astype(int)
    return d


def add_lag_features(
    df: pd.DataFrame,
    group_cols=("Store", "Dept"),
    target_col="Weekly_Sales",
    lags=(1, 2, 3, 4),
    windows=(4, 8, 12),
) -> pd.DataFrame:
    d = df.copy()

    # garantir ordenação correta
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values(list(group_cols) + ["Date"])

    g = d.groupby(list(group_cols), sort=False)[target_col]

    # lags
    for lag in lags:
        d[f"lag_{lag}"] = g.shift(lag)

    # rolling stats (sempre usando shift(1) para não vazar o futuro)
    for w in windows:
        d[f"roll_mean_{w}"] = g.shift(1).transform(
            lambda s: s.rolling(w).mean())
        d[f"roll_std_{w}"] = g.shift(1).transform(lambda s: s.rolling(w).std())

    # opcional: estabilizar NaNs iniciais
    d = d.replace([np.inf, -np.inf], np.nan)

    return d


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # preencher variáveis macro com medianas por loja (simples e robusto)
    macro_cols = ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2",
                  "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"]
    for c in macro_cols:
        if c in d.columns:
            d[c] = d.groupby("Store")[c].transform(
                lambda s: s.fillna(s.median()))
            d[c] = d[c].fillna(d[c].median())

    # lags/rolling podem ficar NaN no início -> vamos remover linhas sem histórico mínimo
    feature_cols = [c for c in d.columns if c.startswith(
        "lag_") or c.startswith("roll_")]
    d = d.dropna(subset=feature_cols)

    # clip de target para evitar negativos (segurança)
    d["Weekly_Sales"] = d["Weekly_Sales"].clip(lower=0.0)

    return d
