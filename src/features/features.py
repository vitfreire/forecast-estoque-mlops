import pandas as pd
import numpy as np


# ==========================================================
# Merge principal
# ==========================================================
def merge_walmart(train: pd.DataFrame,
                   features: pd.DataFrame,
                   stores: pd.DataFrame) -> pd.DataFrame:

    train = train.copy()
    features = features.copy()
    stores = stores.copy()

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on=["Store"], how="left")

    # Resolver IsHoliday robustamente
    if "IsHoliday_y" in df.columns:
        df["IsHoliday"] = df["IsHoliday_y"]
    elif "IsHoliday_x" in df.columns:
        df["IsHoliday"] = df["IsHoliday_x"]
    else:
        df["IsHoliday"] = df.get("IsHoliday", 0)

    df["IsHoliday"] = df["IsHoliday"].fillna(0).astype(int)

    # remover duplicadas
    drop_cols = [c for c in ["IsHoliday_x", "IsHoliday_y"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ==========================================================
# Time Features
# ==========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values(["Store", "Dept", "Date"])

    d["year"] = d["Date"].dt.year.astype("int32")
    d["month"] = d["Date"].dt.month.astype("int8")
    d["weekofyear"] = d["Date"].dt.isocalendar().week.astype("int16")
    d["dayofweek"] = d["Date"].dt.dayofweek.astype("int8")

    d["is_month_start"] = d["Date"].dt.is_month_start.astype(int)
    d["is_month_end"] = d["Date"].dt.is_month_end.astype(int)

    # Encoding cíclico para sazonalidade
    d["week_sin"] = np.sin(2 * np.pi * d["weekofyear"] / 52)
    d["week_cos"] = np.cos(2 * np.pi * d["weekofyear"] / 52)

    return d


# ==========================================================
# Lag + Rolling (seguro e mais rápido)
# ==========================================================
def add_lag_features(
    df: pd.DataFrame,
    group_cols=("Store", "Dept"),
    target_col="Weekly_Sales",
    lags=(1, 2, 3, 4),
    windows=(4, 8, 12),
) -> pd.DataFrame:

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values(list(group_cols) + ["Date"])

    g = d.groupby(list(group_cols), sort=False)[target_col]

    # ------------------
    # LAGS
    # ------------------
    for lag in lags:
        d[f"lag_{lag}"] = g.shift(lag)

    # ------------------
    # Rolling stats 
    # ------------------
    shifted = g.shift(1)

    for w in windows:
        d[f"roll_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        d[f"roll_std_{w}"] = shifted.rolling(window=w, min_periods=1).std()

    # limpar infinitos
    d = d.replace([np.inf, -np.inf], np.nan)

    return d


# ==========================================================
# Finalização 
# ==========================================================
def finalize_features(df: pd.DataFrame) -> pd.DataFrame:

    d = df.copy()

    # ------------------
    # Preenchimento macroeconômico 
    # ------------------
    macro_cols = [
        "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3",
        "MarkDown4", "MarkDown5",
        "CPI", "Unemployment"
    ]

    for c in macro_cols:
        if c in d.columns:
            # mediana por loja
            d[c] = d.groupby("Store")[c].transform(lambda s: s.fillna(s.median()))
            d[c] = d[c].fillna(d[c].median())

    # ------------------
    # Lags / rollings
    # ------------------
    feature_cols = [c for c in d.columns if c.startswith("lag_") or c.startswith("roll_")]

    # Em vez de remover tudo, removemos apenas primeiras semanas sem histórico mínimo
    # Mantemos pelo menos lag_1 válido
    if "lag_1" in d.columns:
        d = d[d["lag_1"].notna()]

    # preencher resto com 0
    d[feature_cols] = d[feature_cols].fillna(0.0)

    # ------------------
    # Target 
    # ------------------
    d["Weekly_Sales"] = d["Weekly_Sales"].clip(lower=0.0)

    return d