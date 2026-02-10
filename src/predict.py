from __future__ import annotations

import json
import os
from typing import Dict, List, Any

import lightgbm as lgb
import pandas as pd


def _load_preprocess_metadata(artifacts_dir: str, model_name: str) -> Dict[str, Any]:
    path = os.path.join(artifacts_dir, f"{model_name}_preprocess.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata não encontrada: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Deve espelhar exatamente a conversão do treino.
    Converte datetime64 em colunas numéricas e remove a coluna original.
    """
    X = X.copy()

    dt_cols = [
        c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    for col in dt_cols:
        s = X[col]

        X[f"{col}_year"] = s.dt.year.astype("int32")
        X[f"{col}_month"] = s.dt.month.astype("int8")
        X[f"{col}_week"] = s.dt.isocalendar().week.astype("int16")
        X[f"{col}_day"] = s.dt.day.astype("int8")
        X[f"{col}_dow"] = s.dt.dayofweek.astype("int8")
        X[f"{col}_dayofyear"] = s.dt.dayofyear.astype("int16")
        X[f"{col}_ts"] = (s.view("int64") // 10**9).astype("int64")

        X = X.drop(columns=[col])

    return X


def _align_columns_to_training(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    - Cria features faltantes com 0
    - Remove extras
    - Reordena exatamente como no treino
    """
    X = X.copy()

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].copy()
    return X


def _apply_types_and_imputation(
    X: pd.DataFrame,
    feature_names: List[str],
    cat_cols: List[str],
    cat_mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Aplica:
    - category nas colunas categóricas + set_categories do treino + fillna('missing')
    - fillna(0) para numéricas
    - se alguma coluna não categórica vier como object, tenta converter para numérico; se não der, falha.
    """
    X = X.copy()

    for col in feature_names:
        if col in cat_cols:
            X[col] = X[col].astype("category")
            X[col] = X[col].cat.set_categories(cat_mapping[col])
            X[col] = X[col].fillna("missing")
        else:
            # numéricas
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(0)
            else:
                # tenta coerção segura para numérico (string numérica, etc.)
                coerced = pd.to_numeric(X[col], errors="coerce")
                if coerced.isna().all() and not X[col].isna().all():
                    raise ValueError(
                        f"Coluna '{col}' veio como dtype={X[col].dtype} no predict e não é categórica no treino. "
                        f"Valores não numéricos detectados. Corrija no build_dataset."
                    )
                X[col] = coerced.fillna(0)

    return X


def predict(
    model: lgb.Booster,
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    model_name: str = "lgbm_walmart_cost",
    artifacts_dir: str = "artifacts",
) -> pd.DataFrame:
    # remove target se vier junto
    data = df.drop(columns=[target_col], errors="ignore").copy()

    meta = _load_preprocess_metadata(
        artifacts_dir=artifacts_dir, model_name=model_name)

    # 1) espelha a conversão de datetime do treino
    data = _convert_datetime_features(data)

    # 2) alinha colunas do treino (cria faltantes, remove extras, reordena)
    X = _align_columns_to_training(data, meta["feature_names"])

    # 3) aplica tipos e imputação com mapping de categorias do treino
    X = _apply_types_and_imputation(
        X=X,
        feature_names=meta["feature_names"],
        cat_cols=meta["cat_cols"],
        cat_mapping=meta["cat_mapping"],
    )

    preds = model.predict(X)

    out = df.copy()
    out["y_pred"] = preds
    return out
