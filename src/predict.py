from __future__ import annotations

import json
import os
from typing import Dict, List, Any

import pandas as pd
import lightgbm as lgb


def _load_preprocess_metadata(artifacts_dir: str, model_name: str) -> Dict[str, Any]:
    """
    Tenta carregar metadata do preprocess.
    Se não existir, retorna estrutura vazia segura.
    """

    path = os.path.join(artifacts_dir, f"{model_name}_preprocess.json")

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_training_features() -> List[str]:
    """
    Carrega lista de features usadas no treinamento
    """

    path = "artifacts/metadata/features.json"

    if not os.path.exists(path):
        raise RuntimeError(f"features.json não encontrado: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    features = data.get("features")

    if features is None:
        raise RuntimeError("features.json inválido: chave 'features' não encontrada")

    return list(features)


def _convert_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Espelha conversão usada no treinamento
    """

    X = X.copy()

    dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]

    for col in dt_cols:

        s = X[col]

        X[f"{col}_year"] = s.dt.year.astype("int32")
        X[f"{col}_month"] = s.dt.month.astype("int8")
        X[f"{col}_week"] = s.dt.isocalendar().week.astype("int16")
        X[f"{col}_day"] = s.dt.day.astype("int8")
        X[f"{col}_dow"] = s.dt.dayofweek.astype("int8")
        X[f"{col}_dayofyear"] = s.dt.dayofyear.astype("int16")
        X[f"{col}_ts"] = (s.astype("int64") // 10**9).astype("int64")

        X = X.drop(columns=[col])

    return X


def _align_columns_to_training(
    X: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Garante que o dataframe tenha exatamente as features usadas no treino
    """

    if feature_names is None:
        raise RuntimeError("feature_names é None")

    X = X.copy()

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].copy()

    return X


def _infer_categorical_columns(X: pd.DataFrame) -> List[str]:
    """
    Detecta automaticamente colunas categóricas
    """

    return X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _apply_types_and_imputation(
    X: pd.DataFrame,
    feature_names: List[str],
    cat_cols: List[str],
    cat_mapping: Dict[str, List[str]],
) -> pd.DataFrame:

    X = X.copy()

    for col in feature_names:

        # força conversão para número
        X[col] = pd.to_numeric(X[col], errors="coerce")

        # preenche valores faltantes
        X[col] = X[col].fillna(0)

        # garante float32 (schema esperado pelo MLflow)
        X[col] = X[col].astype("float32")

    return X


def _enforce_mlflow_schema(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o dataframe respeite exatamente o schema salvo no MLflow
    """

    try:
        schema = model.metadata.get_input_schema()
    except Exception:
        return X

    if schema is None:
        return X

    X = X.copy()

    ordered_cols = []

    for col in schema.inputs:

        name = col.name
        dtype = str(col.type)

        ordered_cols.append(name)

        if name not in X.columns:
            X[name] = 0

        if dtype.startswith("float"):
            X[name] = pd.to_numeric(X[name], errors="coerce").fillna(0).astype("float32")

        elif dtype.startswith("integer"):
            X[name] = pd.to_numeric(X[name], errors="coerce").fillna(0).astype("int32")

        else:
            X[name] = X[name].astype("string")

    return X[ordered_cols]


def predict(
    model,
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    model_name: str = "lgbm_walmart_cost",
    artifacts_dir: str = "artifacts",
) -> pd.DataFrame:

    data = df.drop(columns=[target_col], errors="ignore").copy()

    meta = _load_preprocess_metadata(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
    )

    feature_names = _load_training_features()

    data = _convert_datetime_features(data)

    X = _align_columns_to_training(
        data,
        feature_names,
    )

    # -----------------------------
    # CONVERSÃO NUMÉRICA FORÇADA
    # -----------------------------
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # preencher NaN
    X = X.fillna(0)

    # garantir dtype correto para MLflow schema
    X = X.astype("float32")

    # -----------------------------
    # PREDICT
    # -----------------------------
    preds = model.predict(X)

    out = df.copy()
    out["y_pred"] = preds

    return out