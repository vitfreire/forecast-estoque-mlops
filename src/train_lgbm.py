from __future__ import annotations

import json
import os
from typing import Tuple, List, Dict

import lightgbm as lgb
import pandas as pd


def _split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    return X, y


def _convert_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    LightGBM não aceita datetime64 diretamente.
    Converte datetime em features numéricas e remove a coluna original.
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

        # timestamp contínuo em segundos (para tendência)
        X[f"{col}_ts"] = (s.view("int64") // 10**9).astype("int64")

        X = X.drop(columns=[col])

    return X


def _align_columns(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Garante que train/valid tenham o mesmo conjunto e ordem de colunas.
    - Colunas faltantes são criadas com 0.
    """
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    for col in X_train.columns:
        if col not in X_valid.columns:
            X_valid[col] = 0

    for col in X_valid.columns:
        if col not in X_train.columns:
            X_train[col] = 0

    X_valid = X_valid[X_train.columns]
    return X_train, X_valid


def _coerce_types_and_impute(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    - object -> category
    - alinha categorias entre train/valid com universo união + "missing"
    - datetime já deve ter sido convertido antes de chegar aqui
    - NaN numéricas -> 0
    - NaN categóricas -> "missing"
    Retorna:
      X_train, X_valid, cat_cols, cat_mapping
    """
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    cat_cols: List[str] = []
    cat_mapping: Dict[str, List[str]] = {}

    for col in X_train.columns:
        # Proteção: datetime não pode chegar aqui
        if pd.api.types.is_datetime64_any_dtype(X_train[col]) or pd.api.types.is_datetime64_any_dtype(X_valid[col]):
            raise ValueError(
                f"Coluna datetime detectada após conversão: {col}. Converta antes com _convert_datetime_features().")

        # Se veio object em qualquer um, vira category nos dois
        if X_train[col].dtype == "object" or X_valid[col].dtype == "object":
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")

        # Se é category em qualquer um, trata como categórica
        if X_train[col].dtype.name == "category" or X_valid[col].dtype.name == "category":
            if X_train[col].dtype.name != "category":
                X_train[col] = X_train[col].astype("category")
            if X_valid[col].dtype.name != "category":
                X_valid[col] = X_valid[col].astype("category")

            cat_cols.append(col)

            train_cats = list(X_train[col].cat.categories)
            valid_cats = list(X_valid[col].cat.categories)
            union_cats = sorted(
                set(train_cats) | set(valid_cats) | {"missing"})

            X_train[col] = X_train[col].cat.set_categories(union_cats)
            X_valid[col] = X_valid[col].cat.set_categories(union_cats)

            X_train[col] = X_train[col].fillna("missing")
            X_valid[col] = X_valid[col].fillna("missing")

            cat_mapping[col] = union_cats

        else:
            # Numéricas: garante numérico + fillna(0)
            if pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col] = X_train[col].fillna(0)
            else:
                # Se sobrou algum tipo estranho (ex: string escondida), força para category
                X_train[col] = X_train[col].astype("category")
                X_valid[col] = X_valid[col].astype("category")

                cat_cols.append(col)
                union_cats = sorted(set(list(X_train[col].cat.categories)) | set(
                    list(X_valid[col].cat.categories)) | {"missing"})
                X_train[col] = X_train[col].cat.set_categories(
                    union_cats).fillna("missing")
                X_valid[col] = X_valid[col].cat.set_categories(
                    union_cats).fillna("missing")
                cat_mapping[col] = union_cats

            if pd.api.types.is_numeric_dtype(X_valid[col]):
                X_valid[col] = X_valid[col].fillna(0)

    return X_train, X_valid, cat_cols, cat_mapping


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_preprocess_metadata(
    artifacts_dir: str,
    model_name: str,
    feature_names: List[str],
    cat_cols: List[str],
    cat_mapping: Dict[str, List[str]],
) -> str:
    _ensure_dir(artifacts_dir)

    payload = {
        "model_name": model_name,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "cat_mapping": cat_mapping,
    }

    out_path = os.path.join(artifacts_dir, f"{model_name}_preprocess.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def train_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str = "lgbm_model",
    target_col: str = "Weekly_Sales",
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    artifacts_dir: str = "artifacts",
) -> lgb.Booster:
    """
    Treina LightGBM via lgb.train com:
    - conversão de datetime -> numérico
    - suporte a categóricas (pandas category)
    - imputação (numéricas=0, categóricas="missing")
    - early stopping via callbacks
    - salva metadata de pré-processamento em JSON
    """
    # Split
    X_train, y_train = _split_xy(train_df, target_col)
    X_valid, y_valid = _split_xy(valid_df, target_col)

    # Converte datetime
    X_train = _convert_datetime_features(X_train)
    X_valid = _convert_datetime_features(X_valid)

    # Alinha colunas
    X_train, X_valid = _align_columns(X_train, X_valid)

    # Tipos + imputação + categóricas alinhadas
    X_train, X_valid, cat_cols, cat_mapping = _coerce_types_and_impute(
        X_train, X_valid)

    # Datasets
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
        free_raw_data=False,
    )

    params = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbosity": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )

    # Metadata para inferência consistente
    feature_names = list(X_train.columns)
    meta_path = _save_preprocess_metadata(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        feature_names=feature_names,
        cat_cols=cat_cols,
        cat_mapping=cat_mapping,
    )

    # Salva o modelo também (boa prática)
    _ensure_dir(artifacts_dir)
    model_path = os.path.join(artifacts_dir, f"{model_name}.txt")
    model.save_model(model_path)

    return model
