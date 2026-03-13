import pandas as pd
from typing import Tuple, Dict, Any


def prepare_features(
    train_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constrói o dataset final mesclando train, stores e features.
    Retorna:
        - DataFrame final pronto
        - Dicionário preprocess com metadados necessários
    """

    # ========================
    # Merge principal
    # ========================
    df = train_df.copy()

    df = df.merge(stores_df, on="Store", how="left")
    df = df.merge(features_df, on=["Store", "Date"], how="left")

    # ========================
    # Tratamentos básicos
    # ========================
    df["Date"] = pd.to_datetime(df["Date"])

    # Exemplo de feature temporal
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    # ========================
    # Separação target
    # ========================
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # ========================
    # Identificação de categorias
    # ========================
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype("category")

    feature_names = X.columns.tolist()

    preprocess = {
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "target_col": target_col,
    }

    # Reconstrói DF final
    if y is not None:
        final_df = pd.concat([X, y], axis=1)
    else:
        final_df = X

    return final_df, preprocess
