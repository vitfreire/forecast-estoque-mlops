from __future__ import annotations

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np


class TabularPreprocessor:
    """
    Preprocessor robusto:
    - Remove target
    - Remove Date
    - Detecta categóricas automaticamente
    - One-hot encode
    - Garante consistência entre fit e transform
    """

    def __init__(self, target_col: str):
        self.target_col = target_col
        self.feature_columns: List[str] = []
        self.categorical_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        df = df.copy()

        y = df[self.target_col].astype("float32").to_numpy()
        X = df.drop(columns=[self.target_col])

        if "Date" in X.columns:
            X = X.drop(columns=["Date"])

        # detectar categóricas
        self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # one-hot
        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=False)

        self.feature_columns = X.columns.tolist()

        return X.astype("float32"), y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        X = df.drop(columns=[self.target_col])

        if "Date" in X.columns:
            X = X.drop(columns=["Date"])

        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=False)

        # alinhar colunas
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_columns]

        return X.astype("float32")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_col": self.target_col,
            "feature_columns": self.feature_columns,
            "categorical_cols": self.categorical_cols,
        }