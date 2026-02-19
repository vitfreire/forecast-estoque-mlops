from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseModel, ModelMeta, FitContext


class RFForecastModel(BaseModel):
    def __init__(self):
        self.meta = ModelMeta(
            name="RandomForest",
            maintenance_cost=0.4,  # menor custo de manutenção
        )

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        ctx: FitContext,
    ):

        X_train = train_df.drop(columns=[ctx.target_col])
        y_train = train_df[ctx.target_col]

        self.model.fit(X_train, y_train)

        return self.model

    def predict(
        self,
        model,
        df: pd.DataFrame,
        ctx: FitContext,
    ) -> np.ndarray:

        X = df.drop(columns=[ctx.target_col], errors="ignore")
        return model.predict(X)
