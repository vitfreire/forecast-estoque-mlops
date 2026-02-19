from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel, ModelMeta, FitContext


class XGBForecastModel(BaseModel):
    def __init__(self):
        self.meta = ModelMeta(
            name="XGBoost",
            maintenance_cost=0.7,
        )

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        ctx: FitContext,
    ):

        X_train = train_df.drop(columns=[ctx.target_col])
        y_train = train_df[ctx.target_col]

        X_valid = valid_df.drop(columns=[ctx.target_col])
        y_valid = valid_df[ctx.target_col]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        return self.model

    def predict(
        self,
        model,
        df: pd.DataFrame,
        ctx: FitContext,
    ) -> np.ndarray:

        X = df.drop(columns=[ctx.target_col], errors="ignore")
        return model.predict(X)
