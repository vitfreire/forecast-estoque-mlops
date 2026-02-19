import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

from src.models.base import BaseModel, ModelMeta, FitContext


class LGBMForecastModel(BaseModel):

    def __init__(self):

        self.meta = ModelMeta(
            name="LightGBM",
            maintenance_cost=3.0
        )

        self.model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, ctx: FitContext):

        X_train = train_df.drop(columns=[ctx.target_col])
        y_train = train_df[ctx.target_col]

        X_valid = valid_df.drop(columns=[ctx.target_col])
        y_valid = valid_df[ctx.target_col]

        # Garantir que tudo é numérico
        X_train = X_train.select_dtypes(include=[np.number])
        X_valid = X_valid.select_dtypes(include=[np.number])

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
        )

        return self.model

    def predict(self, model, df: pd.DataFrame, ctx: FitContext):

        X = df.drop(columns=[ctx.target_col])
        X = X.select_dtypes(include=[np.number])

        preds = model.predict(X)

        return preds
