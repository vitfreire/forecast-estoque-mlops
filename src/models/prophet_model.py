from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet

from src.models.base import BaseModel, ModelMeta, FitContext


class ProphetForecastModel(BaseModel):

    meta = ModelMeta(
        name="prophet",
        maintenance_cost=0.5  # custo relativo (0-1)
    )

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        ctx: FitContext,
    ) -> Prophet:

        df = train_df.copy()

        df = df.rename(columns={
            ctx.date_col: "ds",
            ctx.target_col: "y"
        })

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        model.fit(df[["ds", "y"]])

        return model

    def predict(
        self,
        model: Prophet,
        df: pd.DataFrame,
        ctx: FitContext,
    ) -> np.ndarray:

        data = df.copy()
        data = data.rename(columns={ctx.date_col: "ds"})

        forecast = model.predict(data[["ds"]])

        return forecast["yhat"].values
