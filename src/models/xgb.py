from __future__ import annotations

from typing import Optional, Dict

import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor

from .base import ModelMeta


class XGBModel:
    def __init__(self):
        self.meta = ModelMeta(key="xgboost", name="XGBoost", maintenance_cost=2.0)
        self.estimator = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is not None and y_valid is not None:
            self.estimator.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        else:
            self.estimator.fit(X_train, y_train)
        return self.estimator

    def predict(self, model, X):
        return model.predict(X)

    def log_to_mlflow(
        self,
        model,
        artifact_path: str,
        input_example,
        signature,
        extra_artifacts: Optional[Dict[str, str]] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        # evita warning do formato binário: salva JSON internamente via MLflow
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
            registered_model_name=registered_model_name,
        )
        if extra_artifacts:
            for local_path, apath in extra_artifacts.items():
                mlflow.log_artifact(local_path, artifact_path=apath)