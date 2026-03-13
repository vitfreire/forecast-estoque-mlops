from __future__ import annotations

from typing import Optional, Dict

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

from .base import ModelMeta


class RFModel:
    def __init__(self):
        self.meta = ModelMeta(key="random_forest", name="RandomForest", maintenance_cost=0.4)
        self.estimator = RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
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
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
            registered_model_name=registered_model_name,
        )
        if extra_artifacts:
            for local_path, apath in extra_artifacts.items():
                mlflow.log_artifact(local_path, artifact_path=apath)