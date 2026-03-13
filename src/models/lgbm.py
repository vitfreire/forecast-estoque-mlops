from __future__ import annotations

from typing import Any, Optional, Dict

import lightgbm as lgb
import mlflow
import mlflow.lightgbm


from .base import ModelMeta


class LGBMModel:
    def __init__(self):
        self.meta = ModelMeta(key="lightgbm", name="LightGBM", maintenance_cost=3.0)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_sets = [train_set]
        valid_names = ["train"]

        callbacks = [lgb.log_evaluation(period=100)]
        if X_valid is not None and y_valid is not None:
            valid_set = lgb.Dataset(X_valid, label=y_valid, free_raw_data=False)
            valid_sets.append(valid_set)
            valid_names.append("valid")
            callbacks = [
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100),
            ]

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
            "verbosity": -1,
            "seed": 42,
        }

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=3000,
            callbacks=callbacks,
        )
        return model

    def predict(self, model: lgb.Booster, X):
        return model.predict(X, num_iteration=model.best_iteration or model.current_iteration())

    def log_to_mlflow(
        self,
        model: lgb.Booster,
        artifact_path: str,
        input_example,
        signature,
        extra_artifacts: Optional[Dict[str, str]] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
            registered_model_name=registered_model_name,
            extra_pip_requirements=None,
        )
        if extra_artifacts:
            for local_path, apath in extra_artifacts.items():
                mlflow.log_artifact(local_path, artifact_path=apath)