from __future__ import annotations

from typing import Dict, List

from .lgbm import LGBMModel
from .xgb import XGBModel
from .rf import RFModel


def build_model_registry() -> Dict[str, object]:
    return {
        "lightgbm": LGBMModel(),
        "xgboost": XGBModel(),
        "random_forest": RFModel(),
    }


def parse_run_models(env_value: str) -> List[str]:
    allowed = {"lightgbm", "xgboost", "random_forest"}
    parts = [p.strip().lower() for p in (env_value or "").split(",") if p.strip()]
    models = [p for p in parts if p in allowed]
    return models if models else ["lightgbm"]