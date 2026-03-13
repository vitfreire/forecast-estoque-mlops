from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Optional, Dict


@dataclass(frozen=True)
class ModelMeta:
    key: str
    name: str
    maintenance_cost: float


@dataclass(frozen=True)
class FitContext:
    target_col: str = "Weekly_Sales"


class BaseModel(Protocol):
    meta: ModelMeta

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> Any:
        ...

    def predict(self, model: Any, X) -> Any:
        ...

    def log_to_mlflow(
        self,
        model: Any,
        artifact_path: str,
        input_example,
        signature,
        extra_artifacts: Optional[Dict[str, str]] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        ...