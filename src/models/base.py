from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


# =========================
# Contexto de treino
# =========================
@dataclass
class FitContext:
    target_col: str = "Weekly_Sales"
    date_col: str = "Date"
    group_cols: Optional[list[str]] = None


# =========================
# Metadados do modelo
# =========================
@dataclass
class ModelMeta:
    name: str
    maintenance_cost: float
    estimated_cost: float = 0.5
    complexity_score: float = 0.5
    training_time_score: float = 0.5


# =========================
# Classe base
# =========================
class BaseModel:
    meta: ModelMeta

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        ctx: FitContext,
    ) -> Any:
        raise NotImplementedError

    def predict(
        self,
        model: Any,
        df: pd.DataFrame,
        ctx: FitContext,
    ) -> np.ndarray:
        raise NotImplementedError

    def extra_artifacts(self) -> Dict[str, str]:
        return {}
