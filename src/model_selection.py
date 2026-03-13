from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class FitContext:
    """
    Contexto padrão para todos os modelos (para não quebrar pipeline).
    """
    target_col: str = "Weekly_Sales"
    date_col: str = "Date"
    group_cols: Optional[list[str]] = None


@dataclass
class ModelMeta:
    name: str
    maintenance_cost: float  # custo relativo de manter (0..1+)


class BaseModel:
    meta: ModelMeta

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, ctx: FitContext) -> Any:
        raise NotImplementedError

    def predict(self, model: Any, df: pd.DataFrame, ctx: FitContext) -> np.ndarray:
        raise NotImplementedError

    def extra_artifacts(self) -> Dict[str, str]:
        """
        Retorna caminhos locais de artefatos extras a logar no MLflow.
        """
        return {}
