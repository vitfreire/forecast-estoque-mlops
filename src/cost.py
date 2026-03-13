import numpy as np


def cost_error(y_true, y_pred, cost_under: float, cost_over: float) -> float:
    """
    Subprevisão: y_pred < y_true  => ruptura (custo_under por unidade)
    Superprevisão: y_pred > y_true => excesso (custo_over por unidade)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    under = np.maximum(y_true - y_pred, 0.0)
    over = np.maximum(y_pred - y_true, 0.0)

    return float(np.sum(under * cost_under + over * cost_over))
