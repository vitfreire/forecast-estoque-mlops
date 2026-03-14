import numpy as np
import pytest

from src.metrics import mae, rmse, smape


# ── MAE ──────────────────────────────────────────────────────────────────────

def test_mae_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mae(y_true, y_pred) == 0.0


def test_mae_known_value():
    # |1-2| + |2-2| + |3-2| = 1 + 0 + 1 → mean = 2/3
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)


def test_mae_always_nonnegative():
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 100, 200)
    y_pred = rng.normal(0, 100, 200)
    assert mae(y_true, y_pred) >= 0.0


# ── RMSE ─────────────────────────────────────────────────────────────────────

def test_rmse_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0


def test_rmse_known_value():
    # sqrt(mean([1, 1])) = 1.0
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_greater_or_equal_mae():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 500, 300)
    y_pred = rng.uniform(0, 500, 300)
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


# ── SMAPE ─────────────────────────────────────────────────────────────────────

def test_smape_perfect_prediction():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert smape(y_true, y_pred) == pytest.approx(0.0)


def test_smape_range():
    # SMAPE está entre 0 e 200 por definição
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([50.0, 300.0, 0.0])
    result = smape(y_true, y_pred)
    assert 0.0 <= result <= 200.0


def test_smape_symmetric_property():
    # SMAPE é simétrico: trocar y_true e y_pred dá o mesmo resultado
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([80.0, 250.0])
    assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))
