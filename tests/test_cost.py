import numpy as np
import pytest

from src.cost import cost_error


def test_cost_zero_when_perfect():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert cost_error(y_true, y_pred, cost_under=6.0, cost_over=1.5) == 0.0


def test_cost_under_only():
    # y_pred < y_true: só subprevisão
    y_true = np.array([100.0])
    y_pred = np.array([80.0])
    # under = 20 unidades * cost_under=6 = 120
    result = cost_error(y_true, y_pred, cost_under=6.0, cost_over=1.5)
    assert result == pytest.approx(120.0)


def test_cost_over_only():
    # y_pred > y_true: só superprevisão
    y_true = np.array([100.0])
    y_pred = np.array([120.0])
    # over = 20 unidades * cost_over=1.5 = 30
    result = cost_error(y_true, y_pred, cost_under=6.0, cost_over=1.5)
    assert result == pytest.approx(30.0)


def test_cost_asymmetry():
    # subprevisão deve ser mais cara que superprevisão do mesmo desvio
    y_true = np.array([100.0])
    y_pred_under = np.array([90.0])   # 10 abaixo
    y_pred_over = np.array([110.0])   # 10 acima

    cost_under = cost_error(y_true, y_pred_under, cost_under=6.0, cost_over=1.5)
    cost_over = cost_error(y_true, y_pred_over, cost_under=6.0, cost_over=1.5)

    assert cost_under > cost_over


def test_cost_always_nonnegative():
    rng = np.random.default_rng(7)
    y_true = rng.uniform(0, 1000, 100)
    y_pred = rng.uniform(0, 1000, 100)
    assert cost_error(y_true, y_pred, cost_under=6.0, cost_over=1.5) >= 0.0
