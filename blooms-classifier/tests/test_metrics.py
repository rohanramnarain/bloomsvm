import numpy as np

from src.utils.metrics import compute_metrics


def test_metrics_macro():
    y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    res = compute_metrics(y_true, y_pred, y_pred)
    assert res.macro_f1 == 1.0
    assert res.kappa == 1.0
