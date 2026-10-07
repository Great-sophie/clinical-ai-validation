import numpy as np
from src.metrics import discrimination_metrics, threshold_metrics


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.10, 0.90, 0.99])
    d = discrimination_metrics(y_true, y_prob)
    t = threshold_metrics(y_true, y_prob, threshold=0.5)
    assert abs(d["auroc"] - 1.0) < 1e-8
    assert abs(d["auprc"] - 1.0) < 1e-8
    assert abs(t["sensitivity"] - 1.0) < 1e-8
    assert abs(t["specificity"] - 1.0) < 1e-8


def test_brier_range():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.6])
    d = discrimination_metrics(y_true, y_prob)
    assert 0.0 <= d["brier"] <= 1.0
