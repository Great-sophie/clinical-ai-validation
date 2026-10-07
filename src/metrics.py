from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from .calibration import calibration_intercept_slope


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def discrimination_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "AUROC/AUPRC require both positive and negative classes."
        )

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }


def threshold_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict:

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)

    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_binary_predictions(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict:

    out = discrimination_metrics(
        y_true,
        y_prob,
    )

    out.update(
        threshold_metrics(
            y_true,
            y_prob,
            threshold=threshold,
        )
    )

    out.update(
        calibration_intercept_slope(
            y_true,
            y_prob,
        )
    )

    return out
