from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _clip_probabilities(y_prob, eps=1e-6):
    y_prob = np.asarray(y_prob, dtype=float)
    return np.clip(y_prob, eps, 1 - eps)


def probability_to_logit(y_prob, eps=1e-6):
    y_prob = _clip_probabilities(y_prob, eps=eps)
    return np.log(
        y_prob / (1.0 - y_prob)
    )


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic_recalibrator(
    y_true,
    y_prob,
    eps=1e-6,
):
    """
    Fit:

        logit(Y) = intercept + slope * logit(p_original)

    Ideal original calibration:
        intercept = 0
        slope = 1
    """
    y_true = np.asarray(y_true, dtype=int)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Both outcome classes are required."
        )

    x = probability_to_logit(
        y_prob,
        eps=eps,
    ).reshape(-1, 1)

    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(x, y_true)

    except (TypeError, ValueError):
        model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(x, y_true)

    return {
        "intercept": float(
            model.intercept_[0]
        ),
        "slope": float(
            model.coef_[0, 0]
        ),
    }


def apply_logistic_recalibration(
    y_prob,
    intercept,
    slope,
    eps=1e-6,
):
    """
    Apply frozen recalibration parameters:

        logit(p_new)
        =
        intercept
        +
        slope * logit(p_old)
    """
    old_logit = probability_to_logit(
        y_prob,
        eps=eps,
    )

    new_logit = (
        intercept
        + slope * old_logit
    )

    return sigmoid(new_logit)
