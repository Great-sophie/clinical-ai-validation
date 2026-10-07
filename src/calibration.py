import numpy as np
from sklearn.linear_model import LogisticRegression


def calibration_intercept_slope(y_true, y_prob, eps=1e-6):
    """
    Estimate calibration intercept and calibration slope using:

        logit(Y) = intercept + slope * logit(predicted probability)

    Ideal calibration:
        intercept = 0
        slope = 1
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Calibration intercept/slope require both classes."
        )

    y_prob = np.clip(y_prob, eps, 1 - eps)

    logit_prob = np.log(
        y_prob / (1.0 - y_prob)
    ).reshape(-1, 1)

    # Compatibility across scikit-learn versions
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(logit_prob, y_true)
    except (TypeError, ValueError):
        model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(logit_prob, y_true)

    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }
