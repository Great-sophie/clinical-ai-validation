import numpy as np

from src.recalibration import (
    fit_logistic_recalibrator,
    apply_logistic_recalibration,
)


def test_recalibration_outputs_valid_probabilities():

    y_true = np.array([
        0, 0, 0, 1, 1, 1
    ])

    y_prob = np.array([
        0.10,
        0.20,
        0.35,
        0.60,
        0.75,
        0.90,
    ])

    params = fit_logistic_recalibrator(
        y_true,
        y_prob,
    )

    recalibrated = (
        apply_logistic_recalibration(
            y_prob,
            params["intercept"],
            params["slope"],
        )
    )

    assert np.all(
        recalibrated > 0
    )

    assert np.all(
        recalibrated < 1
    )

    assert np.isfinite(
        params["intercept"]
    )

    assert np.isfinite(
        params["slope"]
    )
