from __future__ import annotations

import numpy as np

from .metrics import evaluate_binary_predictions


BOOTSTRAP_KEYS = [
    "auroc",
    "auprc",
    "brier",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "calibration_intercept",
    "calibration_slope",
]


def bootstrap_confidence_intervals(
    y_true,
    y_prob,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
):
    """
    Patient/sample-level nonparametric bootstrap.

    Samples are drawn with replacement.

    Bootstrap samples containing only one outcome class are skipped
    because AUROC and calibration parameters are undefined.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have equal length."
        )

    rng = np.random.default_rng(seed)
    n = len(y_true)

    rows = []

    for i in range(n_bootstrap):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yt = y_true[idx]
        yp = y_prob[idx]

        # Need both classes
        if len(np.unique(yt)) < 2:
            continue

        try:
            metrics = evaluate_binary_predictions(
                yt,
                yp,
                threshold=threshold,
            )
        except (ValueError, RuntimeError):
            continue

        row = {
            "bootstrap_iteration": i
        }

        for key in BOOTSTRAP_KEYS:
            row[key] = metrics[key]

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No valid bootstrap iterations were produced."
        )

    ci = {}

    low_q = 100 * alpha / 2
    high_q = 100 * (1 - alpha / 2)

    for key in BOOTSTRAP_KEYS:

        vals = np.asarray(
            [r[key] for r in rows],
            dtype=float,
        )

        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            ci[key] = {
                "lower": None,
                "upper": None,
            }
            continue

        ci[key] = {
            "lower": float(
                np.percentile(vals, low_q)
            ),
            "upper": float(
                np.percentile(vals, high_q)
            ),
        }

    ci["valid_bootstrap_iterations"] = len(rows)
    ci["requested_bootstrap_iterations"] = int(
        n_bootstrap
    )

    return ci, rows
