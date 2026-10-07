from __future__ import annotations

import numpy as np

from .recalibration import (
    fit_logistic_recalibrator,
    apply_logistic_recalibration,
)


def net_benefit(y_true, y_prob, threshold):
    """
    Model net benefit at one threshold probability.

    NB = TP/N - FP/N * pt/(1-pt)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must lie strictly between 0 and 1."
        )

    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum(
        (y_pred == 1) & (y_true == 1)
    )

    fp = np.sum(
        (y_pred == 1) & (y_true == 0)
    )

    n = len(y_true)

    weight = threshold / (1.0 - threshold)

    return float(
        tp / n
        - fp / n * weight
    )


def treat_all_net_benefit(
    y_true,
    threshold,
):
    """
    Net benefit if every patient is treated.
    """
    y_true = np.asarray(y_true, dtype=int)

    prevalence = float(
        np.mean(y_true)
    )

    weight = threshold / (1.0 - threshold)

    return float(
        prevalence
        - (1.0 - prevalence) * weight
    )


def _percentile_ci(values, alpha=0.05):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return None, None

    lower = np.percentile(
        values,
        100 * alpha / 2,
    )

    upper = np.percentile(
        values,
        100 * (1 - alpha / 2),
    )

    return float(lower), float(upper)


def nested_bootstrap_dca(
    calibration_y_true,
    calibration_y_prob,
    target_y_true,
    target_y_prob,
    thresholds,
    n_bootstrap=1000,
    seed=42,
    alpha=0.05,
):
    """
    Nested two-cohort bootstrap.

    Each iteration:

    1. Resample calibration/internal cohort.
    2. Refit logistic recalibration parameters.
    3. Resample external/target cohort.
    4. Apply frozen bootstrap recalibrator to external probabilities.
    5. Calculate before/after net benefit using the SAME target bootstrap.
    6. Calculate paired delta NB = after - before.

    This propagates uncertainty from both recalibration fitting
    and external validation sampling.
    """

    calibration_y_true = np.asarray(
        calibration_y_true,
        dtype=int,
    )

    calibration_y_prob = np.asarray(
        calibration_y_prob,
        dtype=float,
    )

    target_y_true = np.asarray(
        target_y_true,
        dtype=int,
    )

    target_y_prob = np.asarray(
        target_y_prob,
        dtype=float,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    rng = np.random.default_rng(seed)

    n_cal = len(calibration_y_true)
    n_target = len(target_y_true)

    # --------------------------------------------------------
    # Point estimates:
    # fit once on complete calibration cohort
    # --------------------------------------------------------
    params = fit_logistic_recalibrator(
        calibration_y_true,
        calibration_y_prob,
    )

    target_prob_after = apply_logistic_recalibration(
        target_y_prob,
        intercept=params["intercept"],
        slope=params["slope"],
    )

    point_rows = []

    for pt in thresholds:

        before = net_benefit(
            target_y_true,
            target_y_prob,
            pt,
        )

        after = net_benefit(
            target_y_true,
            target_prob_after,
            pt,
        )

        treat_all = treat_all_net_benefit(
            target_y_true,
            pt,
        )

        point_rows.append({
            "threshold": float(pt),
            "nb_before": before,
            "nb_after": after,
            "delta_nb": after - before,
            "treat_all": treat_all,
            "treat_none": 0.0,
        })

    # --------------------------------------------------------
    # Bootstrap storage
    # --------------------------------------------------------
    boot = {
        float(pt): {
            "before": [],
            "after": [],
            "delta": [],
            "treat_all": [],
        }
        for pt in thresholds
    }

    valid_iterations = 0

    for _ in range(n_bootstrap):

        # ------------------------------
        # Bootstrap calibration cohort
        # ------------------------------
        cal_idx = rng.integers(
            0,
            n_cal,
            size=n_cal,
        )

        cal_y = calibration_y_true[
            cal_idx
        ]

        cal_p = calibration_y_prob[
            cal_idx
        ]

        # Logistic recalibration needs both classes
        if len(np.unique(cal_y)) < 2:
            continue

        try:
            boot_params = (
                fit_logistic_recalibrator(
                    cal_y,
                    cal_p,
                )
            )
        except (ValueError, RuntimeError):
            continue

        # ------------------------------
        # Bootstrap external cohort
        # ------------------------------
        target_idx = rng.integers(
            0,
            n_target,
            size=n_target,
        )

        ext_y = target_y_true[
            target_idx
        ]

        ext_p_before = target_y_prob[
            target_idx
        ]

        ext_p_after = (
            apply_logistic_recalibration(
                ext_p_before,
                intercept=boot_params[
                    "intercept"
                ],
                slope=boot_params[
                    "slope"
                ],
            )
        )

        valid_iterations += 1

        for pt in thresholds:

            key = float(pt)

            before = net_benefit(
                ext_y,
                ext_p_before,
                pt,
            )

            after = net_benefit(
                ext_y,
                ext_p_after,
                pt,
            )

            treat_all = (
                treat_all_net_benefit(
                    ext_y,
                    pt,
                )
            )

            boot[key]["before"].append(
                before
            )

            boot[key]["after"].append(
                after
            )

            # paired difference
            boot[key]["delta"].append(
                after - before
            )

            boot[key]["treat_all"].append(
                treat_all
            )

    if valid_iterations == 0:
        raise RuntimeError(
            "No valid bootstrap iterations."
        )

    # --------------------------------------------------------
    # Confidence intervals
    # --------------------------------------------------------
    output_rows = []

    for point in point_rows:

        pt = point["threshold"]
        values = boot[pt]

        before_lo, before_hi = _percentile_ci(
            values["before"],
            alpha=alpha,
        )

        after_lo, after_hi = _percentile_ci(
            values["after"],
            alpha=alpha,
        )

        delta_lo, delta_hi = _percentile_ci(
            values["delta"],
            alpha=alpha,
        )

        all_lo, all_hi = _percentile_ci(
            values["treat_all"],
            alpha=alpha,
        )

        output_rows.append({
            **point,

            "nb_before_ci_lower": before_lo,
            "nb_before_ci_upper": before_hi,

            "nb_after_ci_lower": after_lo,
            "nb_after_ci_upper": after_hi,

            "delta_nb_ci_lower": delta_lo,
            "delta_nb_ci_upper": delta_hi,

            "treat_all_ci_lower": all_lo,
            "treat_all_ci_upper": all_hi,

            # Convenience: net benefit per 100 patients
            "nb_before_per_100": (
                100 * point["nb_before"]
            ),
            "nb_after_per_100": (
                100 * point["nb_after"]
            ),
            "delta_nb_per_100": (
                100 * point["delta_nb"]
            ),
        })

    return {
        "rows": output_rows,
        "valid_bootstrap_iterations": (
            valid_iterations
        ),
        "requested_bootstrap_iterations": (
            int(n_bootstrap)
        ),
        "recalibration_intercept": (
            params["intercept"]
        ),
        "recalibration_slope": (
            params["slope"]
        ),
    }
