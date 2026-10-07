from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def decision_curve(
    y_true,
    y_prob,
    thresholds=None,
):
    """
    Calculate Decision Curve Analysis net benefit.

    Net benefit:
        TP / N - FP / N * pt / (1 - pt)

    Comparators:
        Treat none = 0
        Treat all  = prevalence - (1 - prevalence) * pt / (1 - pt)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(
            0.01,
            0.80,
            80,
        )

    n = len(y_true)
    prevalence = float(y_true.mean())

    model_nb = []
    treat_all_nb = []
    treat_none_nb = []

    for pt in thresholds:

        if pt <= 0 or pt >= 1:
            raise ValueError(
                "Decision thresholds must lie strictly between 0 and 1."
            )

        y_pred = (
            y_prob >= pt
        ).astype(int)

        tp = np.sum(
            (y_pred == 1)
            & (y_true == 1)
        )

        fp = np.sum(
            (y_pred == 1)
            & (y_true == 0)
        )

        weight = pt / (1 - pt)

        nb_model = (
            tp / n
            - fp / n * weight
        )

        nb_all = (
            prevalence
            - (1 - prevalence)
            * weight
        )

        model_nb.append(
            float(nb_model)
        )

        treat_all_nb.append(
            float(nb_all)
        )

        treat_none_nb.append(
            0.0
        )

    return {
        "thresholds": np.asarray(
            thresholds,
            dtype=float,
        ),
        "model": np.asarray(
            model_nb,
            dtype=float,
        ),
        "treat_all": np.asarray(
            treat_all_nb,
            dtype=float,
        ),
        "treat_none": np.asarray(
            treat_none_nb,
            dtype=float,
        ),
    }


def plot_decision_curve(
    y_true,
    y_prob,
    output_path,
    min_threshold=0.01,
    max_threshold=0.80,
    n_thresholds=80,
):
    thresholds = np.linspace(
        min_threshold,
        max_threshold,
        n_thresholds,
    )

    result = decision_curve(
        y_true,
        y_prob,
        thresholds=thresholds,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        result["thresholds"],
        result["model"],
        label="Model",
    )

    plt.plot(
        result["thresholds"],
        result["treat_all"],
        linestyle="--",
        label="Treat all",
    )

    plt.plot(
        result["thresholds"],
        result["treat_none"],
        linestyle=":",
        label="Treat none",
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Threshold probability"
    )

    plt.ylabel(
        "Net benefit"
    )

    plt.title(
        "Decision Curve Analysis"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    return result
