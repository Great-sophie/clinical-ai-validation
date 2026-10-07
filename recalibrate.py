from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.metrics import evaluate_binary_predictions
from src.recalibration import fit_logistic_recalibrator, apply_logistic_recalibration
from src.dca import decision_curve

REQUIRED_COLUMNS = {"patient_id", "y_true", "y_prob"}


def load_predictions(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df["y_prob"].between(0, 1).all():
        raise ValueError("y_prob must lie in [0, 1].")
    return df


def plot_calibration_before_after(y_true, prob_before, prob_after, output_path):
    before_obs, before_pred = calibration_curve(y_true, prob_before, n_bins=10, strategy="quantile")
    after_obs, after_pred = calibration_curve(y_true, prob_after, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot(before_pred, before_obs, marker="o", label="Before recalibration")
    plt.plot(after_pred, after_obs, marker="o", label="After recalibration")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Calibration: Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_dca_before_after(y_true, prob_before, prob_after, output_path):
    thresholds = np.linspace(0.01, 0.80, 80)
    before = decision_curve(y_true, prob_before, thresholds=thresholds)
    after = decision_curve(y_true, prob_after, thresholds=thresholds)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, before["model"], label="Model before recalibration")
    plt.plot(thresholds, after["model"], label="Model after recalibration")
    plt.plot(thresholds, before["treat_all"], linestyle="--", label="Treat all")
    plt.plot(thresholds, before["treat_none"], linestyle=":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve: Before vs After Recalibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def print_metrics(title, metrics):
    print()
    print(title)
    print("=" * len(title))
    for label, key in [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Brier", "brier"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("PPV", "ppv"),
        ("NPV", "npv"),
        ("F1", "f1"),
        ("Cal intercept", "calibration_intercept"),
        ("Cal slope", "calibration_slope"),
    ]:
        print(f"{label:13s}: {metrics[key]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--target-name", default="external")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_df = load_predictions(args.calibration_predictions)
    target_df = load_predictions(args.target_predictions)

    params = fit_logistic_recalibrator(
        calibration_df["y_true"].to_numpy(),
        calibration_df["y_prob"].to_numpy(),
    )

    print("\nFITTED RECALIBRATION PARAMETERS")
    print("===============================")
    print(f"Intercept: {params['intercept']:.4f}")
    print(f"Slope    : {params['slope']:.4f}")

    prob_before = target_df["y_prob"].astype(float).to_numpy()
    y_true = target_df["y_true"].astype(int).to_numpy()
    prob_after = apply_logistic_recalibration(
        prob_before,
        intercept=params["intercept"],
        slope=params["slope"],
    )

    before_metrics = evaluate_binary_predictions(y_true, prob_before, threshold=args.threshold)
    after_metrics = evaluate_binary_predictions(y_true, prob_after, threshold=args.threshold)
    print_metrics("BEFORE RECALIBRATION", before_metrics)
    print_metrics("AFTER RECALIBRATION", after_metrics)

    result_df = target_df.copy()
    result_df["y_prob_recalibrated"] = prob_after
    result_df.to_csv(output_dir / f"{args.target_name}_recalibrated_predictions.csv", index=False)

    with (output_dir / "recalibration_params.json").open("w") as f:
        json.dump(params, f, indent=2)

    comparison = {
        "target": args.target_name,
        "threshold": args.threshold,
        "recalibration_parameters": params,
        "before": before_metrics,
        "after": after_metrics,
    }
    with (output_dir / f"{args.target_name}_recalibration_comparison.json").open("w") as f:
        json.dump(comparison, f, indent=2)

    plot_calibration_before_after(
        y_true,
        prob_before,
        prob_after,
        output_dir / f"{args.target_name}_calibration_before_after.png",
    )
    plot_dca_before_after(
        y_true,
        prob_before,
        prob_after,
        output_dir / f"{args.target_name}_dca_before_after.png",
    )


if __name__ == "__main__":
    main()
