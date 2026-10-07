from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from src.metrics import evaluate_binary_predictions
from src.bootstrap import bootstrap_confidence_intervals
from src.dca import plot_decision_curve
from src.plots import plot_roc, plot_pr, plot_calibration

REQUIRED_COLUMNS = {"patient_id", "y_true", "y_prob"}


def validate_dataframe(df, cohort_name, output_dir, threshold, n_bootstrap, seed):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df["y_prob"].astype(float).to_numpy()

    if not ((y_prob >= 0) & (y_prob <= 1)).all():
        raise ValueError("y_prob must be in [0, 1].")

    metrics = evaluate_binary_predictions(y_true, y_prob, threshold=threshold)
    ci, bootstrap_rows = bootstrap_confidence_intervals(
        y_true,
        y_prob,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    result = {"cohort": cohort_name, "metrics": metrics, "bootstrap_95_ci": ci}
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / f"{cohort_name}_metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

    with (output_dir / f"{cohort_name}_bootstrap.csv").open("w", newline="") as f:
        fieldnames = list(bootstrap_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    plot_roc(y_true, y_prob, output_dir / f"{cohort_name}_roc_curve.png")
    plot_pr(y_true, y_prob, output_dir / f"{cohort_name}_pr_curve.png")
    plot_calibration(y_true, y_prob, output_dir / f"{cohort_name}_calibration_curve.png")
    plot_decision_curve(y_true, y_prob, output_dir / f"{cohort_name}_decision_curve.png")
    return result


def print_summary(result):
    m = result["metrics"]
    ci = result["bootstrap_95_ci"]

    def fmt(key):
        point = m[key]
        lo = ci[key]["lower"]
        hi = ci[key]["upper"]
        if lo is None or hi is None:
            return f"{point:.4f}"
        return f"{point:.4f} (95% CI {lo:.4f}-{hi:.4f})"

    print()
    print(result["cohort"].upper())
    print("=" * len(result["cohort"]))
    print(f"N            : {m['n']}")
    print(f"Prevalence   : {m['prevalence']:.4f}")
    print(f"AUROC        : {fmt('auroc')}")
    print(f"AUPRC        : {fmt('auprc')}")
    print(f"Brier        : {fmt('brier')}")
    print(f"Sensitivity  : {fmt('sensitivity')}")
    print(f"Specificity  : {fmt('specificity')}")
    print(f"PPV          : {fmt('ppv')}")
    print(f"NPV          : {fmt('npv')}")
    print(f"F1           : {fmt('f1')}")
    print(f"Cal intercept: {fmt('calibration_intercept')}")
    print(f"Cal slope    : {fmt('calibration_slope')}")


def main():
    parser = argparse.ArgumentParser(description="Clinical AI validation for binary predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--cohort-name", default="internal")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    result = validate_dataframe(
        df=df,
        cohort_name=args.cohort_name,
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
