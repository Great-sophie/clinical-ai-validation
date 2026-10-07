from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.dca_bootstrap import nested_bootstrap_dca


def load_predictions(path):
    df = pd.read_csv(path)
    required = {"patient_id", "y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--target-name", default="external")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.10,0.20,0.30,0.40,0.50,0.60])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    calibration = load_predictions(args.calibration_predictions)
    target = load_predictions(args.target_predictions)

    result = nested_bootstrap_dca(
        calibration_y_true=calibration["y_true"].astype(int).to_numpy(),
        calibration_y_prob=calibration["y_prob"].astype(float).to_numpy(),
        target_y_true=target["y_true"].astype(int).to_numpy(),
        target_y_prob=target["y_prob"].astype(float).to_numpy(),
        thresholds=args.thresholds,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    df = pd.DataFrame(result["rows"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.target_name}_dca_threshold_table.csv"
    df.to_csv(output_path, index=False)

    print("\nDCA THRESHOLD TABLE")
    print("===================")
    print(f"Bootstrap: {result['valid_bootstrap_iterations']}/{result['requested_bootstrap_iterations']} valid iterations")
    print(f"Recalibration intercept: {result['recalibration_intercept']:.4f}")
    print(f"Recalibration slope    : {result['recalibration_slope']:.4f}\n")
    print(df[[
        "threshold", "nb_before", "nb_after", "delta_nb",
        "delta_nb_ci_lower", "delta_nb_ci_upper", "delta_nb_per_100"
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
