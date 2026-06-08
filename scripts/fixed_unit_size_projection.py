#!/usr/bin/env python3
"""Project cluster sample sizes from a fixed-row units parquet file."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"


def hoeffding_upper(values: np.ndarray, *, confidence: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    delta = 1.0 - float(confidence)
    return float(values.mean() + math.sqrt(math.log(1.0 / delta) / (2.0 * len(values))))


def required_n(mean_loss: float, target: float, *, confidence: float) -> float:
    margin = float(target) - float(mean_loss)
    if margin <= 0:
        return float("inf")
    delta = 1.0 - float(confidence)
    return float(math.log(1.0 / delta) / (2.0 * margin * margin))


def operational_loss(units: pd.DataFrame, *, lambda_miss: float, lambda_fp: float, fp_cap: float) -> np.ndarray:
    miss = units["miss_rate"].to_numpy(dtype=np.float64)
    fp = np.minimum(units["fp_per_image"].to_numpy(dtype=np.float64), float(fp_cap)) / float(fp_cap)
    return float(lambda_miss) * miss + float(lambda_fp) * fp


def bootstrap_curve(values: np.ndarray, *, target: float, confidence: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    sizes = sorted(set([10, 20, 40, 80, 120, 160, 240, 320, 480, 640, 960, 1280, len(values)]))
    rows = []
    for size in sizes:
        if size <= 0 or size > len(values):
            continue
        reps = 300 if size < len(values) else 1
        means = []
        uppers = []
        for _ in range(reps):
            sample = values if size == len(values) else rng.choice(values, size=size, replace=False)
            means.append(float(sample.mean()))
            uppers.append(hoeffding_upper(sample, confidence=confidence))
        upper_arr = np.asarray(uppers)
        rows.append(
            {
                "subsample_n": int(size),
                "mean_loss_mean": float(np.mean(means)),
                "upper_mean": float(np.mean(upper_arr)),
                "upper_p10": float(np.quantile(upper_arr, 0.10)),
                "upper_p90": float(np.quantile(upper_arr, 0.90)),
                "pass_rate": float(np.mean(upper_arr <= float(target))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target", type=float, default=0.16)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--lambda-miss", type=float, default=0.8)
    parser.add_argument("--lambda-fp", type=float, default=0.2)
    parser.add_argument("--fp-cap", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    units = pd.read_parquet(args.units)
    values = operational_loss(units, lambda_miss=args.lambda_miss, lambda_fp=args.lambda_fp, fp_cap=args.fp_cap)
    mean_loss = float(values.mean())
    upper = hoeffding_upper(values, confidence=args.confidence)
    summary = pd.DataFrame(
        [
            {
                "label": args.label,
                "units_path": str(args.units),
                "unit_count": int(len(values)),
                "mean_loss": mean_loss,
                "hoeffding_upper": upper,
                "target": float(args.target),
                "confidence": float(args.confidence),
                "required_n_if_mean_fixed": required_n(mean_loss, args.target, confidence=args.confidence),
                "current_pass": bool(upper <= float(args.target)),
            }
        ]
    )
    curve = bootstrap_curve(values, target=args.target, confidence=args.confidence, seed=args.seed)
    curve.insert(0, "label", args.label)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TABLE_DIR / f"fixed_unit_size_projection_{args.label}_summary.csv"
    curve_path = TABLE_DIR / f"fixed_unit_size_projection_{args.label}_curve.csv"
    summary.to_csv(summary_path, index=False)
    curve.to_csv(curve_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {curve_path}")


if __name__ == "__main__":
    main()
