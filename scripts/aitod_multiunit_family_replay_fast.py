#!/usr/bin/env python3
"""Fast AITOD finite-family multi-unit replay from existing candidate tables.

This script does not recompute detections. It reuses existing AITOD train
candidate CSVs for the declared NMS/cap family, recomputes Bonferroni-corrected
Hoeffding bounds across contract x threshold x unit, selects one row on train,
and joins fixed-row AITOD validation summaries for the selected row.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"
UNITS = ("image", "block", "sequence")


TRAIN_CANDIDATES = {
    "image": (
        "cluster_contract_search_aitod_train_to_val_uavdt_image_operational_iou0.25_"
        "base_eb_min_loss_aitod_train_to_val_nmscap_strict_fc_candidates.csv"
    ),
    "block": (
        "cluster_contract_search_aitod_train_to_val_uavdt_block_operational_iou0.25_"
        "base_eb_min_loss_aitod_train_to_val_nmscap_block30_strict_fc_candidates.csv"
    ),
    "sequence": (
        "cluster_contract_search_aitod_train_to_val_uavdt_sequence_operational_iou0.25_"
        "base_eb_min_loss_aitod_train_to_val_nmscap_seq_strict_fc_candidates.csv"
    ),
}


def val_summary_path(contract: str, threshold: float, unit: str) -> Path:
    t = f"{threshold:g}"
    return TABLE_DIR / (
        f"fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_{contract}_t{t}_"
        f"{unit}_operational_iou0.25_aitod_val_cache_summary.csv"
    )


def load_train_candidates(contracts: set[str]) -> pd.DataFrame:
    merged = None
    for unit, filename in TRAIN_CANDIDATES.items():
        df = pd.read_csv(TABLE_DIR / filename)
        df = df.loc[df["contract"].isin(contracts)].copy()
        df = df.loc[
            :,
            ["contract", "threshold", "loss_mean", "unit_count", "fp_per_image", "precision", "object_risk"],
        ]
        df = df.rename(
            columns={
                "loss_mean": f"cal_{unit}_mean",
                "unit_count": f"cal_{unit}_n",
                "fp_per_image": f"cal_{unit}_fp_per_image",
                "precision": f"cal_{unit}_precision",
                "object_risk": f"cal_{unit}_object_risk",
            }
        )
        merged = df if merged is None else merged.merge(df, on=["contract", "threshold"], how="inner")
    if merged is None or merged.empty:
        raise RuntimeError("no train candidates loaded")
    return merged


def add_replay_bounds(df: pd.DataFrame, confidence: float, selection_target: float) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    n_tests = out["contract"].nunique() * out["threshold"].nunique() * len(UNITS)
    cal_delta = (1.0 - float(confidence)) / float(n_tests)
    for unit in UNITS:
        rad = np.sqrt(np.log(1.0 / cal_delta) / (2.0 * out[f"cal_{unit}_n"].astype(float)))
        out[f"cal_{unit}_upper"] = out[f"cal_{unit}_mean"] + rad
        out[f"cal_{unit}_pass"] = out[f"cal_{unit}_upper"] <= float(selection_target)
    out["cal_all_units_pass"] = out[[f"cal_{unit}_pass" for unit in UNITS]].all(axis=1)
    out["n_declared_tests"] = int(n_tests)
    out["cal_delta"] = float(cal_delta)
    out["selection_target"] = float(selection_target)
    return out, cal_delta


def select_row(candidates: pd.DataFrame, objective: str) -> pd.DataFrame:
    feasible = candidates.loc[candidates["cal_all_units_pass"]].copy()
    if feasible.empty:
        selected = candidates.assign(
            max_cal_upper=candidates[[f"cal_{unit}_upper" for unit in UNITS]].max(axis=1)
        ).sort_values(
            ["max_cal_upper", "cal_image_fp_per_image", "threshold"],
            ascending=[True, True, False],
            kind="mergesort",
        ).head(1).copy()
        selected["selection_mode"] = "no_train_multiunit_feasible_min_upper"
        return selected
    if objective == "min_loss":
        selected = feasible.assign(
            max_cal_upper=feasible[[f"cal_{unit}_upper" for unit in UNITS]].max(axis=1)
        ).sort_values(
            ["max_cal_upper", "cal_image_fp_per_image", "threshold"],
            ascending=[True, True, False],
            kind="mergesort",
        ).head(1).copy()
        selected["selection_mode"] = "train_multiunit_feasible_min_max_upper"
    else:
        selected = feasible.sort_values(
            ["cal_image_fp_per_image", "threshold"],
            ascending=[True, False],
            kind="mergesort",
        ).head(1).copy()
        selected["selection_mode"] = "train_multiunit_feasible_min_fp"
    return selected


def add_validation(selected: pd.DataFrame, target: float) -> pd.DataFrame:
    out = selected.copy()
    contract = str(out.iloc[0]["contract"])
    threshold = float(out.iloc[0]["threshold"])
    for unit in UNITS:
        path = val_summary_path(contract, threshold, unit)
        if not path.exists():
            out[f"eval_{unit}_available"] = False
            continue
        row = pd.read_csv(path).iloc[0]
        out[f"eval_{unit}_available"] = True
        out[f"eval_{unit}_n"] = int(row["unit_count"])
        out[f"eval_{unit}_mean"] = float(row["loss_mean"])
        out[f"eval_{unit}_upper"] = float(row["loss_upper"])
        out[f"eval_{unit}_pass"] = bool(float(row["loss_upper"]) <= float(target))
        out[f"eval_{unit}_object_risk"] = float(row["object_risk"])
        out[f"eval_{unit}_precision"] = float(row["precision"])
        out[f"eval_{unit}_fp_per_image"] = float(row["fp_per_image"])
    pass_cols = [f"eval_{unit}_pass" for unit in UNITS if f"eval_{unit}_pass" in out.columns]
    out["eval_all_units_pass"] = bool(len(pass_cols) == len(UNITS) and out.loc[:, pass_cols].all(axis=1).iloc[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", default="nms040,nms040_cap300,nms040_cap300_sf003")
    parser.add_argument("--selection-targets", default="0.14,0.16")
    parser.add_argument("--target", type=float, default=0.16)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--objective", choices=["min_fp", "min_loss"], default="min_fp")
    parser.add_argument("--out-prefix", default="aitod_multiunit_family_replay_fast")
    args = parser.parse_args()

    contracts = {x.strip() for x in args.contracts.split(",") if x.strip()}
    base = load_train_candidates(contracts)
    selected_rows = []
    all_candidates = []
    for target_text in [x.strip() for x in args.selection_targets.split(",") if x.strip()]:
        selection_target = float(target_text)
        candidates, _ = add_replay_bounds(base, args.confidence, selection_target)
        selected = select_row(candidates, args.objective)
        selected = add_validation(selected, args.target)
        selected_rows.append(selected)
        all_candidates.append(candidates)

    selected = pd.concat(selected_rows, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)
    selected_path = TABLE_DIR / f"{args.out_prefix}_selected.csv"
    candidates_path = TABLE_DIR / f"{args.out_prefix}_candidates.csv"
    selected.to_csv(selected_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    cols = [
        "selection_target",
        "contract",
        "threshold",
        "n_declared_tests",
        "cal_image_upper",
        "cal_block_upper",
        "cal_sequence_upper",
        "cal_image_fp_per_image",
        "cal_image_precision",
        "selection_mode",
        "eval_image_upper",
        "eval_block_upper",
        "eval_sequence_upper",
        "eval_all_units_pass",
    ]
    print(selected.loc[:, [c for c in cols if c in selected.columns]].round(4).to_string(index=False))
    print(f"Wrote {selected_path}")
    print(f"Wrote {candidates_path}")


if __name__ == "__main__":
    main()
