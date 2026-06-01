#!/usr/bin/env python3
"""Cluster-ratio fixed-sequence LTT audit for frozen UAV detector caches.

This script keeps the manuscript's finite post-processing family, but changes
the statistical sampling unit from individual objects to image/sequence/block
clusters.  The certified risk remains object-weighted miss risk:

    R(c,t) = E[M_g(c,t)] / E[N_g],

where g is a calibration unit.  For a target alpha, the ratio condition is
equivalent to E[M_g - alpha N_g] <= 0.  Thresholds are tested in a fixed
low-to-high sequence for each contract, so the miss-risk budget is spent over
contracts, not over every threshold-grid point.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import cp_upper, filter_manifest, load_cache, object_match_table  # noqa: E402
from post_selection_family_audit import (  # noqa: E402
    TABLE_DIR,
    UAVDT_CACHE,
    VISDRONE_CACHE,
    apply_contract,
    contract_family,
    fixed_thresholds,
    manifest_for,
    sequence_id,
    sequence_order,
)


def one_sided_hoeffding_upper(values: np.ndarray, *, delta: float, value_range: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    if value_range <= 0:
        return float(values.mean())
    rad = float(value_range) * math.sqrt(math.log(1.0 / max(delta, 1e-300)) / (2.0 * len(values)))
    return float(values.mean() + rad)


def empirical_bernstein_upper(values: np.ndarray, *, delta: float, value_range: float) -> float:
    """One-sided empirical Bernstein upper bound for bounded observations.

    Uses a standard Maurer--Pontil style finite-sample radius.  It is reported
    as an empirical-Bernstein screen; Hoeffding is also emitted as a conservative
    baseline.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan")
    if n == 1 or value_range <= 0:
        return one_sided_hoeffding_upper(values, delta=delta, value_range=max(value_range, 1e-12))
    sample_var = float(np.var(values, ddof=1))
    log_term = math.log(2.0 / max(delta, 1e-300))
    rad = math.sqrt(2.0 * sample_var * log_term / n) + 7.0 * float(value_range) * log_term / (3.0 * (n - 1))
    return float(values.mean() + rad)


def bounded_mean_upper(values: np.ndarray, *, delta: float, value_range: float, method: str) -> float:
    if method == "hoeffding":
        return one_sided_hoeffding_upper(values, delta=delta, value_range=value_range)
    if method == "eb":
        return empirical_bernstein_upper(values, delta=delta, value_range=value_range)
    raise ValueError(f"unknown bound method: {method}")


def add_unit_ids(meta: pd.DataFrame, unit: str, block_len: int) -> pd.DataFrame:
    out = meta.loc[:, ["img_id", "img_name"]].drop_duplicates("img_id").copy()
    out["seq"] = out["img_name"].map(sequence_id)
    out["ord"] = out["img_name"].map(sequence_order)
    if unit == "image":
        out["unit_id"] = out["img_id"].astype(int).map(lambda x: f"img:{x}")
    elif unit == "sequence":
        out["unit_id"] = out["seq"].map(lambda x: f"seq:{x}")
    elif unit == "block":
        block = (out["ord"].astype(int) // int(block_len)).astype(int)
        out["unit_id"] = "blk:" + out["seq"].astype(str) + ":" + block.astype(str)
    else:
        raise ValueError(f"unknown unit: {unit}")
    return out.loc[:, ["img_id", "img_name", "seq", "ord", "unit_id"]]


def image_counts_from_scores(objects: pd.DataFrame, pred: pd.DataFrame, meta: pd.DataFrame, threshold: float) -> pd.DataFrame:
    meta_base = meta.loc[:, ["img_id", "img_name"]].drop_duplicates("img_id").copy()
    meta_base["img_id"] = meta_base["img_id"].astype(int)

    if len(objects):
        obj = objects.copy()
        obj["img_id"] = obj["img_id"].astype(int)
        scores = pd.to_numeric(obj["match_score"], errors="coerce").to_numpy(dtype=np.float64)
        obj["detected"] = np.isfinite(scores) & (scores >= float(threshold))
        gt_by_img = obj.groupby("img_id", sort=False).agg(n_gt=("img_id", "size"), tp=("detected", "sum")).reset_index()
    else:
        gt_by_img = pd.DataFrame(columns=["img_id", "n_gt", "tp"])

    pred_frame = pred.copy()
    if len(pred_frame):
        pred_frame["img_id"] = pred_frame["img_id"].astype(int)
        kept = pred_frame.loc[pd.to_numeric(pred_frame["score"], errors="coerce") >= float(threshold)]
        pred_by_img = kept.groupby("img_id", sort=False).size().rename("pred_kept").reset_index()
    else:
        pred_by_img = pd.DataFrame(columns=["img_id", "pred_kept"])

    out = meta_base.merge(gt_by_img, on="img_id", how="left").merge(pred_by_img, on="img_id", how="left")
    for col in ("n_gt", "tp", "pred_kept"):
        out[col] = out[col].fillna(0).astype(int)
    out["misses"] = (out["n_gt"] - out["tp"]).clip(lower=0).astype(int)
    out["fp"] = (out["pred_kept"] - out["tp"]).clip(lower=0).astype(int)
    return out


def unit_counts(
    objects: pd.DataFrame,
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    threshold: float,
    unit: str,
    block_len: int,
) -> pd.DataFrame:
    image_rows = image_counts_from_scores(objects, pred, meta, threshold)
    units = add_unit_ids(meta, unit, block_len)
    rows = image_rows.merge(units.loc[:, ["img_id", "unit_id", "seq"]], on="img_id", how="left")
    grouped = (
        rows.groupby("unit_id", sort=False)
        .agg(
            seq=("seq", "first"),
            n_images=("img_id", "size"),
            n_gt=("n_gt", "sum"),
            misses=("misses", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
        )
        .reset_index()
    )
    grouped["risk"] = np.where(grouped["n_gt"] > 0, grouped["misses"] / grouped["n_gt"], 0.0)
    grouped["fp_img"] = grouped["fp"] / grouped["n_images"].clip(lower=1)
    return grouped


def cluster_bounds(unit_df: pd.DataFrame, *, alpha: float, beta_fp: float, delta_risk: float, delta_fp: float, fp_cap: float, n_bound: float, method: str) -> dict:
    z = unit_df["misses"].to_numpy(dtype=float) - float(alpha) * unit_df["n_gt"].to_numpy(dtype=float)
    fp_values = np.minimum(unit_df["fp_img"].to_numpy(dtype=float), float(fp_cap))
    z_range = float(n_bound)
    z_u = bounded_mean_upper(z, delta=delta_risk, value_range=z_range, method=method)
    z_u_hoeff = one_sided_hoeffding_upper(z, delta=delta_risk, value_range=z_range)
    fp_u = bounded_mean_upper(fp_values, delta=delta_fp, value_range=float(fp_cap), method=method)
    fp_u_hoeff = one_sided_hoeffding_upper(fp_values, delta=delta_fp, value_range=float(fp_cap))
    n_gt = float(unit_df["n_gt"].sum())
    misses = float(unit_df["misses"].sum())
    tp = float(unit_df["tp"].sum())
    fp = float(unit_df["fp"].sum())
    images = float(unit_df["n_images"].sum())
    bad = (unit_df["risk"] > float(alpha)) | (unit_df["fp_img"] > float(beta_fp))
    return {
        "unit_count": int(len(unit_df)),
        "image_count": int(images),
        "n_gt": int(n_gt),
        "misses": int(misses),
        "tp": int(tp),
        "fp": int(fp),
        "object_risk": float(misses / n_gt) if n_gt else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "fp_img": float(fp / images) if images else float("nan"),
        "z_mean": float(np.mean(z)) if len(z) else float("nan"),
        "z_upper": float(z_u),
        "z_upper_hoeffding": float(z_u_hoeff),
        "fp_upper": float(fp_u),
        "fp_upper_hoeffding": float(fp_u_hoeff),
        "n_bound": float(n_bound),
        "risk_feasible": bool(z_u <= 0.0),
        "utility_feasible": bool(fp_u <= float(beta_fp)),
        "bad_units": int(bad.sum()),
        "bad_unit_cp": cp_upper(int(bad.sum()), int(len(unit_df)), 1.0 - delta_risk) if len(unit_df) else float("nan"),
    }


def select_contract_rows(bounds: pd.DataFrame, *, beta_fp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    for contract, group in bounds.groupby("contract", sort=False):
        group = group.sort_values("threshold", kind="mergesort").copy()
        selected = None
        for _, row in group.iterrows():
            if bool(row["risk_feasible"]):
                selected = row
            else:
                break
        if selected is None:
            selected = group.sort_values(["z_upper", "threshold"], ascending=[True, True], kind="mergesort").iloc[0]
            mode = "no_risk_feasible_min_z"
        else:
            mode = "fixed_sequence_ltt"
        out = selected.to_dict()
        out["threshold_selection_mode"] = mode
        out["contract_risk_feasible"] = bool(mode == "fixed_sequence_ltt")
        out["joint_feasible"] = bool(out["contract_risk_feasible"] and out["fp_upper"] <= float(beta_fp))
        candidate_rows.append(out)

    candidates = pd.DataFrame(candidate_rows)
    feasible = candidates.loc[candidates["joint_feasible"]].copy()
    if len(feasible):
        idx = feasible.sort_values(["fp_upper", "z_upper", "threshold"], ascending=[True, True, False], kind="mergesort").index[0]
        selected_mode = "joint_feasible_min_fp_upper"
    else:
        risk_feasible = candidates.loc[candidates["contract_risk_feasible"]].copy()
        if len(risk_feasible):
            idx = risk_feasible.sort_values(["fp_img", "z_upper", "threshold"], ascending=[True, True, False], kind="mergesort").index[0]
            selected_mode = "risk_feasible_min_empirical_fp"
        else:
            idx = candidates.sort_values(["z_upper", "fp_img"], ascending=[True, True], kind="mergesort").index[0]
            selected_mode = "no_risk_feasible_min_z"
    candidates["selected"] = False
    candidates.loc[idx, "selected"] = True
    selected = candidates.loc[[idx]].copy()
    selected["selection_mode"] = selected_mode
    return candidates, selected


def split_manifests(args: argparse.Namespace, split: str) -> tuple[Path | None, Path | None]:
    if args.cal_manifest is not None or args.eval_manifest is not None:
        return args.cal_manifest, args.eval_manifest
    if args.cal_cache is not None or args.eval_cache is not None:
        return None, None
    return manifest_for(args.dataset, split)


def run_split(args: argparse.Namespace, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    default_cache = args.cache if args.cache is not None else (UAVDT_CACHE if args.dataset == "uavdt" else VISDRONE_CACHE)
    cal_cache = args.cal_cache if args.cal_cache is not None else default_cache
    eval_cache = args.eval_cache if args.eval_cache is not None else default_cache
    cal_manifest, eval_manifest = split_manifests(args, split)
    cal_gt_all, cal_pred_all, cal_meta_all = load_cache(cal_cache, None)
    if eval_cache == cal_cache:
        eval_gt_all, eval_pred_all, eval_meta_all = cal_gt_all, cal_pred_all, cal_meta_all
    else:
        eval_gt_all, eval_pred_all, eval_meta_all = load_cache(eval_cache, None)
    cal_gt, _, cal_meta = filter_manifest(cal_gt_all, cal_pred_all, cal_meta_all, cal_manifest)
    eval_gt, _, eval_meta = filter_manifest(eval_gt_all, eval_pred_all, eval_meta_all, eval_manifest)
    family = contract_family(args.family_profile)
    thresholds = fixed_thresholds(args.grid_size)
    delta = 1.0 - float(args.confidence)
    delta_risk = delta / (2.0 * len(family))
    delta_fp = delta / (2.0 * len(family) * len(thresholds))

    bound_rows = []
    selected_unit_rows = []
    for contract in family:
        cal_ppred = apply_contract(cal_pred_all, cal_meta_all, contract)
        eval_ppred = cal_ppred if eval_cache == cal_cache else apply_contract(eval_pred_all, eval_meta_all, contract)
        cal_gt_c, cal_pred, cal_meta_c = filter_manifest(cal_gt_all, cal_ppred, cal_meta_all, cal_manifest)
        eval_gt_c, eval_pred, eval_meta_c = filter_manifest(eval_gt_all, eval_ppred, eval_meta_all, eval_manifest)
        cal_objects, _ = object_match_table(cal_gt_c, cal_pred, cal_meta_c, iou_threshold=args.iou, class_aware=True)
        eval_objects, _ = object_match_table(eval_gt_c, eval_pred, eval_meta_c, iou_threshold=args.iou, class_aware=True)

        cal_units_for_bound = []
        if args.n_bound is None:
            # Screening default: use the maximum unit size visible to the audit.
            # For a deployable theorem statement, pass a predeclared --n-bound.
            for threshold in (thresholds[0],):
                cal_units_for_bound.append(unit_counts(cal_objects, cal_pred, cal_meta_c, threshold=threshold, unit=args.unit, block_len=args.block_len))
            n_bound = max(1.0, max(float(x["n_gt"].max()) if len(x) else 1.0 for x in cal_units_for_bound))
            n_bound_mode = "calibration_observed"
        else:
            n_bound = float(args.n_bound)
            n_bound_mode = "predeclared"

        for threshold in thresholds:
            cal_units = unit_counts(cal_objects, cal_pred, cal_meta_c, threshold=threshold, unit=args.unit, block_len=args.block_len)
            row = {
                "dataset": args.dataset,
                "split": split,
                "unit": args.unit,
                "contract": contract.name,
                "threshold": float(threshold),
                "alpha": float(args.alpha),
                "beta_fp": float(args.beta_fp),
                "confidence": float(args.confidence),
                "risk_delta": float(delta_risk),
                "fp_delta": float(delta_fp),
                "family_size": len(family),
                "threshold_grid_size": len(thresholds),
                "bound_method": args.bound_method,
                "n_bound_mode": n_bound_mode,
                "contract_record": str(asdict(contract)),
            }
            row.update(
                cluster_bounds(
                    cal_units,
                    alpha=args.alpha,
                    beta_fp=args.beta_fp,
                    delta_risk=delta_risk,
                    delta_fp=delta_fp,
                    fp_cap=args.fp_bound_cap,
                    n_bound=n_bound,
                    method=args.bound_method,
                )
            )
            bound_rows.append(row)

    bounds = pd.DataFrame(bound_rows)
    candidates, selected = select_contract_rows(bounds, beta_fp=args.beta_fp)

    for _, sel in selected.iterrows():
        contract = next(c for c in family if c.name == sel["contract"])
        eval_ppred = apply_contract(eval_pred_all, eval_meta_all, contract)
        eval_gt_c, eval_pred, eval_meta_c = filter_manifest(eval_gt_all, eval_ppred, eval_meta_all, eval_manifest)
        eval_objects, _ = object_match_table(eval_gt_c, eval_pred, eval_meta_c, iou_threshold=args.iou, class_aware=True)
        units = unit_counts(eval_objects, eval_pred, eval_meta_c, threshold=float(sel["threshold"]), unit=args.unit, block_len=args.block_len)
        units = units.assign(dataset=args.dataset, split=split, unit=args.unit, contract=contract.name, threshold=float(sel["threshold"]))
        selected_unit_rows.append(units)
        eval_summary = cluster_bounds(
            units,
            alpha=args.alpha,
            beta_fp=args.beta_fp,
            delta_risk=delta_risk,
            delta_fp=delta_fp,
            fp_cap=args.fp_bound_cap,
            n_bound=float(sel["n_bound"]),
            method=args.bound_method,
        )
        for key, value in eval_summary.items():
            selected.loc[:, f"eval_{key}"] = value

    return bounds, candidates, selected, pd.concat(selected_unit_rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uavdt", "visdrone"], default="uavdt")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cal-cache", type=Path, default=None, help="Calibration cache override. If set with --eval-cache, manifests may be omitted.")
    parser.add_argument("--eval-cache", type=Path, default=None, help="Evaluation cache override.")
    parser.add_argument("--cal-manifest", type=Path, default=None, help="Optional explicit calibration manifest.")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="Optional explicit evaluation manifest.")
    parser.add_argument("--splits", default="image_lockbox")
    parser.add_argument("--unit", choices=["image", "sequence", "block"], default="image")
    parser.add_argument("--block-len", type=int, default=30)
    parser.add_argument("--family-profile", choices=["base", "track", "track_only"], default="base")
    parser.add_argument("--alpha", type=float, default=0.16)
    parser.add_argument("--beta-fp", type=float, default=150.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bound-method", choices=["hoeffding", "eb"], default="eb")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--fp-bound-cap", type=float, default=300.0)
    parser.add_argument("--n-bound", type=float, default=None)
    parser.add_argument("--out-prefix", default="cluster_ratio_ltt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_bounds = []
    all_candidates = []
    all_selected = []
    all_units = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        bounds, candidates, selected, units = run_split(args, split)
        all_bounds.append(bounds)
        all_candidates.append(candidates)
        all_selected.append(selected)
        all_units.append(units)

    suffix = f"{args.out_prefix}_{args.dataset}_{args.unit}_a{args.alpha:g}_iou{args.iou:g}_{args.family_profile}_{args.bound_method}"
    bounds = pd.concat(all_bounds, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    units = pd.concat(all_units, ignore_index=True)
    bounds.to_csv(TABLE_DIR / f"{suffix}_bounds.csv", index=False)
    candidates.to_csv(TABLE_DIR / f"{suffix}_candidates.csv", index=False)
    selected.to_csv(TABLE_DIR / f"{suffix}_selected.csv", index=False)
    units.to_parquet(TABLE_DIR / f"{suffix}_selected_units.parquet", index=False)
    print("Cluster-ratio selected contracts:")
    cols = [
        "dataset",
        "split",
        "unit",
        "contract",
        "selection_mode",
        "threshold",
        "z_upper",
        "fp_upper",
        "object_risk",
        "precision",
        "fp_img",
        "eval_object_risk",
        "eval_precision",
        "eval_fp_img",
        "eval_bad_units",
        "eval_bad_unit_cp",
    ]
    print(selected.loc[:, [c for c in cols if c in selected.columns]].round(4).to_string(index=False))
    print(f"Wrote {TABLE_DIR / (suffix + '_selected.csv')}")


if __name__ == "__main__":
    main()
