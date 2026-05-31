#!/usr/bin/env python3
"""Revision experiments for the GRSL risk-guarded fusion letter.

The script keeps the detector caches fixed and regenerates the compact
evidence added for the major revision:

* selected operating points for raw, NMS, and capped fusion;
* calibration-side utility-constrained contracts;
* IoU/alpha sensitivity for the raw and capped contracts.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import (  # noqa: E402
    candidate_thresholds,
    cp_upper,
    evaluate_object_threshold,
    filter_manifest,
    load_cache,
    object_match_table,
)
from fusion_guardrail_audit import postprocess  # noqa: E402


TABLE_DIR = ROOT / "output" / "tables"
UAVDT_COMBINED = ROOT / "data" / "caches" / "uavdt" / "combined_cache"
MANIFEST_DIR = ROOT / "data" / "manifests" / "uavdt"


VARIANTS = (
    ("raw960", None, None, 960),
    ("guard_nms", 0.40, None, None),
    ("guard_cap300", 0.40, 300, None),
)


def split_manifests(split_id: int) -> tuple[Path, Path]:
    return (
        MANIFEST_DIR / f"uavdt_val_random_half_{split_id}_cal153.csv",
        MANIFEST_DIR / f"uavdt_val_random_half_{split_id}_eval152.csv",
    )


def prepare_variant(pred: pd.DataFrame, name: str, nms_iou: float | None, max_det: int | None, resolution: int | None) -> pd.DataFrame:
    out = pred.copy()
    if resolution is not None:
        out = out.loc[out["resolution"].astype(int) == int(resolution)].copy()
    return postprocess(out, score_floor=0.0, nms_iou=nms_iou, max_det=max_det)


def per_split_tables(gt: pd.DataFrame, pred: pd.DataFrame, meta: pd.DataFrame, iou: float):
    rows = {}
    for split_id in range(1, 6):
        cal_manifest, eval_manifest = split_manifests(split_id)
        cal_gt, cal_pred, cal_meta = filter_manifest(gt, pred, meta, cal_manifest)
        eval_gt, eval_pred, eval_meta = filter_manifest(gt, pred, meta, eval_manifest)
        cal_objects, cal_scores = object_match_table(
            cal_gt, cal_pred, cal_meta, iou_threshold=iou, class_aware=True
        )
        eval_objects, eval_scores = object_match_table(
            eval_gt, eval_pred, eval_meta, iou_threshold=iou, class_aware=True
        )
        rows[split_id] = (
            cal_objects,
            cal_scores,
            cal_pred,
            len(cal_meta),
            eval_objects,
            eval_scores,
            len(eval_meta),
        )
    return rows


def select_threshold(
    cal_objects: pd.DataFrame,
    cal_scores,
    cal_pred: pd.DataFrame,
    *,
    alpha_select: float,
    cal_n_images: int,
    fp_budget: float | None = None,
    precision_min: float | None = None,
) -> tuple[pd.Series, bool]:
    rows = []
    for threshold in candidate_thresholds(cal_pred, 161):
        result = evaluate_object_threshold(cal_objects, cal_scores, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "cal_cp": cp_upper(result.misses, result.n_gt, 0.95),
                "cal_risk": result.miss_risk,
                "cal_precision": result.precision,
                "cal_fp_img": result.fp / cal_n_images,
            }
        )
    frame = pd.DataFrame(rows)
    feasible = frame.loc[frame["cal_cp"] <= float(alpha_select)].copy()
    if fp_budget is not None:
        feasible = feasible.loc[feasible["cal_fp_img"] <= float(fp_budget)].copy()
    if precision_min is not None:
        feasible = feasible.loc[feasible["cal_precision"] >= float(precision_min)].copy()
    if len(feasible):
        return feasible.sort_values("threshold", kind="mergesort").iloc[-1], True
    fallback = frame.sort_values(["cal_cp", "threshold"], ascending=[True, False], kind="mergesort").iloc[0]
    return fallback, False


def selected_and_utility(gt: pd.DataFrame, pred: pd.DataFrame, meta: pd.DataFrame) -> None:
    constraints = (
        ("risk-only", 0.152, None, None),
        ("cal-FP<=100", 0.160, 100.0, None),
        ("cal-FP<=75", 0.160, 75.0, None),
        ("cal-P>=0.30", 0.160, None, 0.30),
    )
    rows = []
    for variant, nms_iou, max_det, resolution in VARIANTS:
        vpred = prepare_variant(pred, variant, nms_iou, max_det, resolution)
        splits = per_split_tables(gt, vpred, meta, iou=0.25)
        for split_id, data in splits.items():
            cal_objects, cal_scores, cal_pred, cal_n_images, eval_objects, eval_scores, eval_n_images = data
            for contract, alpha_select, fp_budget, precision_min in constraints:
                if variant != "guard_cap300" and contract != "risk-only":
                    continue
                use_alpha = 0.160 if variant == "raw960" else alpha_select
                selected, feasible = select_threshold(
                    cal_objects,
                    cal_scores,
                    cal_pred,
                    alpha_select=use_alpha,
                    cal_n_images=cal_n_images,
                    fp_budget=fp_budget,
                    precision_min=precision_min,
                )
                result = evaluate_object_threshold(eval_objects, eval_scores, float(selected["threshold"]))
                rows.append(
                    {
                        "variant": variant,
                        "split": f"R{split_id}",
                        "contract": contract,
                        "alpha_select": use_alpha,
                        "threshold": float(selected["threshold"]),
                        "cal_cp": float(selected["cal_cp"]),
                        "cal_precision": float(selected["cal_precision"]),
                        "cal_fp_img": float(selected["cal_fp_img"]),
                        "cal_contract_feasible": bool(feasible),
                        "eval_risk": result.miss_risk,
                        "eval_precision": result.precision,
                        "eval_fp_img": result.fp / eval_n_images,
                        "eval_pass_alpha16": result.miss_risk <= 0.16,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "uavdt_revision_selected_utility_by_split.csv", index=False)
    summary = (
        frame.groupby(["variant", "contract", "alpha_select"], sort=False)
        .agg(
            cal_feasible=("cal_contract_feasible", "mean"),
            eval_pass=("eval_pass_alpha16", "mean"),
            threshold=("threshold", "mean"),
            cp=("cal_cp", "mean"),
            risk=("eval_risk", "mean"),
            worst_risk=("eval_risk", "max"),
            precision=("eval_precision", "mean"),
            fp_img=("eval_fp_img", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "uavdt_revision_selected_utility_summary.csv", index=False)
    print("\nSelected and utility-constrained contracts")
    print(summary.round(4).to_string(index=False))


def iou_alpha_sensitivity(gt: pd.DataFrame, pred: pd.DataFrame, meta: pd.DataFrame) -> None:
    rows = []
    for variant, nms_iou, max_det, resolution in (VARIANTS[0], VARIANTS[2]):
        vpred = prepare_variant(pred, variant, nms_iou, max_det, resolution)
        for iou in (0.25, 0.50):
            splits = per_split_tables(gt, vpred, meta, iou=iou)
            for alpha in (0.10, 0.12, 0.14, 0.16, 0.20):
                alpha_select = 0.152 if (variant == "guard_cap300" and iou == 0.25 and alpha == 0.16) else alpha
                for split_id, data in splits.items():
                    cal_objects, cal_scores, cal_pred, cal_n_images, eval_objects, eval_scores, eval_n_images = data
                    selected, feasible = select_threshold(
                        cal_objects,
                        cal_scores,
                        cal_pred,
                        alpha_select=alpha_select,
                        cal_n_images=cal_n_images,
                    )
                    result = evaluate_object_threshold(eval_objects, eval_scores, float(selected["threshold"]))
                    rows.append(
                        {
                            "variant": variant,
                            "split": f"R{split_id}",
                            "iou": iou,
                            "alpha": alpha,
                            "alpha_select": alpha_select,
                            "threshold": float(selected["threshold"]),
                            "cal_cp": float(selected["cal_cp"]),
                            "cal_feasible": bool(feasible),
                            "eval_risk": result.miss_risk,
                            "eval_precision": result.precision,
                            "eval_fp_img": result.fp / eval_n_images,
                            "eval_pass": result.miss_risk <= alpha,
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "uavdt_revision_iou_alpha_by_split.csv", index=False)
    summary = (
        frame.groupby(["variant", "iou", "alpha", "alpha_select"], sort=False)
        .agg(
            cert_rate=("cal_feasible", "mean"),
            pass_rate=("eval_pass", "mean"),
            threshold=("threshold", "mean"),
            cp=("cal_cp", "mean"),
            risk=("eval_risk", "mean"),
            worst_risk=("eval_risk", "max"),
            precision=("eval_precision", "mean"),
            fp_img=("eval_fp_img", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "uavdt_revision_iou_alpha_summary.csv", index=False)
    print("\nIoU/alpha sensitivity")
    print(summary.round(4).to_string(index=False))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gt, pred, meta = load_cache(UAVDT_COMBINED, None)
    selected_and_utility(gt, pred, meta)
    iou_alpha_sensitivity(gt, pred, meta)


if __name__ == "__main__":
    main()
