#!/usr/bin/env python3
"""Risk-utility guardrail audit for UAVDT 640+960 cache fusion.

The experiment keeps the detector outputs frozen and searches only over a
small, explicitly recorded post-processing family: score floor, class-aware
duplicate suppression, image-level detection cap, and a conservative
calibration target used to select the final confidence threshold.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import (  # noqa: E402
    box_iou_matrix,
    candidate_thresholds,
    cp_upper,
    evaluate_object_threshold,
    filter_manifest,
    load_cache,
    object_match_table,
)


TABLE_DIR = ROOT / "output" / "tables"
UAVDT_COMBINED = Path(os.environ.get("UAVDT_COMBINED", ROOT / "data" / "caches" / "uavdt" / "combined_cache"))
MANIFEST_DIR = Path(os.environ.get("UAVDT_MANIFEST_DIR", ROOT / "data" / "manifests" / "uavdt"))


def class_aware_nms(pred: pd.DataFrame, iou_threshold: float | None) -> pd.DataFrame:
    if iou_threshold is None or pred.empty:
        return pred.sort_values("score", ascending=False, kind="mergesort").copy()

    kept_parts = []
    for (_, cls), group in pred.groupby(["img_id", "cls"], sort=False):
        group = group.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        boxes = group.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        keep: list[int] = []
        remaining = np.arange(len(group))
        while len(remaining):
            current = int(remaining[0])
            keep.append(current)
            if len(remaining) == 1:
                break
            ious = box_iou_matrix(boxes[[current]], boxes[remaining[1:]])[0]
            remaining = remaining[1:][ious <= float(iou_threshold)]
        kept_parts.append(group.iloc[keep])

    if not kept_parts:
        return pred.iloc[0:0].copy()
    return pd.concat(kept_parts, ignore_index=True)


def cap_per_image(pred: pd.DataFrame, max_det: int | None) -> pd.DataFrame:
    if max_det is None or pred.empty:
        return pred.copy()
    return (
        pred.sort_values(["img_id", "score"], ascending=[True, False], kind="mergesort")
        .groupby("img_id", sort=False)
        .head(int(max_det))
        .reset_index(drop=True)
    )


def postprocess(
    pred: pd.DataFrame,
    *,
    score_floor: float,
    nms_iou: float | None,
    max_det: int | None,
) -> pd.DataFrame:
    out = pred.loc[pd.to_numeric(pred["score"], errors="coerce") >= float(score_floor)].copy()
    out = class_aware_nms(out, nms_iou)
    out = cap_per_image(out, max_det)
    return out.reset_index(drop=True)


def choose_with_guardrail(
    objects: pd.DataFrame,
    pred_scores: np.ndarray,
    pred_frame: pd.DataFrame,
    *,
    alpha_select: float,
    confidence: float,
    grid_size: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    selected = None
    for threshold in candidate_thresholds(pred_frame, grid_size):
        result = evaluate_object_threshold(objects, pred_scores, threshold)
        upper = cp_upper(result.misses, result.n_gt, confidence)
        row = {
            "threshold": float(threshold),
            "cal_miss_risk": result.miss_risk,
            "cal_cp_upper": upper,
            "cal_recall": result.recall,
            "cal_precision": result.precision,
            "cal_fp": result.fp,
            "cal_satisfies_alpha_select": bool(upper <= float(alpha_select)),
        }
        rows.append(row)
        if row["cal_satisfies_alpha_select"]:
            selected = row
    if selected is None:
        selected = min(rows, key=lambda row: (row["cal_cp_upper"], -row["cal_recall"]))
        selected = {**selected, "selected_mode": "no_feasible_min_cp"}
    else:
        selected = {**selected, "selected_mode": "largest_guardrail_feasible"}
    return selected, pd.DataFrame(rows)


def parse_floats(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for item in raw.split(","):
        item = item.strip()
        values.append(None if item.lower() in {"none", "na", ""} else float(item))
    return values


def parse_ints(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for item in raw.split(","):
        item = item.strip()
        values.append(None if item.lower() in {"none", "na", ""} else int(item))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=UAVDT_COMBINED)
    parser.add_argument("--alpha-eval", type=float, default=0.16)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--alpha-select", default="0.14,0.145,0.15,0.155,0.16")
    parser.add_argument("--score-floors", default="0,0.001,0.005,0.01")
    parser.add_argument("--nms-ious", default="none,0.5,0.6,0.7")
    parser.add_argument("--max-dets", default="none,150,200,300")
    parser.add_argument("--out-prefix", default="uavdt_fusion_guardrail")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gt, pred, meta = load_cache(args.cache, None)

    split_rows = []
    summary_rows = []
    for score_floor in [float(x) for x in parse_floats(args.score_floors) if x is not None]:
        for nms_iou in parse_floats(args.nms_ious):
            for max_det in parse_ints(args.max_dets):
                ppred = postprocess(pred, score_floor=score_floor, nms_iou=nms_iou, max_det=max_det)
                precomputed = {}
                for split_id in range(1, 6):
                    cal_manifest = MANIFEST_DIR / f"uavdt_val_random_half_{split_id}_cal153.csv"
                    eval_manifest = MANIFEST_DIR / f"uavdt_val_random_half_{split_id}_eval152.csv"
                    cal_gt, cal_pred, cal_meta = filter_manifest(gt, ppred, meta, cal_manifest)
                    eval_gt, eval_pred, eval_meta = filter_manifest(gt, ppred, meta, eval_manifest)
                    cal_objects, cal_scores = object_match_table(
                        cal_gt, cal_pred, cal_meta, iou_threshold=args.iou, class_aware=True
                    )
                    eval_objects, eval_scores = object_match_table(
                        eval_gt, eval_pred, eval_meta, iou_threshold=args.iou, class_aware=True
                    )
                    precomputed[split_id] = (
                        cal_objects,
                        cal_scores,
                        cal_pred,
                        eval_objects,
                        eval_scores,
                        len(eval_meta),
                    )

                for alpha_select in [float(x) for x in parse_floats(args.alpha_select) if x is not None]:
                    rows = []
                    for split_id, data in precomputed.items():
                        cal_objects, cal_scores, cal_pred, eval_objects, eval_scores, n_eval_images = data
                        selected, _ = choose_with_guardrail(
                            cal_objects,
                            cal_scores,
                            cal_pred,
                            alpha_select=alpha_select,
                            confidence=args.confidence,
                            grid_size=args.grid_size,
                        )
                        threshold = float(selected["threshold"])
                        eval_result = evaluate_object_threshold(eval_objects, eval_scores, threshold)
                        row = {
                            "split": f"random{split_id}",
                            "score_floor": score_floor,
                            "nms_iou": "none" if nms_iou is None else nms_iou,
                            "max_det": "none" if max_det is None else max_det,
                            "alpha_select": alpha_select,
                            "alpha_eval": args.alpha_eval,
                            "threshold": threshold,
                            "cal_cp_upper": float(selected["cal_cp_upper"]),
                            "cal_risk": float(selected["cal_miss_risk"]),
                            "eval_risk": eval_result.miss_risk,
                            "eval_recall": eval_result.recall,
                            "eval_precision": eval_result.precision,
                            "eval_fp": eval_result.fp,
                            "eval_fp_per_image": float(eval_result.fp / n_eval_images),
                            "cal_pass_eval_alpha": bool(float(selected["cal_cp_upper"]) <= args.alpha_eval),
                            "eval_pass": bool(eval_result.miss_risk <= args.alpha_eval),
                            "selected_mode": selected["selected_mode"],
                        }
                        rows.append(row)
                        split_rows.append(row)
                    frame = pd.DataFrame(rows)
                    summary_rows.append(
                        {
                            "score_floor": score_floor,
                            "nms_iou": "none" if nms_iou is None else nms_iou,
                            "max_det": "none" if max_det is None else max_det,
                            "alpha_select": alpha_select,
                            "cal_pass_rate": frame["cal_pass_eval_alpha"].mean(),
                            "eval_pass_rate": frame["eval_pass"].mean(),
                            "mean_cal_cp_upper": frame["cal_cp_upper"].mean(),
                            "mean_eval_risk": frame["eval_risk"].mean(),
                            "worst_eval_risk": frame["eval_risk"].max(),
                            "mean_precision": frame["eval_precision"].mean(),
                            "mean_fp_per_image": frame["eval_fp_per_image"].mean(),
                            "max_fp_per_image": frame["eval_fp_per_image"].max(),
                        }
                    )

    splits = pd.DataFrame(split_rows)
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["eval_pass_rate", "mean_fp_per_image", "mean_eval_risk", "mean_precision"],
        ascending=[False, True, True, False],
        kind="mergesort",
    )
    split_path = TABLE_DIR / f"{args.out_prefix}_splits.csv"
    summary_path = TABLE_DIR / f"{args.out_prefix}_summary.csv"
    splits.to_csv(split_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Top candidates:")
    print(summary.head(20).round(4).to_string(index=False))
    print(f"Wrote {split_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
