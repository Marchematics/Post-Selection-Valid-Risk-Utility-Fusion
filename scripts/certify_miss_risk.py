#!/usr/bin/env python3
"""Finite-sample miss-risk certification for cached detector outputs.

The script consumes cached ground-truth and prediction parquet files with the
schema used by the local UAVDT/VisDrone STRIDE experiments. It freezes a
detector, scans confidence thresholds on a calibration split, computes a
one-sided Clopper-Pearson risk upper bound, and evaluates the selected
threshold on a held-out split.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class EvalResult:
    threshold: float
    n_gt: int
    misses: int
    tp: int
    fp: int

    @property
    def miss_risk(self) -> float:
        return float(self.misses / self.n_gt) if self.n_gt else float("nan")

    @property
    def recall(self) -> float:
        return float(self.tp / self.n_gt) if self.n_gt else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom else float("nan")

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
            return float("nan")
        return float(2.0 * p * r / (p + r))


def cp_upper(k: int, n: int, confidence: float) -> float:
    if n <= 0:
        return float("nan")
    if k <= 0:
        return float(1.0 - math.pow(1.0 - confidence, 1.0 / float(n)))
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(confidence, k + 1, n - k))


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match_image(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    threshold: float,
    iou_threshold: float,
    class_aware: bool,
) -> tuple[np.ndarray, int, int]:
    gt = gt.loc[~gt["is_ignore"].astype(bool)].copy()
    pred = pred.loc[pred["score"].astype(float) >= float(threshold)].copy()
    if len(gt) == 0:
        return np.zeros(0, dtype=bool), 0, int(len(pred))
    if len(pred) == 0:
        return np.zeros(len(gt), dtype=bool), 0, 0

    pred = pred.sort_values("score", ascending=False, kind="mergesort")
    gt_boxes = gt.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    pred_boxes = pred.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    ious = box_iou_matrix(pred_boxes, gt_boxes)
    if class_aware:
        pred_cls = pred["cls"].to_numpy()
        gt_cls = gt["cls"].to_numpy()
        ious = np.where(pred_cls[:, None] == gt_cls[None, :], ious, -1.0)

    matched_gt = np.zeros(len(gt), dtype=bool)
    tp = 0
    fp = 0
    for pred_idx in range(len(pred)):
        gt_idx = int(np.argmax(ious[pred_idx]))
        best_iou = float(ious[pred_idx, gt_idx]) if len(gt) else -1.0
        if best_iou >= float(iou_threshold) and not matched_gt[gt_idx]:
            matched_gt[gt_idx] = True
            tp += 1
        else:
            fp += 1
    return matched_gt, tp, fp


def evaluate_threshold(
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    *,
    threshold: float,
    iou_threshold: float,
    class_aware: bool,
) -> EvalResult:
    gt_groups = {int(k): v for k, v in gt_frame.groupby("img_id", sort=False)}
    pred_groups = {int(k): v for k, v in pred_frame.groupby("img_id", sort=False)}

    n_gt = 0
    misses = 0
    tp_total = 0
    fp_total = 0
    for img_id, gt in gt_groups.items():
        pred = pred_groups.get(img_id)
        if pred is None:
            pred = pred_frame.iloc[0:0]
        matched, tp, fp = greedy_match_image(
            gt,
            pred,
            threshold=threshold,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
        )
        n_gt += int(len(matched))
        misses += int((~matched).sum())
        tp_total += int(tp)
        fp_total += int(fp)

    extra_img_ids = set(pred_groups) - set(gt_groups)
    for img_id in extra_img_ids:
        fp_total += int((pred_groups[img_id]["score"].astype(float) >= float(threshold)).sum())

    return EvalResult(threshold=float(threshold), n_gt=n_gt, misses=misses, tp=tp_total, fp=fp_total)


def object_match_table(
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    iou_threshold: float,
    class_aware: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return one row per GT object with its assigned prediction score.

    For each image, predictions are processed once in descending score order,
    matching the usual detector evaluation rule. A GT object is detected at
    threshold t iff its assigned score is at least t.
    """
    gt_groups = {int(k): v.copy() for k, v in gt_frame.groupby("img_id", sort=False)}
    pred_groups = {int(k): v.copy() for k, v in pred_frame.groupby("img_id", sort=False)}
    object_rows = []
    all_scores = pd.to_numeric(pred_frame["score"], errors="coerce").dropna().to_numpy(dtype=np.float64)

    meta_cols = ["img_id", "num_gt_valid", "ratio_small"]
    meta_small = meta.loc[:, [col for col in meta_cols if col in meta.columns]].drop_duplicates("img_id")

    for img_id, gt_raw in gt_groups.items():
        gt = gt_raw.loc[~gt_raw["is_ignore"].astype(bool)].copy().reset_index(drop=True)
        pred = pred_groups.get(img_id)
        if pred is None:
            pred = pred_frame.iloc[0:0].copy()
        else:
            pred = pred.copy().sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)

        assigned = np.full(len(gt), np.nan, dtype=np.float64)
        if len(gt) and len(pred):
            gt_boxes = gt.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
            pred_boxes = pred.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
            ious = box_iou_matrix(pred_boxes, gt_boxes)
            if class_aware:
                pred_cls = pred["cls"].to_numpy()
                gt_cls = gt["cls"].to_numpy()
                ious = np.where(pred_cls[:, None] == gt_cls[None, :], ious, -1.0)
            matched = np.zeros(len(gt), dtype=bool)
            for pred_idx in range(len(pred)):
                gt_idx = int(np.argmax(ious[pred_idx]))
                best_iou = float(ious[pred_idx, gt_idx]) if len(gt) else -1.0
                if best_iou >= float(iou_threshold) and not matched[gt_idx]:
                    matched[gt_idx] = True
                    assigned[gt_idx] = float(pred.loc[pred_idx, "score"])

        if len(gt):
            rows = gt.loc[:, ["img_id", "img_name", "cls", "is_small_lt24", "is_tiny_lt16"]].copy()
            rows["match_score"] = assigned
            object_rows.append(rows)

    if object_rows:
        objects = pd.concat(object_rows, ignore_index=True)
    else:
        objects = pd.DataFrame(columns=["img_id", "img_name", "cls", "is_small_lt24", "is_tiny_lt16", "match_score"])
    objects["img_id"] = objects["img_id"].astype(int)
    objects = objects.merge(meta_small, on="img_id", how="left")
    return objects, all_scores


def evaluate_object_threshold(objects: pd.DataFrame, pred_scores: np.ndarray, threshold: float) -> EvalResult:
    match_scores = pd.to_numeric(objects["match_score"], errors="coerce").to_numpy(dtype=np.float64)
    detected = np.isfinite(match_scores) & (match_scores >= float(threshold))
    tp = int(detected.sum())
    n_gt = int(len(objects))
    pred_kept = int((np.asarray(pred_scores, dtype=np.float64) >= float(threshold)).sum())
    fp = max(0, pred_kept - tp)
    return EvalResult(threshold=float(threshold), n_gt=n_gt, misses=n_gt - tp, tp=tp, fp=fp)


def load_cache(cache_dir: Path, resolution: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_parquet(cache_dir / "gt_rows.parquet").copy()
    pred = pd.read_parquet(cache_dir / "pred_rows.parquet").copy()
    meta = pd.read_csv(cache_dir / "image_meta.csv").copy()
    if resolution is not None and "resolution" in pred.columns:
        pred = pred.loc[pred["resolution"].astype(int) == int(resolution)].copy()
    for frame in (gt, pred, meta):
        frame["img_id"] = frame["img_id"].astype(int)
    return gt, pred, meta


def filter_manifest(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    manifest_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if manifest_path is None:
        return gt, pred, meta
    manifest = pd.read_csv(manifest_path)
    if "img_id" not in manifest.columns:
        raise ValueError(f"Manifest lacks img_id column: {manifest_path}")
    ids = set(manifest["img_id"].astype(int).tolist())
    return (
        gt.loc[gt["img_id"].astype(int).isin(ids)].copy(),
        pred.loc[pred["img_id"].astype(int).isin(ids)].copy(),
        meta.loc[meta["img_id"].astype(int).isin(ids)].copy(),
    )


def candidate_thresholds(pred: pd.DataFrame, grid_size: int) -> list[float]:
    scores = pd.to_numeric(pred["score"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(scores) == 0:
        return [0.0]
    quantiles = np.linspace(0.0, 1.0, int(grid_size))
    values = np.quantile(scores, quantiles)
    values = np.unique(np.clip(values, 0.0, 1.0))
    anchors = np.asarray([0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
    return sorted(float(v) for v in np.unique(np.concatenate([values, anchors])) if np.isfinite(v))


def choose_threshold(
    cal_objects: pd.DataFrame,
    cal_pred_scores: np.ndarray,
    *,
    thresholds: Iterable[float],
    alpha: float,
    confidence: float,
    iou_threshold: float,
    class_aware: bool,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    selected = None
    # For miss-risk control, larger thresholds are more selective. We select the
    # largest threshold whose finite-sample upper risk bound still satisfies alpha.
    for threshold in sorted(thresholds):
        result = evaluate_object_threshold(cal_objects, cal_pred_scores, threshold)
        upper = cp_upper(result.misses, result.n_gt, confidence)
        row = {
            "threshold": result.threshold,
            "n_gt": result.n_gt,
            "misses": result.misses,
            "tp": result.tp,
            "fp": result.fp,
            "empirical_miss_risk": result.miss_risk,
            "cp_upper": upper,
            "recall": result.recall,
            "precision": result.precision,
            "f1": result.f1,
            "satisfies_alpha": bool(upper <= float(alpha)),
        }
        rows.append(row)
        if row["satisfies_alpha"]:
            selected = row
    if selected is None:
        selected = min(rows, key=lambda r: (r["cp_upper"], -r["recall"]))
        selected = {**selected, "selected_mode": "no_feasible_min_cp"}
    else:
        selected = {**selected, "selected_mode": "largest_feasible_threshold"}
    return selected, pd.DataFrame(rows)


def subgroup_masks(gt: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.Series]:
    gt = gt.merge(
        meta.loc[:, ["img_id", "num_gt_valid", "ratio_small"]],
        on="img_id",
        how="left",
    )
    dense_cut = float(meta["num_gt_valid"].quantile(0.75)) if len(meta) else float("inf")
    return {
        "all": pd.Series(True, index=gt.index),
        "small_lt24": gt["is_small_lt24"].astype(bool),
        "tiny_lt16": gt["is_tiny_lt16"].astype(bool),
        "dense_top_quartile": gt["num_gt_valid"].astype(float) >= dense_cut,
        "small_rich_images": gt["ratio_small"].astype(float) > 0.0,
    }


def evaluate_subgroups(
    objects: pd.DataFrame,
    pred_scores: np.ndarray,
    *,
    threshold: float,
    confidence: float,
    iou_threshold: float,
    class_aware: bool,
    split_name: str,
) -> pd.DataFrame:
    rows = []
    dense_cut = float(objects["num_gt_valid"].quantile(0.75)) if len(objects) else float("inf")
    masks = {
        "all": pd.Series(True, index=objects.index),
        "small_lt24": objects["is_small_lt24"].astype(bool),
        "tiny_lt16": objects["is_tiny_lt16"].astype(bool),
        "dense_top_quartile": objects["num_gt_valid"].astype(float) >= dense_cut,
        "small_rich_images": objects["ratio_small"].astype(float) > 0.0,
    }
    for subgroup, mask in masks.items():
        sub_objects = objects.loc[mask.to_numpy()].copy()
        if len(sub_objects) == 0:
            continue
        result = evaluate_object_threshold(sub_objects, pred_scores, threshold)
        rows.append(
            {
                "split": split_name,
                "subgroup": subgroup,
                "threshold": threshold,
                "n_gt": result.n_gt,
                "misses": result.misses,
                "empirical_miss_risk": result.miss_risk,
                "cp_upper": cp_upper(result.misses, result.n_gt, confidence),
                "recall": result.recall,
                "precision": result.precision,
                "f1": result.f1,
                "tp": result.tp,
                "fp": result.fp,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal-cache", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--cal-manifest", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--dataset", default="uavdt")
    parser.add_argument("--detector", default="yolov8l-baseline640")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("output/tables/uavdt_miss_risk"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cal_gt, cal_pred, cal_meta = load_cache(args.cal_cache, args.resolution)
    eval_gt, eval_pred, eval_meta = load_cache(args.eval_cache, args.resolution)
    cal_gt, cal_pred, cal_meta = filter_manifest(cal_gt, cal_pred, cal_meta, args.cal_manifest)
    eval_gt, eval_pred, eval_meta = filter_manifest(eval_gt, eval_pred, eval_meta, args.eval_manifest)
    class_aware = not bool(args.class_agnostic)
    cal_objects, cal_pred_scores = object_match_table(
        cal_gt,
        cal_pred,
        cal_meta,
        iou_threshold=args.iou,
        class_aware=class_aware,
    )
    eval_objects, eval_pred_scores = object_match_table(
        eval_gt,
        eval_pred,
        eval_meta,
        iou_threshold=args.iou,
        class_aware=class_aware,
    )

    thresholds = candidate_thresholds(cal_pred, args.grid_size)
    selected, sweep = choose_threshold(
        cal_objects,
        cal_pred_scores,
        thresholds=thresholds,
        alpha=args.alpha,
        confidence=args.confidence,
        iou_threshold=args.iou,
        class_aware=class_aware,
    )
    threshold = float(selected["threshold"])

    eval_result = evaluate_object_threshold(eval_objects, eval_pred_scores, threshold)
    eval_row = {
        "dataset": args.dataset,
        "detector": args.detector,
        "split": "eval",
        "alpha": args.alpha,
        "confidence": args.confidence,
        "iou": args.iou,
        "class_aware": class_aware,
        "threshold": threshold,
        "n_gt": eval_result.n_gt,
        "misses": eval_result.misses,
        "empirical_miss_risk": eval_result.miss_risk,
        "cp_upper": cp_upper(eval_result.misses, eval_result.n_gt, args.confidence),
        "recall": eval_result.recall,
        "precision": eval_result.precision,
        "f1": eval_result.f1,
        "tp": eval_result.tp,
        "fp": eval_result.fp,
    }

    cal_subgroups = evaluate_subgroups(
        cal_objects,
        cal_pred_scores,
        threshold=threshold,
        confidence=args.confidence,
        iou_threshold=args.iou,
        class_aware=class_aware,
        split_name="calibration",
    )
    eval_subgroups = evaluate_subgroups(
        eval_objects,
        eval_pred_scores,
        threshold=threshold,
        confidence=args.confidence,
        iou_threshold=args.iou,
        class_aware=class_aware,
        split_name="eval",
    )
    subgroups = pd.concat([cal_subgroups, eval_subgroups], ignore_index=True)

    summary = {
        "dataset": args.dataset,
        "detector": args.detector,
        "alpha": args.alpha,
        "confidence": args.confidence,
        "iou": args.iou,
        "resolution": args.resolution,
        "class_aware": class_aware,
        "calibration_cache": str(args.cal_cache),
        "evaluation_cache": str(args.eval_cache),
        "selected": selected,
        "evaluation": eval_row,
        "go_no_go": {
            "calibration_cp_upper_le_alpha": bool(float(selected["cp_upper"]) <= float(args.alpha)),
            "eval_empirical_risk_le_alpha": bool(float(eval_row["empirical_miss_risk"]) <= float(args.alpha)),
            "eval_recall": float(eval_row["recall"]),
        },
    }

    sweep.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame([eval_row]).to_csv(args.out_dir / "overall_eval.csv", index=False)
    subgroups.to_csv(args.out_dir / "subgroup_eval.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["go_no_go"], indent=2))
    print(f"selected threshold: {threshold:.6f}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
