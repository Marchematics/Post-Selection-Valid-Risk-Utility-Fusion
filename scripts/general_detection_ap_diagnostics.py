#!/usr/bin/env python3
"""General AP diagnostics for cached detector contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import box_iou_matrix, filter_manifest, load_cache  # noqa: E402
from post_selection_family_audit import apply_contract, contract_family  # noqa: E402


TABLE_DIR = ROOT / "output" / "tables"


def ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return float("nan")
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def class_ap(gt: pd.DataFrame, pred: pd.DataFrame, cls: int, iou_thr: float) -> tuple[float, int, int]:
    gt_c = gt.loc[(~gt["is_ignore"].astype(bool)) & (gt["cls"].astype(int) == int(cls))].copy()
    pred_c = pred.loc[pred["cls"].astype(int) == int(cls)].copy()
    n_gt = int(len(gt_c))
    if n_gt == 0:
        return float("nan"), 0, int(len(pred_c))
    if pred_c.empty:
        return 0.0, n_gt, 0

    gt_groups = {int(k): v.reset_index(drop=True) for k, v in gt_c.groupby("img_id", sort=False)}
    pred_c = pred_c.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    used_by_img = {img_id: np.zeros(len(group), dtype=bool) for img_id, group in gt_groups.items()}
    detections: list[tuple[float, float, float]] = []

    for img_id, pgroup in pred_c.groupby("img_id", sort=False):
        img_id = int(img_id)
        ggroup = gt_groups.get(img_id)
        if ggroup is None or len(ggroup) == 0:
            for score in pgroup["score"].to_numpy(dtype=np.float64):
                detections.append((float(score), 0.0, 1.0))
            continue
        pboxes = pgroup.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        gboxes = ggroup.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        ious = box_iou_matrix(pboxes, gboxes)
        used = used_by_img[img_id]
        for pred_idx, score in enumerate(pgroup["score"].to_numpy(dtype=np.float64)):
            gt_idx = int(np.argmax(ious[pred_idx]))
            best = float(ious[pred_idx, gt_idx])
            if best >= float(iou_thr) and not used[gt_idx]:
                used[gt_idx] = True
                detections.append((float(score), 1.0, 0.0))
            else:
                detections.append((float(score), 0.0, 1.0))

    if not detections:
        return 0.0, n_gt, 0
    det = pd.DataFrame(detections, columns=["score", "tp", "fp"]).sort_values(
        "score", ascending=False, kind="mergesort"
    )
    tp_cum = np.cumsum(det["tp"].to_numpy(dtype=float))
    fp_cum = np.cumsum(det["fp"].to_numpy(dtype=float))
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return ap_from_pr(recall, precision), n_gt, int(len(pred_c))


def evaluate_ap(gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float) -> tuple[dict[str, float], pd.DataFrame]:
    classes = sorted(set(gt.loc[~gt["is_ignore"].astype(bool), "cls"].astype(int).tolist()))
    rows = []
    for cls in classes:
        ap, n_gt, n_pred = class_ap(gt, pred, cls, iou_thr)
        rows.append({"cls": cls, "iou": float(iou_thr), "ap": ap, "n_gt": n_gt, "n_pred": n_pred})
    frame = pd.DataFrame(rows)
    valid = frame.loc[frame["n_gt"] > 0].copy()
    macro = float(valid["ap"].mean()) if len(valid) else float("nan")
    weighted = float(np.average(valid["ap"], weights=valid["n_gt"])) if len(valid) else float("nan")
    return {"macro_ap": macro, "weighted_ap": weighted}, frame


def selected_contracts(names: list[str], profile: str):
    family = {contract.name: contract for contract in contract_family(profile)}
    missing = [name for name in names if name not in family]
    if missing:
        raise ValueError(f"Unknown contracts for {profile}: {missing}")
    return [family[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--family-profile", choices=["base", "track", "track_only"], default="base")
    parser.add_argument("--contracts", default="raw640,raw960,nms040,nms040_cap300")
    parser.add_argument("--ious", default="0.25,0.50,0.75")
    parser.add_argument("--out-prefix", default="general_detection_ap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gt, pred, meta = load_cache(args.cache, None)
    gt, pred, meta = filter_manifest(gt, pred, meta, args.manifest)
    contracts = selected_contracts([x.strip() for x in args.contracts.split(",") if x.strip()], args.family_profile)
    ious = [float(x.strip()) for x in args.ious.split(",") if x.strip()]

    summary_rows = []
    class_rows = []
    for contract in contracts:
        ppred = apply_contract(pred, meta, contract)
        row = {
            "tag": args.tag,
            "cache": str(args.cache),
            "manifest": str(args.manifest) if args.manifest is not None else "",
            "contract": contract.name,
            "n_images": int(len(meta)),
            "n_gt": int((~gt["is_ignore"].astype(bool)).sum()),
            "n_pred": int(len(ppred)),
        }
        for iou in ious:
            vals, per_class = evaluate_ap(gt, ppred, iou)
            row[f"mAP{iou:g}"] = vals["macro_ap"]
            row[f"wAP{iou:g}"] = vals["weighted_ap"]
            per_class = per_class.assign(tag=args.tag, contract=contract.name)
            class_rows.append(per_class)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    classes = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    suffix = f"{args.out_prefix}_{args.tag}"
    summary.to_csv(TABLE_DIR / f"{suffix}_summary.csv", index=False)
    classes.to_csv(TABLE_DIR / f"{suffix}_per_class.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Wrote {TABLE_DIR / (suffix + '_summary.csv')}")


if __name__ == "__main__":
    main()
