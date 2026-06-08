#!/usr/bin/env python3
"""Top-K per-image review simulation for fixed detector contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import greedy_match_image, load_cache  # noqa: E402
from post_selection_family_audit import apply_contract, contract_family  # noqa: E402


DEFAULT_ROWS = [
    {
        "label": "aitod_raw960_t0.0075",
        "cache": "/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640",
        "contract": "raw960",
        "threshold": 0.0075,
    },
    {
        "label": "aitod_nmscap_t0.125",
        "cache": "/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640",
        "contract": "nms040_cap300",
        "threshold": 0.125,
    },
    {
        "label": "uavdt_raw960_t0.175",
        "cache": "/root/zjh_UAV_detection/experiments/uavdt/oracle_route/cache_test_baseline640",
        "contract": "raw960",
        "threshold": 0.175,
    },
    {
        "label": "uavdt_nmscap_t0.125",
        "cache": "/root/zjh_UAV_detection/experiments/uavdt/oracle_route/cache_test_baseline640",
        "contract": "nms040_cap300",
        "threshold": 0.125,
    },
    {
        "label": "visdrone_raw960_t0.175",
        "cache": "/root/zjh_UAV_detection/experiments/visdrone/oracle_route/cache_val_baseline640",
        "contract": "raw960",
        "threshold": 0.175,
    },
    {
        "label": "visdrone_nmscap_t0.125",
        "cache": "/root/zjh_UAV_detection/experiments/visdrone/oracle_route/cache_val_baseline640",
        "contract": "nms040_cap300",
        "threshold": 0.125,
    },
]


def get_contract(name: str):
    for contract in contract_family("base"):
        if contract.name == name:
            return contract
    raise ValueError(f"unknown contract: {name}")


def topk_frame(pred: pd.DataFrame, threshold: float, k: int) -> pd.DataFrame:
    kept = pred.loc[pd.to_numeric(pred["score"], errors="coerce") >= float(threshold)].copy()
    if k <= 0 or kept.empty:
        return kept.iloc[0:0].copy()
    return (
        kept.sort_values(["img_id", "score"], ascending=[True, False], kind="mergesort")
        .groupby("img_id", sort=False)
        .head(int(k))
        .reset_index(drop=True)
    )


def evaluate(label: str, cache: Path, contract_name: str, threshold: float, k_values: list[int], iou: float) -> list[dict]:
    gt, pred, meta = load_cache(cache, None)
    ppred = apply_contract(pred, meta, get_contract(contract_name))
    gt_groups = {int(key): frame for key, frame in gt.groupby("img_id", sort=False)}

    rows = []
    for k in k_values:
        reviewed = topk_frame(ppred, threshold, k)
        pred_groups = {int(key): frame for key, frame in reviewed.groupby("img_id", sort=False)}
        tp_total = 0
        fp_total = 0
        n_gt_total = 0
        reviewed_images = 0
        for _, image in meta.iterrows():
            img_id = int(image["img_id"])
            gt_img = gt_groups.get(img_id, gt.iloc[0:0])
            pred_img = pred_groups.get(img_id, reviewed.iloc[0:0])
            matched, tp, fp = greedy_match_image(
                gt_img,
                pred_img,
                threshold=0.0,
                iou_threshold=float(iou),
                class_aware=True,
            )
            n_gt_total += int(len(matched))
            tp_total += int(tp)
            fp_total += int(fp)
            reviewed_images += int(len(pred_img) > 0)
        boxes_total = int(len(reviewed))
        misses = int(n_gt_total - tp_total)
        rows.append(
            {
                "label": label,
                "cache": str(cache),
                "contract": contract_name,
                "threshold": float(threshold),
                "top_k": int(k),
                "iou": float(iou),
                "image_count": int(len(meta)),
                "reviewed_images": int(reviewed_images),
                "reviewed_image_share": float(reviewed_images / max(len(meta), 1)),
                "gt": int(n_gt_total),
                "tp": int(tp_total),
                "fp": int(fp_total),
                "misses": int(misses),
                "boxes_total": boxes_total,
                "boxes_per_image": float(boxes_total / max(len(meta), 1)),
                "recall": float(tp_total / max(n_gt_total, 1)),
                "miss_rate": float(misses / max(n_gt_total, 1)),
                "precision": float(tp_total / max(tp_total + fp_total, 1)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", default="10,25,50,100,300")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("output/tables/topk_review_simulation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    rows = []
    for spec in DEFAULT_ROWS:
        rows.extend(
            evaluate(
                spec["label"],
                Path(spec["cache"]),
                spec["contract"],
                float(spec["threshold"]),
                k_values,
                args.iou,
            )
        )
    out = pd.DataFrame(rows)

    by_label = {label: frame for label, frame in out.groupby("label", sort=False)}
    pairs = [
        ("aitod_raw960_t0.0075", "aitod_nmscap_t0.125"),
        ("uavdt_raw960_t0.175", "uavdt_nmscap_t0.125"),
        ("visdrone_raw960_t0.175", "visdrone_nmscap_t0.125"),
    ]
    for baseline, candidate in pairs:
        if baseline not in by_label or candidate not in by_label:
            continue
        base = by_label[baseline].set_index("top_k")
        for idx, row in out.loc[out["label"] == candidate].iterrows():
            k = int(row["top_k"])
            if k not in base.index:
                continue
            base_row = base.loc[k]
            out.loc[idx, "baseline_label"] = baseline
            out.loc[idx, "delta_recall_vs_baseline"] = float(row["recall"] - base_row["recall"])
            out.loc[idx, "delta_precision_vs_baseline"] = float(row["precision"] - base_row["precision"])
            out.loc[idx, "delta_boxes_vs_baseline"] = float(row["boxes_total"] - base_row["boxes_total"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
