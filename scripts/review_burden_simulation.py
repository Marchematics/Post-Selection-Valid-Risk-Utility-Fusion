#!/usr/bin/env python3
"""Simple human-review burden simulation from fixed-candidate unit tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROWS = {
    "aitod_raw960_t0.0075": "output/tables/fixed_cluster_candidate_eval_aitod_uavdt_raw960_t0.0075_image_operational_iou0.25_aitod_val_cache_units.parquet",
    "aitod_nmscap_t0.075": "output/tables/fixed_cluster_candidate_eval_aitod_uavdt_nms040_cap300_t0.075_image_operational_iou0.25_aitod_val_cache_units.parquet",
    "aitod_nmscap_t0.15": "output/tables/fixed_cluster_candidate_eval_aitod_uavdt_nms040_cap300_t0.15_image_operational_iou0.25_aitod_val_cache_units.parquet",
    "uavdt_raw960_t0.175": "output/tables/fixed_cluster_candidate_eval_lockbox_uavdt_raw960_t0.175_image_operational_iou0.25_uavdt_test_cache_units.parquet",
    "uavdt_nmscap_t0.075": "output/tables/fixed_cluster_candidate_eval_lockbox_uavdt_nms040_cap300_t0.075_image_operational_iou0.25_uavdt_test_cache_units.parquet",
    "uavdt_nmscap_t0.15": "output/tables/fixed_cluster_candidate_eval_lockbox_uavdt_nms040_cap300_t0.15_image_operational_iou0.25_uavdt_test_cache_units.parquet",
    "visdrone_nmscap_t0.075": "output/tables/fixed_cluster_candidate_eval_visdrone_oracle_visdrone_nms040_cap300_t0.075_image_operational_iou0.25_visdrone_oracle_val_cache_units.parquet",
    "visdrone_nmscap_t0.15": "output/tables/fixed_cluster_candidate_eval_visdrone_oracle_visdrone_nms040_cap300_t0.15_image_operational_iou0.25_visdrone_oracle_val_cache_units.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/tables/review_burden_simulation.csv"))
    parser.add_argument("--seconds-per-box", default="1,2,5")
    return parser.parse_args()


def summarize(label: str, path: Path, seconds_per_box: list[float]) -> dict:
    units = pd.read_parquet(path)
    boxes = units["tp"] + units["fp"]
    n_images = float(units["n_images"].sum()) if "n_images" in units.columns else float(len(units))
    row = {
        "label": label,
        "source": str(path),
        "unit_count": int(len(units)),
        "image_count": int(n_images),
        "gt": int(units["n_gt"].sum()),
        "tp": int(units["tp"].sum()),
        "fp": int(units["fp"].sum()),
        "misses": int(units["misses"].sum()),
        "boxes_total": int(boxes.sum()),
        "boxes_per_image": float(boxes.sum() / max(n_images, 1.0)),
        "fp_per_image": float(units["fp"].sum() / max(n_images, 1.0)),
        "miss_rate": float(units["misses"].sum() / max(units["n_gt"].sum(), 1.0)),
        "precision": float(units["tp"].sum() / max(units["tp"].sum() + units["fp"].sum(), 1.0)),
        "boxes_median_unit": float(boxes.median()),
        "boxes_p90_unit": float(boxes.quantile(0.90)),
        "boxes_p95_unit": float(boxes.quantile(0.95)),
        "boxes_max_unit": float(boxes.max()),
        "share_units_boxes_ge_50": float((boxes >= 50).mean()),
        "share_units_boxes_ge_100": float((boxes >= 100).mean()),
        "share_units_boxes_ge_150": float((boxes >= 150).mean()),
    }
    for seconds in seconds_per_box:
        minutes = float(boxes.sum() * seconds / 60.0)
        row[f"review_minutes_at_{seconds:g}s_per_box"] = minutes
        row[f"review_hours_at_{seconds:g}s_per_box"] = minutes / 60.0
    return row


def main() -> None:
    args = parse_args()
    seconds_per_box = [float(x.strip()) for x in args.seconds_per_box.split(",") if x.strip()]
    rows = []
    for label, rel_path in DEFAULT_ROWS.items():
        path = Path(rel_path)
        if path.exists():
            rows.append(summarize(label, path, seconds_per_box))
    out = pd.DataFrame(rows)

    # Pairwise savings for rows with a raw baseline.
    if not out.empty:
        by_label = {row["label"]: row for _, row in out.iterrows()}
        for base, final in [
            ("aitod_raw960_t0.0075", "aitod_nmscap_t0.075"),
            ("aitod_raw960_t0.0075", "aitod_nmscap_t0.15"),
            ("uavdt_raw960_t0.175", "uavdt_nmscap_t0.075"),
            ("uavdt_raw960_t0.175", "uavdt_nmscap_t0.15"),
        ]:
            if base in by_label and final in by_label:
                base_boxes = float(by_label[base]["boxes_total"])
                final_boxes = float(by_label[final]["boxes_total"])
                mask = out["label"] == final
                out.loc[mask, "baseline_label"] = base
                out.loc[mask, "boxes_saved_vs_baseline"] = base_boxes - final_boxes
                out.loc[mask, "box_reduction_vs_baseline"] = (base_boxes - final_boxes) / max(base_boxes, 1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
