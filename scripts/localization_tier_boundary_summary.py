#!/usr/bin/env python3
"""Summarize IoU 0.35/0.50 localization-tier boundary diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"


ROWS = [
    (
        "AITOD val",
        "0.35 fixed NMS+cap@0.125",
        "image",
        "fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.35_aitod_val_cache_summary.csv",
    ),
    (
        "AITOD val",
        "0.35 fixed NMS+cap@0.125",
        "block",
        "fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.35_aitod_val_cache_summary.csv",
    ),
    (
        "AITOD val",
        "0.35 fixed NMS+cap@0.125",
        "sequence",
        "fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.35_aitod_val_cache_summary.csv",
    ),
    (
        "AITOD val",
        "0.35 train-selected NMS min-loss",
        "image",
        "cluster_contract_search_hoeffding_aitod_iou35_nms3_minloss_uavdt_image_operational_iou0.35_base_hoeffding_min_loss_aitod_train_to_val_nmscap_iou35_image_strict_fc_selected.csv",
    ),
    (
        "AITOD val",
        "0.35 train-selected NMS min-loss",
        "block",
        "cluster_contract_search_hoeffding_aitod_iou35_nms3_minloss_uavdt_block_operational_iou0.35_base_hoeffding_min_loss_aitod_train_to_val_nmscap_iou35_block_strict_fc_selected.csv",
    ),
    (
        "AITOD val",
        "0.35 train-selected NMS min-loss",
        "sequence",
        "cluster_contract_search_hoeffding_aitod_iou35_nms3_minloss_uavdt_sequence_operational_iou0.35_base_hoeffding_min_loss_aitod_train_to_val_nmscap_iou35_sequence_strict_fc_selected.csv",
    ),
    (
        "UAVDT test",
        "0.35 fixed NMS+cap@0.125",
        "image",
        "fixed_cluster_candidate_eval_hoeffding_iou35_uavdt_uavdt_nms040_cap300_t0.125_image_operational_iou0.35_uavdt_test_cache_summary.csv",
    ),
    (
        "VisDrone val",
        "0.35 fixed NMS+cap@0.125",
        "image",
        "fixed_cluster_candidate_eval_hoeffding_iou35_visdrone_visdrone_nms040_cap300_t0.125_image_operational_iou0.35_visdrone_oracle_val_cache_summary.csv",
    ),
    (
        "AITOD val",
        "0.50 train-selected NMS+cap search",
        "image",
        "cluster_contract_search_aitod_train_to_val_stress_uavdt_image_operational_iou0.5_base_eb_min_loss_aitod_train_to_val_nmscap_iou50_image_strict_fc_selected.csv",
    ),
    (
        "AITOD val",
        "0.50 train-selected NMS stress",
        "sequence",
        "cluster_contract_search_aitod_train_to_val_stress_uavdt_sequence_operational_iou0.5_base_eb_min_loss_aitod_train_to_val_nmscap_iou50_sequence_strict_fc_selected.csv",
    ),
    (
        "UAVDT test",
        "0.50 train+val-to-test search",
        "image",
        "cluster_contract_search_uavdt_trainval_to_test_stress_uavdt_image_operational_iou0.5_base_eb_min_loss_trainval_to_test_iou50_strict_fc_selected.csv",
    ),
]


def load_row(source: str, diagnostic: str, unit: str, filename: str) -> dict:
    path = TABLE_DIR / filename
    row = pd.read_csv(path).iloc[0]
    return {
        "source": source,
        "diagnostic": diagnostic,
        "unit": unit,
        "contract": row["contract"],
        "threshold": float(row["threshold"]),
        "iou": float(row["iou"]),
        "loss_mean": float(row["loss_mean"]),
        "loss_upper": float(row["loss_upper"]),
        "loss_pass": bool(row["loss_pass"]),
        "object_risk": float(row["object_risk"]),
        "precision": float(row["precision"]),
        "fp_per_image": float(row["fp_per_image"]),
        "unit_count": int(row["unit_count"]),
        "evidence_file": filename,
    }


def main() -> None:
    out = pd.DataFrame(load_row(*row) for row in ROWS)
    out_path = TABLE_DIR / "localization_tier_boundary_summary.csv"
    out.to_csv(out_path, index=False)
    cols = [
        "source",
        "diagnostic",
        "unit",
        "contract",
        "threshold",
        "iou",
        "loss_upper",
        "loss_pass",
        "object_risk",
        "precision",
        "fp_per_image",
    ]
    print(out.loc[:, cols].round(4).to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
