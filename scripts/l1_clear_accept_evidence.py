#!/usr/bin/env python3
"""Generate compact L1 evidence tables for the GRSL revision.

The active GRSL manuscript uses AITOD RT-DETR and AITOD YOLO11n-family
train-to-validation summaries for the review-budget result. This script keeps
the review-budget and boundary-table utilities used during the L1 revision.
It also retains an optional legacy UAVDT YOLOv8L helper for boundary checks;
that helper is not the AITOD second-detector result reported in the paper.
The script does not train detectors or modify manuscript bibliography files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import filter_manifest, load_cache, object_match_table  # noqa: E402
from cluster_contract_search_fast import loss_values, prepared_arrays, summarize, unit_counts_at  # noqa: E402
from fixed_cluster_candidate_eval import find_contract  # noqa: E402
from post_selection_family_audit import apply_contract  # noqa: E402


TABLE_DIR = ROOT / "output" / "tables"
L1_DIR = ROOT / "output" / "l1"

UAVDT_GT_CACHE = Path(
    os.environ.get("UAVDT_GT_CACHE", "/root/zjh_UAV_detection/experiments/uavdt/oracle_route/cache_val_baseline640")
)
YOLOV8L_640 = Path(
    os.environ.get(
        "YOLOV8L_640_PRED",
        "/root/zjh_UAV_detection/experiments/uavdt/oracle_route/router_runs/"
        "yolov8l_paired_replication/cache_yolov8n_640/pred_rows.parquet",
    )
)
YOLOV8L_960 = Path(
    os.environ.get(
        "YOLOV8L_960_PRED",
        "/root/zjh_UAV_detection/experiments/uavdt/oracle_route/router_runs/"
        "yolov8l_paired_replication/cache_yolov8n_960/pred_rows.parquet",
    )
)
YOLOV8L_CACHE = L1_DIR / "yolov8l_uavdt_val_combined_cache"


def ensure_yolov8l_cache(force: bool = False) -> Path:
    """Create a standard combined cache for the frozen YOLOv8L UAVDT rows."""

    required = [UAVDT_GT_CACHE / "gt_rows.parquet", UAVDT_GT_CACHE / "image_meta.csv", YOLOV8L_640, YOLOV8L_960]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs for YOLOv8L cache: " + ", ".join(missing))

    YOLOV8L_CACHE.mkdir(parents=True, exist_ok=True)
    pred_path = YOLOV8L_CACHE / "pred_rows.parquet"
    if force or not pred_path.exists():
        pred640 = pd.read_parquet(YOLOV8L_640).copy()
        pred960 = pd.read_parquet(YOLOV8L_960).copy()
        pred = pd.concat([pred640, pred960], ignore_index=True)
        pred["img_id"] = pred["img_id"].astype(int)
        pred["resolution"] = pred["resolution"].astype(int)
        pred["cls"] = pred["cls"].astype(int)
        pred = pred.sort_values(["img_id", "resolution", "score"], ascending=[True, True, False], kind="mergesort")
        pred.to_parquet(pred_path, index=False)

    for name in ("gt_rows.parquet", "image_meta.csv"):
        target = YOLOV8L_CACHE / name
        if force or not target.exists():
            shutil.copy2(UAVDT_GT_CACHE / name, target)

    return YOLOV8L_CACHE


def audit_fixed_row(
    *,
    cache: Path,
    detector: str,
    dataset: str,
    contract_name: str,
    threshold: float,
    unit: str,
    iou: float = 0.25,
    loss: str = "operational",
    target: float = 0.16,
    confidence: float = 0.95,
    lambda_miss: float = 0.8,
    lambda_fp: float = 0.2,
    fp_cap: float = 300.0,
) -> dict:
    """Evaluate one fixed contract/threshold on the whole cache."""

    class Args:
        pass

    args = Args()
    args.unit = unit
    args.block_len = 30
    args.sequence_parser = "strict"
    args.loss = loss
    args.target = target
    args.confidence = confidence
    args.bound_method = "hoeffding"
    args.lambda_miss = lambda_miss
    args.lambda_fp = lambda_fp
    args.fp_cap = fp_cap

    gt, pred, meta = load_cache(cache, None)
    contract = find_contract(contract_name, "base")
    ppred = apply_contract(pred, meta, contract)
    objects, _ = object_match_table(gt, ppred, meta, iou_threshold=iou, class_aware=True)
    arr = prepared_arrays(objects, ppred, meta, unit, args.block_len, args.sequence_parser)
    counts = unit_counts_at(arr, threshold)
    vals, value_range = loss_values(counts, args)
    row = {
        "detector": detector,
        "dataset": dataset,
        "cache": str(cache),
        "contract": contract_name,
        "threshold": float(threshold),
        "unit": unit,
        "iou": float(iou),
        "loss": loss,
        "target": float(target),
    }
    row.update(summarize(counts, vals, delta=1.0 - confidence, value_range=value_range, args=args))
    return row


def second_detector_table(force_cache: bool = False) -> pd.DataFrame:
    cache = ensure_yolov8l_cache(force=force_cache)
    rows = []
    for contract, threshold in [("raw960", 0.125), ("raw960", 0.175), ("nms040_cap300", 0.125), ("nms040_cap300", 0.15)]:
        for unit in ("image", "block", "sequence"):
            rows.append(
                audit_fixed_row(
                    cache=cache,
                    detector="YOLOv8L",
                    dataset="UAVDT val",
                    contract_name=contract,
                    threshold=threshold,
                    unit=unit,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "l1_second_detector_uavdt_yolov8l_summary.csv", index=False)
    return out


def method_baseline_table() -> pd.DataFrame:
    """Summarize available LTT/CRC-style cluster-ratio rows."""

    specs = [
        (
            "LTT image-unit family baseline",
            TABLE_DIR / "cluster_ratio_trainval_imagehash90_full_uavdt_image_a0.16_iou0.25_base_eb_selected.csv",
        ),
        (
            "LTT sequence-unit family baseline",
            TABLE_DIR / "cluster_ratio_trainval_seqhash90_probe_uavdt_sequence_a0.16_iou0.25_base_eb_selected.csv",
        ),
    ]
    rows = []
    for label, path in specs:
        if not path.exists():
            rows.append({"baseline": label, "available": False, "source_csv": str(path)})
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            rows.append({"baseline": label, "available": False, "source_csv": str(path)})
            continue
        row = frame.iloc[0].to_dict()
        rows.append(
            {
                "baseline": label,
                "available": True,
                "dataset": row.get("dataset", ""),
                "unit": row.get("unit", ""),
                "contract": row.get("contract", ""),
                "threshold": row.get("threshold", np.nan),
                "selection_mode": row.get("selection_mode", ""),
                "cal_risk_feasible": row.get("risk_feasible", np.nan),
                "cal_utility_feasible": row.get("utility_feasible", np.nan),
                "eval_object_risk": row.get("eval_object_risk", np.nan),
                "eval_precision": row.get("eval_precision", np.nan),
                "eval_fp_img": row.get("eval_fp_img", np.nan),
                "eval_z_upper": row.get("eval_z_upper", np.nan),
                "eval_risk_feasible": row.get("eval_risk_feasible", np.nan),
                "eval_bad_units": row.get("eval_bad_units", np.nan),
                "eval_bad_unit_cp": row.get("eval_bad_unit_cp", np.nan),
                "source_csv": str(path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "l1_method_baseline_summary.csv", index=False)
    return out


def review_budget_anchor_table() -> pd.DataFrame:
    source = TABLE_DIR / "topk_review_simulation.csv"
    frame = pd.read_csv(source)
    keep = frame.loc[
        frame["label"].isin(["aitod_raw960_t0.0075", "aitod_nmscap_t0.125"])
        & frame["top_k"].isin([25, 50, 100, 300])
    ].copy()
    keep["review_budget_interpretation"] = np.where(
        keep["top_k"].astype(int) == 50,
        "Primary offline review anchor: same loose-triage recall at about half the boxes.",
        "Sensitivity point for the review-list budget.",
    )
    keep.to_csv(TABLE_DIR / "l1_review_budget_anchor.csv", index=False)
    return keep


def transfer_boundary_map() -> pd.DataFrame:
    audit = pd.read_csv(TABLE_DIR / "cluster_unit_feasibility_uavdt_iou0.25_summary.csv")
    vis = pd.read_csv(TABLE_DIR / "cluster_unit_feasibility_visdrone_iou0.25_summary.csv")
    burden = pd.read_csv(TABLE_DIR / "review_burden_simulation_v2.csv")
    rows = [
        {
            "source": "AITOD val",
            "cluster_support": "image/block/parsed-sequence pass",
            "utility_result": "review boxes decrease and precision increases",
            "interpretation": "positive low-IoU object-presence triage case",
        },
        {
            "source": "UAVDT test",
            "cluster_support": "image pass; block/sequence fail",
            "utility_result": "review boxes increase and precision decreases versus raw row",
            "interpretation": "boundary case: image-unit risk feasible, transfer utility not positive",
        },
        {
            "source": "VisDrone val",
            "cluster_support": "image/sequence fail",
            "utility_result": "review boxes increase and precision decreases versus raw row",
            "interpretation": "no-go source-transfer diagnostic",
        },
    ]
    out = pd.DataFrame(rows)
    # Attach compact numeric support from public tables for reproducibility.
    out["numeric_source_csvs"] = "; ".join(
        [
            str(TABLE_DIR / "cluster_unit_feasibility_uavdt_iou0.25_summary.csv"),
            str(TABLE_DIR / "cluster_unit_feasibility_visdrone_iou0.25_summary.csv"),
            str(TABLE_DIR / "review_burden_simulation_v2.csv"),
        ]
    )
    out.to_csv(TABLE_DIR / "l1_transfer_boundary_map.csv", index=False)
    # Touch loaded frames so failed/missing schemas surface during execution.
    _ = (len(audit), len(vis), len(burden))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    L1_DIR.mkdir(parents=True, exist_ok=True)

    second = second_detector_table(force_cache=args.force_cache)
    method = method_baseline_table()
    review = review_budget_anchor_table()
    transfer = transfer_boundary_map()

    print("Second detector fixed-row audit:")
    print(
        second.loc[
            :,
            ["detector", "contract", "threshold", "unit", "loss_upper", "loss_pass", "object_risk", "precision", "fp_per_image"],
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nMethod baseline summary:")
    print(
        method.loc[
            :,
            ["baseline", "unit", "contract", "threshold", "eval_object_risk", "eval_precision", "eval_fp_img", "eval_risk_feasible"],
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nReview-budget anchor rows:", len(review))
    print("Transfer boundary rows:", len(transfer))


if __name__ == "__main__":
    main()
