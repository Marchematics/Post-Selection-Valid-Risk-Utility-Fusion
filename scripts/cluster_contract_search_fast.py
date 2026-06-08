#!/usr/bin/env python3
"""Fast cluster-unit contract search for TGRS evidence mining.

This development script selects contract+threshold using calibration cluster
loss bounds and evaluates the selected row on held-out clusters. It precomputes
per-object and per-prediction image/unit indices so threshold scans are cheap.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import filter_manifest, load_cache, object_match_table  # noqa: E402
from post_selection_family_audit import (  # noqa: E402
    Contract,
    apply_contract,
    contract_family,
    fixed_thresholds,
    sequence_id,
    sequence_order,
)


TABLE_DIR = ROOT / "output" / "tables"
UAVDT_CACHE = Path(
    "/root/zjh_UAV_detection/stride/outputs/uavdt/"
    "familyswitch_yolo11m640_to_rtdetrl640/val/combined_cache"
)
AITOD_CACHE = Path(
    "/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640"
)
VISDRONE_CACHE = Path(
    "/root/zjh_UAV_detection/stride/outputs/visdrone/"
    "familyswitch_yolo11m640_to_rtdetrl640/val/combined_cache"
)
UAVDT_MANIFEST_DIR = Path("/root/zjh_UAV_detection/reproducibility/manifests/uavdt")
AITOD_MANIFEST_DIR = Path("/root/zjh_UAV_detection/reproducibility/manifests/aitod")
VISDRONE_MANIFEST_DIR = Path("/root/zjh_UAV_detection/reproducibility/manifests/visdrone")


def manifest_pair(dataset: str, split: str) -> tuple[Path, Path]:
    if dataset == "uavdt":
        if split.startswith("random"):
            idx = split.replace("random", "")
            return (
                UAVDT_MANIFEST_DIR / f"uavdt_val_random_half_{idx}_cal153.csv",
                UAVDT_MANIFEST_DIR / f"uavdt_val_random_half_{idx}_eval152.csv",
            )
        if split in {"sequence", "seq"}:
            return UAVDT_MANIFEST_DIR / "uavdt_val_seq_cal153.csv", UAVDT_MANIFEST_DIR / "uavdt_val_seq_eval152.csv"
    if dataset == "aitod":
        if split in {"val", "whole", "aitod_val_cache"}:
            return AITOD_MANIFEST_DIR / "aitod_val_all.csv", AITOD_MANIFEST_DIR / "aitod_val_all.csv"
        if split in {"train", "aitod_train_cache"}:
            return AITOD_MANIFEST_DIR / "aitod_train_all.csv", AITOD_MANIFEST_DIR / "aitod_train_all.csv"
    if dataset == "visdrone":
        if split in {"sequence", "seq"}:
            return VISDRONE_MANIFEST_DIR / "visdrone_val_seq_cal274.csv", VISDRONE_MANIFEST_DIR / "visdrone_val_seq_eval274.csv"
    raise ValueError(f"unsupported dataset/split: {dataset}/{split}")


def strict_sequence_id(name: str) -> str:
    """Parse UAV sequence IDs after train/val prefixing.

    The legacy GRSL parser is intentionally preserved as an option, but it
    treats many UAVDT train names such as `train:DJI-405-720p00201.jpg` as
    separate sequences. For cluster-unit TGRS evidence, the stricter parser
    strips the split prefix and frame-number suffix.
    """

    stem = Path(str(name)).stem
    if ":" in stem:
        _, stem = stem.split(":", 1)
    if re.match(r"^\d{7}_", stem):
        return stem.split("_", 1)[0]
    stem = re.sub(r"_[0-9]+$", "", stem)
    stem = re.sub(r"(\d{3,5})$", "", stem).rstrip("_-")
    return stem


def parsed_sequence_id(name: str, parser: str) -> str:
    if parser == "legacy":
        return sequence_id(name)
    if parser == "strict":
        return strict_sequence_id(name)
    raise ValueError(parser)


def upper(values: np.ndarray, *, delta: float, value_range: float, method: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    if method == "hoeffding" or len(values) < 2:
        return float(values.mean() + value_range * math.sqrt(math.log(1.0 / max(delta, 1e-300)) / (2.0 * len(values))))
    if method == "eb":
        sample_var = float(np.var(values, ddof=1))
        log_term = math.log(2.0 / max(delta, 1e-300))
        rad = math.sqrt(2.0 * sample_var * log_term / len(values)) + 7.0 * value_range * log_term / (3.0 * (len(values) - 1))
        return float(values.mean() + rad)
    raise ValueError(method)


def unit_table(meta: pd.DataFrame, unit: str, block_len: int, sequence_parser: str) -> tuple[pd.DataFrame, dict[int, int]]:
    frame = meta.loc[:, ["img_id", "img_name"]].drop_duplicates("img_id").sort_values("img_id", kind="mergesort").copy()
    frame["seq"] = frame["img_name"].map(lambda x: parsed_sequence_id(str(x), sequence_parser))
    frame["ord"] = frame["img_name"].map(sequence_order)
    if unit == "image":
        frame["unit_id"] = frame["img_id"].astype(int).map(lambda x: f"img:{x}")
    elif unit == "sequence":
        frame["unit_id"] = frame["seq"].map(lambda x: f"seq:{x}")
    elif unit == "block":
        block = (frame["ord"].astype(int) // int(block_len)).astype(int)
        frame["unit_id"] = "blk:" + frame["seq"].astype(str) + ":" + block.astype(str)
    else:
        raise ValueError(unit)
    units = frame.loc[:, ["unit_id"]].drop_duplicates().reset_index(drop=True)
    units["unit_idx"] = np.arange(len(units), dtype=np.int64)
    frame = frame.merge(units, on="unit_id", how="left")
    return frame, dict(zip(frame["img_id"].astype(int), frame["unit_idx"].astype(int)))


def prepared_arrays(objects: pd.DataFrame, pred: pd.DataFrame, meta: pd.DataFrame, unit: str, block_len: int, sequence_parser: str) -> dict:
    image_units, img_to_unit = unit_table(meta, unit, block_len, sequence_parser)
    n_units = int(image_units["unit_idx"].max() + 1) if len(image_units) else 0
    img_ids = image_units["img_id"].astype(int).to_numpy()
    img_idx_map = {int(img_id): idx for idx, img_id in enumerate(img_ids)}
    img_unit_idx = image_units["unit_idx"].astype(int).to_numpy()

    obj = objects.copy()
    if len(obj):
        obj_img_idx = obj["img_id"].astype(int).map(img_idx_map).to_numpy(dtype=np.int64)
        obj_unit_idx = obj["img_id"].astype(int).map(img_to_unit).to_numpy(dtype=np.int64)
        obj_scores = pd.to_numeric(obj["match_score"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
    else:
        obj_img_idx = np.zeros(0, dtype=np.int64)
        obj_unit_idx = np.zeros(0, dtype=np.int64)
        obj_scores = np.zeros(0, dtype=np.float64)

    pred_frame = pred.copy()
    if len(pred_frame):
        pred_frame = pred_frame.loc[pred_frame["img_id"].astype(int).isin(img_idx_map)].copy()
        pred_img_idx = pred_frame["img_id"].astype(int).map(img_idx_map).to_numpy(dtype=np.int64)
        pred_scores = pd.to_numeric(pred_frame["score"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
    else:
        pred_img_idx = np.zeros(0, dtype=np.int64)
        pred_scores = np.zeros(0, dtype=np.float64)

    n_images = len(img_ids)
    n_gt_img = np.bincount(obj_img_idx, minlength=n_images).astype(np.int64) if len(obj_img_idx) else np.zeros(n_images, dtype=np.int64)
    n_gt_unit = np.bincount(obj_unit_idx, minlength=n_units).astype(np.int64) if len(obj_unit_idx) else np.zeros(n_units, dtype=np.int64)
    n_images_unit = np.bincount(img_unit_idx, minlength=n_units).astype(np.int64) if len(img_unit_idx) else np.zeros(n_units, dtype=np.int64)
    return {
        "n_units": n_units,
        "n_images": n_images,
        "obj_scores": obj_scores,
        "obj_img_idx": obj_img_idx,
        "obj_unit_idx": obj_unit_idx,
        "pred_scores": pred_scores,
        "pred_img_idx": pred_img_idx,
        "img_unit_idx": img_unit_idx,
        "n_gt_img": n_gt_img,
        "n_gt_unit": n_gt_unit,
        "n_images_unit": n_images_unit,
    }


def unit_counts_at(arr: dict, threshold: float) -> dict:
    n_images = arr["n_images"]
    n_units = arr["n_units"]
    detected_obj = arr["obj_scores"] >= float(threshold)
    tp_img = (
        np.bincount(arr["obj_img_idx"], weights=detected_obj.astype(np.int64), minlength=n_images).astype(np.int64)
        if len(arr["obj_img_idx"])
        else np.zeros(n_images, dtype=np.int64)
    )
    pred_kept_img = (
        np.bincount(arr["pred_img_idx"], weights=(arr["pred_scores"] >= float(threshold)).astype(np.int64), minlength=n_images).astype(np.int64)
        if len(arr["pred_img_idx"])
        else np.zeros(n_images, dtype=np.int64)
    )
    misses_img = arr["n_gt_img"] - tp_img
    fp_img = np.maximum(0, pred_kept_img - tp_img)
    misses_unit = np.bincount(arr["img_unit_idx"], weights=misses_img, minlength=n_units).astype(np.float64)
    tp_unit = np.bincount(arr["img_unit_idx"], weights=tp_img, minlength=n_units).astype(np.float64)
    fp_unit = np.bincount(arr["img_unit_idx"], weights=fp_img, minlength=n_units).astype(np.float64)
    n_gt_unit = arr["n_gt_unit"].astype(np.float64)
    n_images_unit = arr["n_images_unit"].astype(np.float64)
    miss_rate = np.divide(misses_unit, np.maximum(n_gt_unit, 1.0))
    fp_per_image = np.divide(fp_unit, np.maximum(n_images_unit, 1.0))
    return {
        "misses_unit": misses_unit,
        "tp_unit": tp_unit,
        "fp_unit": fp_unit,
        "n_gt_unit": n_gt_unit,
        "n_images_unit": n_images_unit,
        "miss_rate": miss_rate,
        "fp_per_image": fp_per_image,
    }


def loss_values(counts: dict, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    if args.loss == "miss":
        return counts["miss_rate"], 1.0
    fp_scaled = np.minimum(counts["fp_per_image"], float(args.fp_cap)) / float(args.fp_cap)
    return float(args.lambda_miss) * counts["miss_rate"] + float(args.lambda_fp) * fp_scaled, float(args.lambda_miss) + float(args.lambda_fp)


def summarize(counts: dict, values: np.ndarray, *, delta: float, value_range: float, args: argparse.Namespace) -> dict:
    misses = float(counts["misses_unit"].sum())
    tp = float(counts["tp_unit"].sum())
    fp = float(counts["fp_unit"].sum())
    n_gt = float(counts["n_gt_unit"].sum())
    n_images = float(counts["n_images_unit"].sum())
    loss_upper = upper(values, delta=delta, value_range=value_range, method=args.bound_method)
    return {
        "unit_count": int(len(values)),
        "image_count": int(n_images),
        "object_count": int(n_gt),
        "loss_mean": float(np.mean(values)) if len(values) else float("nan"),
        "loss_upper": float(loss_upper),
        "loss_pass": bool(loss_upper <= float(args.target)),
        "object_risk": float(misses / n_gt) if n_gt else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "fp_per_image": float(fp / n_images) if n_images else float("nan"),
        "median_fp_per_unit_image": float(np.median(counts["fp_per_image"])) if len(values) else float("nan"),
        "p90_fp_per_unit_image": float(np.quantile(counts["fp_per_image"], 0.90)) if len(values) else float("nan"),
        "share_units_fp_ge_100": float(np.mean(counts["fp_per_image"] >= 100.0)) if len(values) else float("nan"),
    }


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.dataset == "uavdt":
        default = UAVDT_CACHE
    elif args.dataset == "aitod":
        default = AITOD_CACHE
    else:
        default = VISDRONE_CACHE
    cache = args.cache if args.cache is not None else default
    cal_cache = args.cal_cache if args.cal_cache is not None else cache
    eval_cache = args.eval_cache if args.eval_cache is not None else cache
    cal_gt_all, cal_pred_all, cal_meta_all = load_cache(cal_cache, None)
    eval_gt_all, eval_pred_all, eval_meta_all = load_cache(eval_cache, None)
    family = contract_family(args.family_profile)
    if args.contracts:
        wanted = {name.strip() for name in args.contracts.split(",") if name.strip()}
        family = tuple(contract for contract in family if contract.name in wanted)
        missing = sorted(wanted - {contract.name for contract in family})
        if missing:
            raise ValueError(f"Unknown contracts for {args.family_profile}: {missing}")
        if not family:
            raise ValueError("--contracts filtered the family to zero contracts")
    thresholds = fixed_thresholds(args.grid_size)
    if args.max_thresholds and args.max_thresholds < len(thresholds):
        idx = np.linspace(0, len(thresholds) - 1, args.max_thresholds).round().astype(int)
        thresholds = [thresholds[i] for i in sorted(set(idx))]
    delta = 1.0 - float(args.confidence)
    n_candidates = len(family) * len(thresholds)
    cal_delta = delta / n_candidates if args.family_correct else delta

    all_candidates = []
    selected_rows = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if args.cal_manifest is None and args.eval_manifest is None and args.cal_cache is None and args.eval_cache is None:
            cal_manifest, eval_manifest = manifest_pair(args.dataset, split)
        else:
            cal_manifest, eval_manifest = args.cal_manifest, args.eval_manifest
        split_candidates = []
        prepared = []
        for contract in family:
            cal_ppred = apply_contract(cal_pred_all, cal_meta_all, contract)
            eval_ppred = apply_contract(eval_pred_all, eval_meta_all, contract)
            cal_gt, cal_pred, cal_meta = filter_manifest(cal_gt_all, cal_ppred, cal_meta_all, cal_manifest)
            eval_gt, eval_pred, eval_meta = filter_manifest(eval_gt_all, eval_ppred, eval_meta_all, eval_manifest)
            cal_objects, _ = object_match_table(cal_gt, cal_pred, cal_meta, iou_threshold=args.iou, class_aware=True)
            eval_objects, _ = object_match_table(eval_gt, eval_pred, eval_meta, iou_threshold=args.iou, class_aware=True)
            cal_arr = prepared_arrays(cal_objects, cal_pred, cal_meta, args.unit, args.block_len, args.sequence_parser)
            eval_arr = prepared_arrays(eval_objects, eval_pred, eval_meta, args.unit, args.block_len, args.sequence_parser)
            prepared.append((contract, eval_arr))
            for threshold in thresholds:
                counts = unit_counts_at(cal_arr, threshold)
                vals, value_range = loss_values(counts, args)
                row = {
                    "dataset": args.dataset,
                    "split": split,
                    "side": "cal",
                    "contract": contract.name,
                    "threshold": float(threshold),
                    "unit": args.unit,
                    "loss": args.loss,
                    "target": float(args.target),
                    "iou": float(args.iou),
                    "family_profile": args.family_profile,
                    "family_correct": bool(args.family_correct),
                    "cal_delta": float(cal_delta),
                    "cal_cache": str(cal_cache),
                    "eval_cache": str(eval_cache),
                    "cal_manifest": str(cal_manifest) if cal_manifest is not None else "",
                    "eval_manifest": str(eval_manifest) if eval_manifest is not None else "",
                    "contract_record": str(asdict(contract)),
                }
                row.update(summarize(counts, vals, delta=cal_delta, value_range=value_range, args=args))
                split_candidates.append(row)
        candidates = pd.DataFrame(split_candidates)
        feasible = candidates.loc[candidates["loss_pass"]].copy()
        if len(feasible):
            if args.selection_objective == "min_loss":
                selected_idx = feasible.sort_values(
                    ["loss_upper", "fp_per_image", "threshold"],
                    ascending=[True, True, False],
                    kind="mergesort",
                ).index[0]
                mode = "cal_cluster_feasible_min_loss"
            else:
                selected_idx = feasible.sort_values(
                    ["fp_per_image", "loss_upper", "threshold"],
                    ascending=[True, True, False],
                    kind="mergesort",
                ).index[0]
                mode = "cal_cluster_feasible_min_fp"
        else:
            selected_idx = candidates.sort_values(["loss_upper", "fp_per_image", "threshold"], ascending=[True, True, False], kind="mergesort").index[0]
            mode = "no_cal_cluster_feasible_min_upper"
        candidates["selected"] = False
        candidates.loc[selected_idx, "selected"] = True
        selected_cal = candidates.loc[selected_idx]
        contract_name = str(selected_cal["contract"])
        eval_arr = next(arr for contract, arr in prepared if contract.name == contract_name)
        eval_counts = unit_counts_at(eval_arr, float(selected_cal["threshold"]))
        eval_vals, eval_range = loss_values(eval_counts, args)
        eval_row = {
            "dataset": args.dataset,
            "split": split,
            "side": "eval",
            "contract": contract_name,
            "threshold": float(selected_cal["threshold"]),
            "unit": args.unit,
            "loss": args.loss,
            "target": float(args.target),
            "iou": float(args.iou),
            "selection_mode": mode,
            "cal_cache": str(cal_cache),
            "eval_cache": str(eval_cache),
            "cal_manifest": str(cal_manifest) if cal_manifest is not None else "",
            "eval_manifest": str(eval_manifest) if eval_manifest is not None else "",
            "cal_loss_mean": float(selected_cal["loss_mean"]),
            "cal_loss_upper": float(selected_cal["loss_upper"]),
            "cal_loss_pass": bool(selected_cal["loss_pass"]),
            "cal_fp_per_image": float(selected_cal["fp_per_image"]),
        }
        eval_row.update(summarize(eval_counts, eval_vals, delta=delta, value_range=eval_range, args=args))
        selected_rows.append(eval_row)
        all_candidates.append(candidates)
    return pd.concat(all_candidates, ignore_index=True), pd.DataFrame(selected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uavdt", "aitod", "visdrone"], default="uavdt")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cal-cache", type=Path, default=None)
    parser.add_argument("--eval-cache", type=Path, default=None)
    parser.add_argument("--cal-manifest", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--splits", default="sequence")
    parser.add_argument("--unit", choices=["image", "sequence", "block"], default="sequence")
    parser.add_argument("--block-len", type=int, default=30)
    parser.add_argument("--loss", choices=["miss", "operational"], default="miss")
    parser.add_argument("--target", type=float, default=0.16)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bound-method", choices=["eb", "hoeffding"], default="eb")
    parser.add_argument("--family-profile", choices=["base", "track", "track_only"], default="base")
    parser.add_argument("--contracts", default="", help="Optional comma-separated contract-name filter within the family profile.")
    parser.add_argument("--family-correct", action="store_true")
    parser.add_argument("--sequence-parser", choices=["legacy", "strict"], default="strict")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--max-thresholds", type=int, default=0)
    parser.add_argument("--lambda-miss", type=float, default=0.8)
    parser.add_argument("--lambda-fp", type=float, default=0.2)
    parser.add_argument("--fp-cap", type=float, default=300.0)
    parser.add_argument("--selection-objective", choices=["min_fp", "min_loss"], default="min_fp")
    parser.add_argument("--out-prefix", default="cluster_contract_search_fast")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    candidates, selected = run(args)
    split_tag = "_".join(s.strip() for s in args.splits.split(",") if s.strip())
    suffix = (
        f"{args.out_prefix}_{args.dataset}_{args.unit}_{args.loss}_iou{args.iou:g}_"
        f"{args.family_profile}_{args.bound_method}_{args.selection_objective}_{split_tag}"
        f"_{args.sequence_parser}"
        f"{'_fc' if args.family_correct else ''}"
    )
    candidates.to_csv(TABLE_DIR / f"{suffix}_candidates.csv", index=False)
    selected.to_csv(TABLE_DIR / f"{suffix}_selected.csv", index=False)
    cols = [
        "dataset",
        "split",
        "contract",
        "threshold",
        "unit",
        "loss",
        "selection_mode",
        "cal_loss_upper",
        "loss_mean",
        "loss_upper",
        "loss_pass",
        "object_risk",
        "precision",
        "fp_per_image",
    ]
    print(selected.loc[:, cols].round(4).to_string(index=False))
    print("eval pass count:", int(selected["loss_pass"].sum()), "/", len(selected))
    print(f"Wrote {TABLE_DIR / (suffix + '_selected.csv')}")


if __name__ == "__main__":
    main()
