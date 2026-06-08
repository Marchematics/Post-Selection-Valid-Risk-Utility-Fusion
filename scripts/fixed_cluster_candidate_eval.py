#!/usr/bin/env python3
"""Evaluate a fixed contract+threshold under cluster losses."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import filter_manifest, load_cache, object_match_table  # noqa: E402
from cluster_contract_search_fast import (  # noqa: E402
    AITOD_CACHE,
    UAVDT_CACHE,
    VISDRONE_CACHE,
    loss_values,
    manifest_pair,
    prepared_arrays,
    summarize,
    unit_counts_at,
)
from post_selection_family_audit import apply_contract, contract_family  # noqa: E402


TABLE_DIR = ROOT / "output" / "tables"


def find_contract(name: str, family_profile: str):
    for contract in contract_family(family_profile):
        if contract.name == name:
            return contract
    raise ValueError(f"contract {name!r} not in family {family_profile!r}")


def default_cache(dataset: str) -> Path:
    if dataset == "uavdt":
        return UAVDT_CACHE
    if dataset == "aitod":
        return AITOD_CACHE
    if dataset == "visdrone":
        return VISDRONE_CACHE
    raise ValueError(dataset)


def eval_one(args: argparse.Namespace, split: str) -> tuple[dict, pd.DataFrame]:
    cache = args.eval_cache if args.eval_cache is not None else (args.cache if args.cache is not None else default_cache(args.dataset))
    gt, pred, meta = load_cache(cache, None)
    contract = find_contract(args.contract, args.family_profile)
    ppred = apply_contract(pred, meta, contract)
    if args.whole_cache:
        eval_manifest = None
    elif args.eval_manifest is not None:
        eval_manifest = args.eval_manifest
    else:
        _, eval_manifest = manifest_pair(args.dataset, split)
    eval_gt, eval_pred, eval_meta = filter_manifest(gt, ppred, meta, eval_manifest)
    objects, _ = object_match_table(eval_gt, eval_pred, eval_meta, iou_threshold=args.iou, class_aware=True)
    arr = prepared_arrays(objects, eval_pred, eval_meta, args.unit, args.block_len, args.sequence_parser)
    counts = unit_counts_at(arr, args.threshold)
    vals, value_range = loss_values(counts, args)
    delta = 1.0 - float(args.confidence)
    row = {
        "dataset": args.dataset,
        "split": split,
        "contract": args.contract,
        "threshold": float(args.threshold),
        "unit": args.unit,
        "loss": args.loss,
        "target": float(args.target),
        "iou": float(args.iou),
        "confidence": float(args.confidence),
        "sequence_parser": args.sequence_parser,
        "cache": str(cache),
        "eval_manifest": str(eval_manifest) if eval_manifest is not None else "",
        "whole_cache": bool(args.whole_cache),
    }
    row.update(summarize(counts, vals, delta=delta, value_range=value_range, args=args))
    units = pd.DataFrame(
        {
            "miss_rate": counts["miss_rate"],
            "fp_per_image": counts["fp_per_image"],
            "n_gt": counts["n_gt_unit"],
            "misses": counts["misses_unit"],
            "tp": counts["tp_unit"],
            "fp": counts["fp_unit"],
            "n_images": counts["n_images_unit"],
        }
    )
    units = units.assign(
        dataset=args.dataset,
        split=split,
        contract=args.contract,
        threshold=float(args.threshold),
        unit=args.unit,
        loss=args.loss,
        iou=float(args.iou),
    )
    return row, units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uavdt", "aitod", "visdrone"], default="uavdt")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--eval-cache", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--whole-cache", action="store_true")
    parser.add_argument("--splits", default="sequence")
    parser.add_argument("--contract", default="raw960")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--family-profile", choices=["base", "track", "track_only"], default="base")
    parser.add_argument("--unit", choices=["image", "sequence", "block"], default="sequence")
    parser.add_argument("--block-len", type=int, default=30)
    parser.add_argument("--sequence-parser", choices=["legacy", "strict"], default="strict")
    parser.add_argument("--loss", choices=["miss", "operational"], default="operational")
    parser.add_argument("--target", type=float, default=0.16)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bound-method", choices=["eb", "hoeffding"], default="eb")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--lambda-miss", type=float, default=0.8)
    parser.add_argument("--lambda-fp", type=float, default=0.2)
    parser.add_argument("--fp-cap", type=float, default=300.0)
    parser.add_argument("--out-prefix", default="fixed_cluster_candidate_eval")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    unit_frames = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        row, units = eval_one(args, split)
        rows.append(row)
        unit_frames.append(units)
    summary = pd.DataFrame(rows)
    units = pd.concat(unit_frames, ignore_index=True)
    split_tag = "_".join(s.strip() for s in args.splits.split(",") if s.strip())
    suffix = (
        f"{args.out_prefix}_{args.dataset}_{args.contract}_t{args.threshold:g}_"
        f"{args.unit}_{args.loss}_iou{args.iou:g}_{split_tag}"
    )
    summary.to_csv(TABLE_DIR / f"{suffix}_summary.csv", index=False)
    units.to_parquet(TABLE_DIR / f"{suffix}_units.parquet", index=False)
    cols = [
        "dataset",
        "split",
        "contract",
        "threshold",
        "unit",
        "loss",
        "loss_mean",
        "loss_upper",
        "loss_pass",
        "object_risk",
        "precision",
        "fp_per_image",
        "unit_count",
    ]
    print(summary.loc[:, cols].round(4).to_string(index=False))
    print("pass count:", int(summary["loss_pass"].sum()), "/", len(summary))
    print(f"Wrote {TABLE_DIR / (suffix + '_summary.csv')}")


if __name__ == "__main__":
    main()
