#!/usr/bin/env python3
"""Build a deterministic combined-cache split for cluster-ratio audits."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def stable_bucket(text: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def sequence_id(name: str) -> str:
    stem = Path(str(name)).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def load_with_offset(cache: Path, *, offset: int, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_parquet(cache / "gt_rows.parquet").copy()
    pred = pd.read_parquet(cache / "pred_rows.parquet").copy()
    meta = pd.read_csv(cache / "image_meta.csv").copy()
    for frame in (gt, pred, meta):
        frame["img_id"] = frame["img_id"].astype(int) + int(offset)
        if "img_name" in frame.columns:
            frame["img_name"] = prefix + ":" + frame["img_name"].astype(str)
    meta["source_split"] = prefix
    return gt, pred, meta


def write_manifest(meta: pd.DataFrame, mask: pd.Series, path: Path, split_name: str) -> None:
    out = meta.loc[mask, ["img_id", "img_name"]].copy().sort_values("img_id", kind="mergesort").reset_index(drop=True)
    out.insert(0, "position", range(len(out)))
    out.insert(0, "split_name", split_name)
    out.insert(0, "dataset", "uavdt")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--out-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cal-pct", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_meta = pd.read_csv(args.train_cache / "image_meta.csv")
    offset = int(train_meta["img_id"].max()) + 1
    gt_train, pred_train, meta_train = load_with_offset(args.train_cache, offset=0, prefix="train")
    gt_val, pred_val, meta_val = load_with_offset(args.val_cache, offset=offset, prefix="val")
    gt = pd.concat([gt_train, gt_val], ignore_index=True)
    pred = pd.concat([pred_train, pred_val], ignore_index=True)
    meta = pd.concat([meta_train, meta_val], ignore_index=True).sort_values("img_id", kind="mergesort").reset_index(drop=True)

    args.out_cache.mkdir(parents=True, exist_ok=True)
    gt.to_parquet(args.out_cache / "gt_rows.parquet", index=False)
    pred.to_parquet(args.out_cache / "pred_rows.parquet", index=False)
    meta.to_csv(args.out_cache / "image_meta.csv", index=False)

    image_bucket = meta["img_name"].map(lambda x: stable_bucket(str(x)))
    image_cal = image_bucket < int(args.cal_pct)
    write_manifest(meta, image_cal, args.out_dir / "uavdt_trainval_imagehash_cal.csv", "uavdt_trainval_imagehash_cal")
    write_manifest(meta, ~image_cal, args.out_dir / "uavdt_trainval_imagehash_eval.csv", "uavdt_trainval_imagehash_eval")

    seq = meta["img_name"].map(sequence_id)
    seq_bucket = seq.map(lambda x: stable_bucket(str(x)))
    seq_cal = seq_bucket < int(args.cal_pct)
    write_manifest(meta, seq_cal, args.out_dir / "uavdt_trainval_seqhash_cal.csv", "uavdt_trainval_seqhash_cal")
    write_manifest(meta, ~seq_cal, args.out_dir / "uavdt_trainval_seqhash_eval.csv", "uavdt_trainval_seqhash_eval")

    print("Wrote combined cache:", args.out_cache)
    print("images", len(meta), "gt", int((~gt["is_ignore"].astype(bool)).sum()), "pred", len(pred))
    print("image cal/eval", int(image_cal.sum()), int((~image_cal).sum()))
    print("sequence cal/eval", int(seq_cal.sum()), int((~seq_cal).sum()), "unique seq", int(seq.nunique()))


if __name__ == "__main__":
    main()
