#!/usr/bin/env python3
"""Build an AITOD YOLO-family frozen prediction cache.

This script is used only for the L1 second-detector stress check. It consumes
the existing AITOD ground-truth cache plus a YOLO-format AITOD validation image
directory, runs a trained Ultralytics YOLO-family model at fixed resolutions,
and emits the same cache schema used by the audit scripts.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO


DEFAULT_GT_CACHE = Path(
    os.environ.get("AITOD_GT_CACHE", "/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640")
)
DEFAULT_IMAGE_DIR = Path(
    os.environ.get("AITOD_VAL_IMAGE_DIR", "/root/zjh_UAV_detection/experiments/aitod/data/AI-TOD-YOLO/images/val")
)
DEFAULT_WEIGHTS = Path(
    os.environ.get(
        "AITOD_YOLO_WEIGHTS",
        "/root/zjh_UAV_detection/experiments/aitod/runs/aitod_d10_baseline2stage_s2/weights/best.pt",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-cache", type=Path, required=True)
    parser.add_argument("--imgsz", default="640,960")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_cache.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_cache / "pred_rows.parquet"
    if pred_path.exists() and not args.force:
        print(f"Cache exists; use --force to rebuild: {pred_path}")
        return

    for name in ("gt_rows.parquet", "image_meta.csv"):
        source = args.gt_cache / name
        if not source.exists():
            raise FileNotFoundError(source)
        target = args.out_cache / name
        if source.suffix == ".parquet":
            pd.read_parquet(source).to_parquet(target, index=False)
        else:
            pd.read_csv(source).to_csv(target, index=False)

    meta = pd.read_csv(args.gt_cache / "image_meta.csv")
    paths = [args.image_dir / str(name) for name in meta["img_name"].tolist()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing AITOD validation images, first missing: " + missing[0])

    img_to_id = dict(zip(meta["img_name"].astype(str), meta["img_id"].astype(int)))
    rows: list[dict] = []
    model = YOLO(str(args.weights))
    resolutions = [int(x.strip()) for x in args.imgsz.split(",") if x.strip()]
    for resolution in resolutions:
        print(f"Running {args.weights.name} at imgsz={resolution} on {len(paths)} images")
        for start in range(0, len(paths), max(1, int(args.chunk_size))):
            chunk = paths[start : start + max(1, int(args.chunk_size))]
            results = model.predict(
                source=[str(path) for path in chunk],
                imgsz=resolution,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                batch=args.batch,
                half=args.half,
                stream=True,
                verbose=False,
            )
            for offset, result in enumerate(results):
                # Ultralytics may replace list-input paths with temporary
                # names such as image0.jpg; the stream order follows input.
                img_name = chunk[offset].name
                img_id = int(img_to_id[img_name])
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                conf = result.boxes.conf.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, klass in zip(xyxy, conf, cls):
                    rows.append(
                        {
                            "img_id": img_id,
                            "img_name": img_name,
                            "resolution": int(resolution),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "score": float(score),
                            "cls": int(klass),
                        }
                    )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pred = pd.DataFrame(rows, columns=["img_id", "img_name", "resolution", "x1", "y1", "x2", "y2", "score", "cls"])
    pred = pred.sort_values(["img_id", "resolution", "score"], ascending=[True, True, False], kind="mergesort")
    pred.to_parquet(pred_path, index=False)
    print(f"Wrote {pred_path} with {len(pred)} predictions")


if __name__ == "__main__":
    main()
