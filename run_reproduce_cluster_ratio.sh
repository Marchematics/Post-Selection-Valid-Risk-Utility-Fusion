#!/usr/bin/env bash
set -euo pipefail

python scripts/cluster_ratio_ltt_audit.py \
  --dataset uavdt \
  --cache data/caches/uavdt_trainval_combined_cache \
  --cal-manifest data/manifests/uavdt_trainval/uavdt_trainval_imagehash_cal.csv \
  --eval-manifest data/manifests/uavdt_trainval/uavdt_trainval_imagehash_eval.csv \
  --splits trainval_imagehash90 \
  --unit image \
  --family-profile base \
  --alpha 0.16 \
  --beta-fp 150 \
  --bound-method eb \
  --grid-size 161 \
  --out-prefix cluster_ratio_trainval_imagehash90_full
