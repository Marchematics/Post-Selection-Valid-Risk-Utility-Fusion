#!/usr/bin/env bash
set -euo pipefail

# Validation-only cluster-ratio diagnostic. Expected outcome: abstention;
# this is the clean small-sample boundary reported in the Letter.
python scripts/cluster_ratio_ltt_audit.py \
  --dataset uavdt \
  --cache data/caches/uavdt_trainval_combined_cache \
  --cal-manifest data/manifests/uavdt_valonly/uavdt_val_imagehash_cal.csv \
  --eval-manifest data/manifests/uavdt_valonly/uavdt_val_imagehash_eval.csv \
  --splits valonly_imagehash \
  --unit image \
  --family-profile base \
  --alpha 0.16 \
  --beta-fp 150 \
  --bound-method eb \
  --grid-size 161 \
  --n-bound 259 \
  --out-prefix cluster_ratio_valonly_imagehash_nmax

# Train+val development stress row. This is not the manuscript headline because
# it mixes train/val sources and the image-hash split is not sequence-disjoint.
python scripts/cluster_ratio_ltt_audit.py \
  --dataset uavdt \
  --cache data/caches/uavdt_trainval_combined_cache \
  --cal-manifest data/manifests/uavdt_trainval/uavdt_trainval_imagehash_cal.csv \
  --eval-manifest data/manifests/uavdt_trainval/uavdt_trainval_imagehash_eval.csv \
  --splits trainval_imagehash90_nmax259 \
  --unit image \
  --family-profile base \
  --alpha 0.16 \
  --beta-fp 150 \
  --bound-method eb \
  --grid-size 161 \
  --n-bound 259 \
  --out-prefix cluster_ratio_trainval_imagehash90_nmax
