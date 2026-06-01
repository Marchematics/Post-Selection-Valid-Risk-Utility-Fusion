#!/usr/bin/env bash
set -euo pipefail

python scripts/post_selection_family_audit.py \
  --dataset uavdt \
  --splits random1,random2,random3,random4,random5,image_lockbox,sequence_lockbox \
  --alpha 0.16 \
  --iou 0.25 \
  --out-prefix post_selection_family

python scripts/post_selection_family_audit.py \
  --dataset uavdt \
  --splits image_lockbox \
  --alpha 0.16 \
  --iou 0.25 \
  --family-correct \
  --out-prefix post_selection_family

python scripts/post_selection_family_audit.py \
  --dataset uavdt \
  --splits sequence_lockbox \
  --alpha 0.16 \
  --alpha-select 0.0525 \
  --iou 0.25 \
  --family-profile track \
  --family-correct \
  --out-prefix post_selection_track_probe_fc

python scripts/post_selection_family_audit.py \
  --dataset uavdt \
  --splits random1,random2,random3,random4,random5,image_lockbox \
  --alpha 0.16 \
  --alpha-select 0.151 \
  --iou 0.25 \
  --out-prefix post_selection_margin

python scripts/post_selection_family_audit.py \
  --dataset visdrone \
  --splits image_lockbox,sequence \
  --alpha 0.20 \
  --iou 0.25 \
  --out-prefix post_selection_family

python scripts/post_selection_family_audit.py \
  --dataset visdrone \
  --splits image_lockbox,sequence \
  --alpha 0.25 \
  --iou 0.25 \
  --out-prefix post_selection_family

python scripts/post_selection_family_audit.py \
  --dataset visdrone \
  --splits image_lockbox,sequence \
  --alpha 0.29 \
  --iou 0.25 \
  --out-prefix post_selection_family_guarded_select

python scripts/post_selection_family_audit.py \
  --dataset visdrone \
  --splits image_lockbox,sequence \
  --alpha 0.30 \
  --iou 0.25 \
  --out-prefix post_selection_family

python scripts/post_selection_family_audit.py \
  --dataset uavdt \
  --splits image_lockbox \
  --alpha 0.25 \
  --iou 0.50 \
  --out-prefix post_selection_family_iou

python scripts/post_selection_family_audit.py \
  --dataset visdrone \
  --splits image_lockbox \
  --alpha 0.30 \
  --beta-fp 200 \
  --iou 0.50 \
  --out-prefix post_selection_family_iou

python paper/figures/gen_fig2_sequence_sensitivity.py
