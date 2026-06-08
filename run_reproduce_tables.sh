#!/usr/bin/env bash
set -euo pipefail

echo "This repository includes selected submitted result CSVs."
echo "Inspect docs/claim_audit/PAPER_CLAIM_AUDIT.md for the table-to-file map."
echo "Rerunning the audit scripts requires compatible derived detector caches."

python - <<'PY'
from pathlib import Path
required = [
    "docs/claim_audit/PAPER_CLAIM_AUDIT.md",
    "docs/REPRODUCIBILITY_CARD.md",
    "output/tables/grsl_budget_constrained_main_summary.csv",
    "output/tables/l1_aitod_yolo11n_fixed_aitod_nms040_cap300_t0.125_image_operational_iou0.25_sequence_summary.csv",
    "output/tables/review_burden_simulation_v2.csv",
    "output/tables/topk_review_simulation.csv",
    "paper/figures/review_budget_tradeoff.pdf",
    "scripts/l1_clear_accept_evidence.py",
    "scripts/build_aitod_yolo_family_cache.py",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("Missing required availability files: " + ", ".join(missing))
print("Availability files present:", len(required))
PY
