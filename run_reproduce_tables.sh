#!/usr/bin/env bash
set -euo pipefail

echo "This repository includes selected submitted result CSVs."
echo "Inspect docs/claim_audit/PAPER_CLAIM_AUDIT.md for the table-to-file map."
echo "Rerunning the audit scripts requires compatible derived detector caches."

python - <<'PY'
from pathlib import Path
required = [
    "docs/claim_audit/PAPER_CLAIM_AUDIT.md",
    "output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_val_cache_summary.csv",
    "output/tables/review_burden_simulation_v2.csv",
    "output/tables/localization_tier_boundary_summary.csv",
    "paper/figures/risk_utility_diagnostics.pdf",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("Missing required availability files: " + ", ".join(missing))
print("Availability files present:", len(required))
PY
