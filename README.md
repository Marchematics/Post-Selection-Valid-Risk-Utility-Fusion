# Cluster-Aware Risk-Utility Audit for Low-IoU UAV Object-Presence Triage

This repository provides the code-availability package for the GRSL manuscript
**"Cluster-Aware Risk-Utility Audit for Low-IoU UAV Object-Presence Triage"**.

The repository contains audit scripts, selected result CSVs, a claim-to-result
map, and figure-generation material for the submitted Letter. It does not
redistribute raw UAV benchmark images.

## Contents

- `scripts/`: audit, finite-family replay, AP, localization-tier, review-burden,
  and figure-generation scripts.
- `output/tables/`: selected CSV summaries used by the manuscript tables and
  diagnostics.
- `paper/figures/risk_utility_diagnostics.pdf`: generated figure used in the
  manuscript.
- `docs/claim_audit/PAPER_CLAIM_AUDIT.md`: mapping from manuscript numbers to
  CSV outputs.
- `docs/DATA_AVAILABILITY.md`: data and code availability notes.
- `docs/SOURCE_DATA_MAP.md`: table/figure-to-file map.

## Main Evidence

The current GRSL claim is intentionally bounded:

- A fixed `nms040_cap300@0.125` 640+960 contract passes AITOD image, block, and
  sequence cluster-unit operational-loss audits at IoU 0.25.
- The same fixed row passes UAVDT image units but fails UAVDT block/sequence
  units and VisDrone image/sequence units.
- AITOD review boxes fall by 64.1% relative to raw RT-DETR-L/960, while
  precision rises from 0.1834 to 0.4886.
- IoU 0.35 is supportable on AITOD only with a higher-clutter row, and IoU 0.50
  remains unsupported at the same target.

The repository should therefore be read as code and result availability for a
cluster-aware operating-point audit, not as a deployment-safety certificate.

## Environment

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

Required Python packages are `numpy`, `pandas`, `scipy`, `matplotlib`, and
`pyarrow`.

## Reproducing Tables

The included CSVs are the submitted result summaries. To inspect the numerical
claim mapping:

```bash
sed -n '1,240p' docs/claim_audit/PAPER_CLAIM_AUDIT.md
```

The scripts can be rerun when compatible derived detector caches are available
locally. Raw public benchmark images are not included in this repository.
Selected commands and file outputs are listed in `docs/SOURCE_DATA_MAP.md`.
