#!/usr/bin/env python3
"""Summarize Tier-0 fusion-vs-NMS certification rows.

Inputs are the family-corrected candidate CSVs produced by
`post_selection_family_audit.py` on the clean UAVDT image-hash split.
The script adds exact 95% binomial risk intervals and an evaluation
class breakdown for the selected threshold of each contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import evaluate_threshold, filter_manifest, load_cache  # noqa: E402
from post_selection_family_audit import apply_contract, FAMILY, TABLE_DIR  # noqa: E402


ALPHA = 0.16
CONFIDENCE = 0.95
CLASS_NAMES = {0: "vehicle/main", 1: "rare-1", 2: "rare-2"}


def cp_interval(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    q = (1.0 - confidence) / 2.0
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(q, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(1.0 - q, k + 1, n - k))
    return lo, hi


def summarize_candidates() -> pd.DataFrame:
    rows = []
    for iou_tag in ["0.25", "0.5"]:
        path = TABLE_DIR / f"tier0_fusion_vs_nms_clean_uavdt_a0.16_iou{iou_tag}_family_candidates.csv"
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            lo, hi = cp_interval(int(r["eval_misses"]), int(r["eval_n_gt"]))
            rows.append(
                {
                    "iou": float(r["iou"]),
                    "contract": r["contract"],
                    "selected": bool(r["selected"]),
                    "risk_feas": bool(r["threshold_feasible"]),
                    "fp_feas": bool(r["utility_feasible"]),
                    "joint_feas": bool(r["joint_feasible"]),
                    "threshold": float(r["threshold"]),
                    "CP_U": float(r["cal_cp"]),
                    "FP_U_cert": float(r["cal_fp_upper"]),
                    "eval_risk": float(r["eval_risk"]),
                    "risk_CI95_lo": lo,
                    "risk_CI95_hi": hi,
                    "eval_precision": float(r["eval_precision"]),
                    "eval_FP_img": float(r["eval_fp_img"]),
                    "eval_FP_U": float(r["eval_fp_upper"]),
                    "eval_misses": int(r["eval_misses"]),
                    "eval_n": int(r["eval_n_gt"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "tier0_fusion_vs_nms_clean_summary.csv", index=False)
    return out


def class_breakdown() -> pd.DataFrame:
    gt, pred, meta = load_cache(ROOT / "data" / "caches" / "uavdt" / "combined_cache", None)
    _, _, eval_meta = filter_manifest(gt, pred, meta, TABLE_DIR / "uavdt_revision_image_lockbox_eval.csv")
    eval_ids = set(eval_meta["img_id"].astype(int).tolist())
    gt_eval = gt.loc[gt["img_id"].astype(int).isin(eval_ids)].copy()
    meta_eval = meta.loc[meta["img_id"].astype(int).isin(eval_ids)].copy()

    candidates = pd.concat(
        [
            pd.read_csv(TABLE_DIR / "tier0_fusion_vs_nms_clean_uavdt_a0.16_iou0.25_family_candidates.csv"),
            pd.read_csv(TABLE_DIR / "tier0_fusion_vs_nms_clean_uavdt_a0.16_iou0.5_family_candidates.csv"),
        ],
        ignore_index=True,
    )
    by_contract = {c.name: c for c in FAMILY}
    rows = []
    for _, r in candidates.iterrows():
        contract = by_contract[str(r["contract"])]
        ppred = apply_contract(pred, meta, contract)
        pred_eval = ppred.loc[ppred["img_id"].astype(int).isin(eval_ids)].copy()
        for cls in sorted(gt_eval["cls"].dropna().unique().tolist()):
            gt_c = gt_eval.loc[gt_eval["cls"] == cls].copy()
            pred_c = pred_eval.loc[pred_eval["cls"] == cls].copy()
            res = evaluate_threshold(
                gt_c,
                pred_c,
                threshold=float(r["threshold"]),
                iou_threshold=float(r["iou"]),
                class_aware=True,
            )
            lo, hi = cp_interval(res.misses, res.n_gt)
            rows.append(
                {
                    "iou": float(r["iou"]),
                    "contract": r["contract"],
                    "cls": int(cls),
                    "class_name": CLASS_NAMES.get(int(cls), f"class-{int(cls)}"),
                    "threshold": float(r["threshold"]),
                    "risk": res.miss_risk,
                    "risk_CI95_lo": lo,
                    "risk_CI95_hi": hi,
                    "precision": res.precision,
                    "fp_img": float(res.fp / len(meta_eval)) if len(meta_eval) else float("nan"),
                    "misses": int(res.misses),
                    "n_gt": int(res.n_gt),
                    "tp": int(res.tp),
                    "fp": int(res.fp),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "tier0_fusion_vs_nms_clean_by_class.csv", index=False)
    return out


def main() -> None:
    summary = summarize_candidates()
    by_class = class_breakdown()
    display_cols = [
        "iou",
        "contract",
        "selected",
        "joint_feas",
        "threshold",
        "CP_U",
        "FP_U_cert",
        "eval_risk",
        "risk_CI95_lo",
        "risk_CI95_hi",
        "eval_precision",
        "eval_FP_img",
    ]
    print("\nTier-0 clean image-hash summary")
    print(summary.loc[:, display_cols].round(4).to_string(index=False))
    print("\nBy-class rows for NMS/cap/support_floor")
    keep = by_class["contract"].isin(["nms040", "nms040_cap300", "support_floor"])
    print(
        by_class.loc[
            keep,
            ["iou", "contract", "class_name", "risk", "risk_CI95_lo", "risk_CI95_hi", "precision", "fp_img", "misses", "n_gt"],
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nWrote {TABLE_DIR / 'tier0_fusion_vs_nms_clean_summary.csv'}")
    print(f"Wrote {TABLE_DIR / 'tier0_fusion_vs_nms_clean_by_class.csv'}")


if __name__ == "__main__":
    main()
