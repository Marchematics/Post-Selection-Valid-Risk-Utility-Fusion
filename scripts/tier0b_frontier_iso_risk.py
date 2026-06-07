#!/usr/bin/env python3
"""Dense iso-risk frontier for NMS+cap versus source-support.

This answers whether the source-support pass at alpha=0.16 is a true
risk-FP frontier shift or only a consequence of spending more risk budget.
The implementation computes one greedy full-score matching per image and then
evaluates all thresholds by counting prediction/TP scores above threshold.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import box_iou_matrix, cp_upper, filter_manifest, load_cache  # noqa: E402
from post_selection_family_audit import FAMILY, TABLE_DIR, apply_contract, fixed_thresholds, hoeffding_upper_bounded  # noqa: E402


ALPHA = 0.16
BETA = 150.0
CONFIDENCE = 0.95
FP_CAP = 300.0
GRID_SIZE = 4001
CONTRACT_NAMES = ("nms040_cap300", "support_floor")


def sequence_id(name: str) -> str:
    stem = Path(str(name)).stem
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) > 2 else stem


def full_greedy_scores(
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    iou_threshold: float = 0.25,
) -> dict:
    """Return score arrays sufficient to evaluate all thresholds.

    For each image, `pred_scores_by_image` contains all post-processed
    prediction scores. `tp_scores_by_image` contains scores of predictions
    that receive a greedy GT assignment when all predictions are considered.
    Since lower-score predictions cannot change higher-score assignments,
    threshold evaluation is:

    TP(t) = # assigned scores >= t
    FP_g(t) = # pred scores in image g >= t - # assigned scores in image g >= t
    misses(t) = n_gt - TP(t)
    """
    gt_groups = {int(k): v.loc[~v["is_ignore"].astype(bool)].copy() for k, v in gt_frame.groupby("img_id", sort=False)}
    pred_groups = {int(k): v.copy() for k, v in pred_frame.groupby("img_id", sort=False)}
    pred_scores_by_image: list[np.ndarray] = []
    tp_scores_by_image: list[np.ndarray] = []
    object_scores: list[float] = []
    image_names: list[str] = []

    for _, image in meta.sort_values("img_id", kind="mergesort").iterrows():
        img_id = int(image["img_id"])
        image_names.append(str(image["img_name"]))
        gt = gt_groups.get(img_id, gt_frame.iloc[0:0]).reset_index(drop=True)
        pred = pred_groups.get(img_id, pred_frame.iloc[0:0]).copy()
        pred = pred.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        pred_scores = pd.to_numeric(pred["score"], errors="coerce").dropna().to_numpy(dtype=float)
        assigned_scores: list[float] = []
        obj_scores = np.full(len(gt), -np.inf, dtype=float)
        if len(gt) and len(pred):
            gt_boxes = gt.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            pred_boxes = pred.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            ious = box_iou_matrix(pred_boxes, gt_boxes)
            pred_cls = pred["cls"].to_numpy()
            gt_cls = gt["cls"].to_numpy()
            ious = np.where(pred_cls[:, None] == gt_cls[None, :], ious, -1.0)
            matched = np.zeros(len(gt), dtype=bool)
            scores = pd.to_numeric(pred["score"], errors="coerce").to_numpy(dtype=float)
            for pred_idx in range(len(pred)):
                gt_idx = int(np.argmax(ious[pred_idx]))
                best = float(ious[pred_idx, gt_idx]) if len(gt) else -1.0
                if best >= float(iou_threshold) and not matched[gt_idx]:
                    matched[gt_idx] = True
                    obj_scores[gt_idx] = float(scores[pred_idx])
                    assigned_scores.append(float(scores[pred_idx]))
        object_scores.extend(obj_scores.tolist())
        pred_scores_by_image.append(np.sort(pred_scores))
        tp_scores_by_image.append(np.sort(np.asarray(assigned_scores, dtype=float)))

    return {
        "object_scores": np.asarray(object_scores, dtype=float),
        "pred_scores_by_image": pred_scores_by_image,
        "tp_scores_by_image": tp_scores_by_image,
        "image_names": image_names,
        "n_images": len(image_names),
    }


def count_ge(sorted_scores: np.ndarray, threshold: float) -> int:
    return int(len(sorted_scores) - np.searchsorted(sorted_scores, threshold, side="left"))


def counts_at(prepared: dict, threshold: float) -> dict:
    obj = prepared["object_scores"]
    tp = int(np.isfinite(obj).sum() if threshold <= -np.inf else (obj >= threshold).sum())
    n_gt = int(len(obj))
    pred_total = 0
    tp_total_by_image = 0
    fp_by_image = []
    for pred_scores, tp_scores in zip(prepared["pred_scores_by_image"], prepared["tp_scores_by_image"], strict=True):
        p = count_ge(pred_scores, threshold)
        t = count_ge(tp_scores, threshold)
        pred_total += p
        tp_total_by_image += t
        fp_by_image.append(max(0, p - t))
    return {
        "n_gt": n_gt,
        "tp": tp,
        "misses": n_gt - tp,
        "fp": pred_total - tp_total_by_image,
        "fp_by_image": np.asarray(fp_by_image, dtype=float),
    }


def dense_thresholds(prepared: dict) -> list[float]:
    vals = [0.0]
    object_scores = prepared["object_scores"]
    vals.extend(object_scores[np.isfinite(object_scores)].tolist())
    for arr in prepared["pred_scores_by_image"]:
        vals.extend(arr.tolist())
    vals.extend(np.linspace(0.0, 0.08, GRID_SIZE).tolist())
    vals.extend(fixed_thresholds(401))
    out = np.unique(np.asarray(vals, dtype=float))
    out = out[(out >= 0.0) & (out <= 1.0)]
    return [float(v) for v in out]


def run_contract(contract_name: str) -> pd.DataFrame:
    gt, pred, meta = load_cache(ROOT / "data" / "caches" / "uavdt" / "combined_cache", None)
    cal_manifest = TABLE_DIR / "uavdt_revision_image_lockbox_cal.csv"
    eval_manifest = TABLE_DIR / "uavdt_revision_image_lockbox_eval.csv"
    contract = {c.name: c for c in FAMILY}[contract_name]
    ppred = apply_contract(pred, meta, contract)
    cal_gt, cal_pred, cal_meta = filter_manifest(gt, ppred, meta, cal_manifest)
    eval_gt, eval_pred, eval_meta = filter_manifest(gt, ppred, meta, eval_manifest)
    cal = full_greedy_scores(cal_gt, cal_pred, cal_meta, iou_threshold=0.25)
    ev = full_greedy_scores(eval_gt, eval_pred, eval_meta, iou_threshold=0.25)

    m = len(FAMILY)
    threshold_count = len(fixed_thresholds(161))
    risk_conf = 1.0 - (1.0 - CONFIDENCE) / (2.0 * float(m))
    fp_conf = 1.0 - (1.0 - CONFIDENCE) / (2.0 * float(m) * float(threshold_count))
    rows = []
    for threshold in dense_thresholds(cal):
        cc = counts_at(cal, threshold)
        ec = counts_at(ev, threshold)
        cal_cp = cp_upper(cc["misses"], cc["n_gt"], risk_conf)
        cal_fp_upper = hoeffding_upper_bounded(cc["fp_by_image"], fp_conf, FP_CAP)
        eval_fp = int(ec["fp"])
        eval_tp = int(ec["tp"])
        rows.append(
            {
                "contract": contract_name,
                "threshold": float(threshold),
                "risk_confidence": risk_conf,
                "fp_confidence": fp_conf,
                "cal_cp": cal_cp,
                "cal_misses": int(cc["misses"]),
                "cal_n_gt": int(cc["n_gt"]),
                "cal_fp_upper": cal_fp_upper,
                "cal_fp_img": float(cc["fp_by_image"].mean()),
                "eval_risk": float(ec["misses"] / ec["n_gt"]),
                "eval_misses": int(ec["misses"]),
                "eval_n_gt": int(ec["n_gt"]),
                "eval_precision": float(eval_tp / (eval_tp + eval_fp)) if (eval_tp + eval_fp) else math.nan,
                "eval_fp_img": float(eval_fp / ev["n_images"]),
                "eval_fp_upper": hoeffding_upper_bounded(ec["fp_by_image"], CONFIDENCE, FP_CAP),
            }
        )
    return pd.DataFrame(rows)


def summarize(frontier: pd.DataFrame) -> pd.DataFrame:
    support = frontier.loc[frontier["contract"] == "support_floor"].copy()
    support_feas = support.loc[support["cal_cp"] <= ALPHA].copy()
    support_sel = support_feas.sort_values(["cal_fp_upper", "cal_cp"], ascending=[True, True]).iloc[0]
    target_cp = float(support_sel["cal_cp"])
    rows = []
    for contract, group in frontier.groupby("contract", sort=False):
        risk_feas = group.loc[group["cal_cp"] <= ALPHA].copy()
        if len(risk_feas):
            min_fp = risk_feas.sort_values(["cal_fp_upper", "cal_cp"], ascending=[True, True]).iloc[0]
            closest = risk_feas.iloc[(risk_feas["cal_cp"] - target_cp).abs().argsort().iloc[0]]
        else:
            min_fp = group.sort_values(["cal_cp", "cal_fp_upper"], ascending=[True, True]).iloc[0]
            closest = min_fp
        rows.append(
            {
                "contract": contract,
                "has_risk_feasible": bool(len(risk_feas)),
                "min_fp_under_alpha_threshold": float(min_fp["threshold"]),
                "min_fp_under_alpha_cp": float(min_fp["cal_cp"]),
                "min_fp_under_alpha_fp_u": float(min_fp["cal_fp_upper"]),
                "min_fp_under_alpha_eval_fp_img": float(min_fp["eval_fp_img"]),
                "min_fp_under_alpha_eval_risk": float(min_fp["eval_risk"]),
                "closest_to_support_cp_threshold": float(closest["threshold"]),
                "closest_to_support_cp": float(closest["cal_cp"]),
                "closest_to_support_fp_u": float(closest["cal_fp_upper"]),
                "closest_to_support_eval_fp_img": float(closest["eval_fp_img"]),
                "closest_to_support_eval_risk": float(closest["eval_risk"]),
                "support_target_cp": target_cp,
            }
        )
    return pd.DataFrame(rows)


def beta_sensitivity(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for beta in np.arange(140.0, 160.0001, 1.0):
        for contract, group in frontier.groupby("contract", sort=False):
            risk_feas = group.loc[group["cal_cp"] <= ALPHA]
            best = float(risk_feas["cal_fp_upper"].min()) if len(risk_feas) else math.nan
            rows.append({"beta": float(beta), "contract": contract, "passes": bool(np.isfinite(best) and best <= beta), "best_fp_u": best})
    return pd.DataFrame(rows)


def plot_frontier(frontier: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.55))
    colors = {"nms040_cap300": "#4C78A8", "support_floor": "#E45756"}
    labels = {"nms040_cap300": "NMS+cap", "support_floor": "source-support"}
    for contract, group in frontier.groupby("contract", sort=False):
        sub = group.sort_values("cal_cp").drop_duplicates("cal_cp")
        ax.plot(sub["cal_cp"], sub["cal_fp_upper"], lw=1.25, color=colors[contract], label=labels[contract])
    ax.axvline(ALPHA, color="#555555", lw=0.9, ls="--", label=r"$\alpha=.16$")
    ax.axhline(BETA, color="#777777", lw=0.9, ls=":", label=r"$\beta=150$")
    ax.set_xlim(0.145, 0.162)
    ax.set_ylim(125, 165)
    ax.set_xlabel("calibration CP upper")
    ax.set_ylabel("certified FP-U")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


def main() -> None:
    frames = [run_contract(name) for name in CONTRACT_NAMES]
    frontier = pd.concat(frames, ignore_index=True)
    out_frontier = TABLE_DIR / "tier0b_iso_risk_frontier_nms_vs_support.csv"
    out_summary = TABLE_DIR / "tier0b_iso_risk_frontier_summary.csv"
    out_beta = TABLE_DIR / "tier0b_beta_sensitivity_nms_vs_support.csv"
    out_fig = ROOT / "paper" / "figures" / "tier0b_iso_risk_frontier_nms_vs_support.pdf"
    frontier.to_csv(out_frontier, index=False)
    summary = summarize(frontier)
    summary.to_csv(out_summary, index=False)
    beta = beta_sensitivity(frontier)
    beta.to_csv(out_beta, index=False)
    plot_frontier(frontier, out_fig)
    print("Summary")
    print(summary.round(4).to_string(index=False))
    print("\nBeta sensitivity")
    print(beta.pivot(index="beta", columns="contract", values="passes").to_string())
    print(f"\nWrote {out_frontier}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_beta}")
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
