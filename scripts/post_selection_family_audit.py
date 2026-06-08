#!/usr/bin/env python3
"""Post-selection finite-family risk-utility audits for the GRSL letter.

The script keeps detector caches frozen and evaluates a predeclared finite
family of post-processing contracts.  Each contract may change source handling,
support-aware score transforms, class-aware NMS, and image caps.  Thresholds
are selected on the calibration split only; the selected contract is the
feasible candidate with the smallest calibration FP upper bound.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from certify_miss_risk import (  # noqa: E402
    box_iou_matrix,
    cp_upper,
    evaluate_object_threshold,
    filter_manifest,
    greedy_match_image,
    load_cache,
    object_match_table,
)
from fusion_guardrail_audit import cap_per_image, class_aware_nms  # noqa: E402


TABLE_DIR = ROOT / "output" / "tables"
UAVDT_CACHE = Path(os.environ.get("UAVDT_CACHE", ROOT / "data" / "caches" / "uavdt" / "combined_cache"))
AITOD_CACHE = Path(os.environ.get("AITOD_CACHE", ROOT / "data" / "caches" / "aitod" / "combined_cache"))
VISDRONE_CACHE = Path(os.environ.get("VISDRONE_CACHE", ROOT / "data" / "caches" / "visdrone" / "combined_cache"))
UAVDT_MANIFEST_DIR = Path(os.environ.get("UAVDT_MANIFEST_DIR", ROOT / "data" / "manifests" / "uavdt"))
AITOD_MANIFEST_DIR = Path(os.environ.get("AITOD_MANIFEST_DIR", ROOT / "data" / "manifests" / "aitod"))
VISDRONE_MANIFEST_DIR = Path(os.environ.get("VISDRONE_MANIFEST_DIR", ROOT / "data" / "manifests" / "visdrone"))


@dataclass(frozen=True)
class Contract:
    name: str
    resolution: int | None = None
    nms_iou: float | None = None
    max_det: int | None = None
    score_floor: float = 0.0
    support_iou: float | None = None
    dual_bonus: float = 0.0
    single_penalty_640: float = 1.0
    single_penalty_960: float = 1.0
    unsupported_floor: float = 0.0
    seq_center: float | None = None
    seq_bonus: float = 0.0
    seq_penalty: float = 1.0
    seq_unsupported_floor: float = 0.0


FAMILY: tuple[Contract, ...] = (
    Contract("raw640", resolution=640),
    Contract("raw960", resolution=960),
    Contract("union"),
    Contract("nms040", nms_iou=0.40),
    Contract("nms040_cap300", nms_iou=0.40, max_det=300),
    Contract("nms040_cap300_sf003", nms_iou=0.40, max_det=300, score_floor=0.003),
    Contract(
        "support_soft_a",
        nms_iou=0.40,
        max_det=300,
        support_iou=0.45,
        dual_bonus=0.04,
        single_penalty_640=0.80,
        single_penalty_960=0.90,
    ),
    Contract(
        "support_soft_b",
        nms_iou=0.40,
        max_det=300,
        support_iou=0.50,
        dual_bonus=0.06,
        single_penalty_640=0.70,
        single_penalty_960=0.85,
    ),
    Contract(
        "support_floor",
        nms_iou=0.40,
        max_det=300,
        support_iou=0.45,
        dual_bonus=0.06,
        single_penalty_640=0.70,
        single_penalty_960=0.85,
        unsupported_floor=0.015,
    ),
    Contract(
        "support_cap200",
        nms_iou=0.40,
        max_det=200,
        support_iou=0.45,
        dual_bonus=0.06,
        single_penalty_640=0.75,
        single_penalty_960=0.88,
    ),
)


TRACK_CONTRACTS: tuple[Contract, ...] = (
    Contract(
        "track_soft_a",
        nms_iou=0.40,
        max_det=300,
        seq_center=0.025,
        seq_bonus=0.04,
        seq_penalty=0.90,
    ),
    Contract(
        "track_soft_b",
        nms_iou=0.40,
        max_det=300,
        seq_center=0.035,
        seq_bonus=0.06,
        seq_penalty=0.85,
    ),
    Contract(
        "source_track_a",
        nms_iou=0.40,
        max_det=300,
        support_iou=0.45,
        dual_bonus=0.04,
        single_penalty_640=0.80,
        single_penalty_960=0.90,
        seq_center=0.030,
        seq_bonus=0.06,
        seq_penalty=0.90,
    ),
    Contract(
        "source_track_b",
        nms_iou=0.40,
        max_det=300,
        support_iou=0.45,
        dual_bonus=0.06,
        single_penalty_640=0.70,
        single_penalty_960=0.85,
        unsupported_floor=0.005,
        seq_center=0.040,
        seq_bonus=0.08,
        seq_penalty=0.95,
    ),
    Contract(
        "track_filter",
        nms_iou=0.40,
        max_det=300,
        seq_center=0.050,
        seq_bonus=0.08,
        seq_penalty=0.80,
        seq_unsupported_floor=0.005,
    ),
)


def contract_family(profile: str) -> tuple[Contract, ...]:
    if profile == "base":
        return FAMILY
    if profile == "track":
        return FAMILY + TRACK_CONTRACTS
    if profile == "track_only":
        return TRACK_CONTRACTS
    raise ValueError(f"unknown family profile: {profile}")


def fixed_thresholds(grid_size: int) -> list[float]:
    """Deterministic threshold grid used by the finite-family theorem."""
    base = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    anchors = np.asarray(
        [
            0.001,
            0.003,
            0.005,
            0.0075,
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
            0.035,
            0.04,
            0.045,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.25,
            0.35,
            0.50,
            0.75,
            0.90,
        ],
        dtype=np.float64,
    )
    return sorted(float(v) for v in np.unique(np.concatenate([base, anchors])) if np.isfinite(v))


def sequence_id(name: str) -> str:
    stem = Path(str(name)).stem
    if re.match(r"^\d{7}_", stem):
        return stem.split("_", 1)[0]
    return re.sub(r"_[0-9]+$", "", stem)


def sequence_order(name: str) -> int:
    nums = re.findall(r"\d+", Path(str(name)).stem)
    return int(nums[-1]) if nums else 0


def add_source_support(pred: pd.DataFrame, support_iou: float | None) -> pd.DataFrame:
    out = pred.copy()
    out["source_supported"] = False
    out["support_max_iou"] = 0.0
    if support_iou is None or "resolution" not in out.columns or out.empty:
        return out

    supported = pd.Series(False, index=out.index)
    max_iou = pd.Series(0.0, index=out.index, dtype=float)
    for (_, _), group in out.groupby(["img_id", "cls"], sort=False):
        if group["resolution"].nunique() < 2:
            continue
        boxes = group.loc[:, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
        resolutions = group["resolution"].to_numpy()
        ious = box_iou_matrix(boxes, boxes)
        np.fill_diagonal(ious, -1.0)
        other = resolutions[:, None] != resolutions[None, :]
        ious = np.where(other, ious, -1.0)
        local_max = ious.max(axis=1) if len(group) else np.zeros(0)
        idx = group.index
        supported.loc[idx] = local_max >= float(support_iou)
        max_iou.loc[idx] = np.maximum(0.0, local_max)
    out["source_supported"] = supported.to_numpy(dtype=bool)
    out["support_max_iou"] = max_iou.to_numpy(dtype=float)
    return out


def add_sequence_support(pred: pd.DataFrame, meta: pd.DataFrame, center_thr: float | None) -> pd.DataFrame:
    out = pred.copy()
    out["seq_supported"] = False
    if center_thr is None or out.empty:
        return out

    info = meta.loc[:, ["img_id", "img_name", "W", "H"]].drop_duplicates("img_id").copy()
    info["seq"] = info["img_name"].map(sequence_id)
    info["ord"] = info["img_name"].map(sequence_order)
    info = info.sort_values(["seq", "ord", "img_id"], kind="mergesort")
    out = out.merge(info.loc[:, ["img_id", "seq", "ord", "W", "H"]], on="img_id", how="left")
    out["cxn"] = ((out["x1"] + out["x2"]) * 0.5) / out["W"].astype(float).clip(lower=1.0)
    out["cyn"] = ((out["y1"] + out["y2"]) * 0.5) / out["H"].astype(float).clip(lower=1.0)

    supported = pd.Series(False, index=out.index)
    for (_, cls), group in out.groupby(["seq", "cls"], sort=False):
        if pd.isna(group["seq"].iloc[0]) or group["img_id"].nunique() < 2:
            continue
        group = group.sort_values(["ord", "img_id"], kind="mergesort")
        by_img = {int(k): v for k, v in group.groupby("img_id", sort=False)}
        img_order = list(by_img)
        neighbors: dict[int, set[int]] = {}
        for pos, img_id in enumerate(img_order):
            nb = set()
            if pos > 0:
                nb.add(img_order[pos - 1])
            if pos + 1 < len(img_order):
                nb.add(img_order[pos + 1])
            neighbors[img_id] = nb
        for img_id, g in by_img.items():
            nb_frames = [by_img[n] for n in neighbors[img_id] if n in by_img]
            if not nb_frames:
                continue
            nb = pd.concat(nb_frames, ignore_index=False)
            a = g.loc[:, ["cxn", "cyn"]].to_numpy(dtype=np.float64)
            b = nb.loc[:, ["cxn", "cyn"]].to_numpy(dtype=np.float64)
            dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
            supported.loc[g.index] = dist.min(axis=1) <= float(center_thr)
    out["seq_supported"] = supported.to_numpy(dtype=bool)
    return out.drop(columns=["seq", "ord", "W", "H", "cxn", "cyn"])


def apply_contract(pred: pd.DataFrame, meta: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    out = pred.copy()
    if contract.resolution is not None:
        out = out.loc[out["resolution"].astype(int) == int(contract.resolution)].copy()
    out = out.loc[pd.to_numeric(out["score"], errors="coerce") >= float(contract.score_floor)].copy()
    out = add_source_support(out, contract.support_iou)
    out = add_sequence_support(out, meta, contract.seq_center)

    score = pd.to_numeric(out["score"], errors="coerce").astype(float).to_numpy()
    if "source_supported" in out.columns:
        supported = out["source_supported"].astype(bool).to_numpy()
        res = out["resolution"].astype(int).to_numpy() if "resolution" in out.columns else np.zeros(len(out), dtype=int)
        score = np.where(supported, np.minimum(1.0, score + float(contract.dual_bonus)), score)
        score = np.where((~supported) & (res == 640), score * float(contract.single_penalty_640), score)
        score = np.where((~supported) & (res == 960), score * float(contract.single_penalty_960), score)
        keep = supported | (pd.to_numeric(out["score"], errors="coerce").astype(float).to_numpy() >= float(contract.unsupported_floor))
        out = out.loc[keep].copy()
        score = score[keep]
    if "seq_supported" in out.columns and contract.seq_center is not None:
        seq_supported = out["seq_supported"].astype(bool).to_numpy()
        orig_score = pd.to_numeric(out["score"], errors="coerce").astype(float).to_numpy()
        score = np.where(seq_supported, np.minimum(1.0, score + float(contract.seq_bonus)), score)
        score = np.where(seq_supported, score, score * float(contract.seq_penalty))
        keep = seq_supported | (orig_score >= float(contract.seq_unsupported_floor))
        out = out.loc[keep].copy()
        score = score[keep]
    out["score"] = np.clip(score, 0.0, 1.0)
    out = class_aware_nms(out, contract.nms_iou)
    out = cap_per_image(out, contract.max_det)
    return out.reset_index(drop=True)


def per_image_fp(
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    threshold: float,
    iou: float,
) -> pd.DataFrame:
    gt_groups = {int(k): v for k, v in gt_frame.groupby("img_id", sort=False)}
    pred_groups = {int(k): v for k, v in pred_frame.groupby("img_id", sort=False)}
    rows = []
    for _, image in meta.sort_values("img_id", kind="mergesort").iterrows():
        img_id = int(image["img_id"])
        gt = gt_groups.get(img_id, gt_frame.iloc[0:0])
        pred = pred_groups.get(img_id, pred_frame.iloc[0:0])
        matched, tp, fp = greedy_match_image(gt, pred, threshold=threshold, iou_threshold=iou, class_aware=True)
        n_gt = int(len(matched))
        rows.append(
            {
                "img_id": img_id,
                "img_name": image["img_name"],
                "seq": sequence_id(str(image["img_name"])),
                "n_gt": n_gt,
                "misses": int(n_gt - tp),
                "tp": int(tp),
                "fp": int(fp),
                "risk": float((n_gt - tp) / n_gt) if n_gt else np.nan,
            }
        )
    return pd.DataFrame(rows)


def hoeffding_upper_bounded(values: np.ndarray, confidence: float, bound: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    norm = np.clip(values / float(bound), 0.0, 1.0)
    rad = math.sqrt(math.log(1.0 / max(1e-12, 1.0 - float(confidence))) / (2.0 * len(norm)))
    return float(min(1.0, norm.mean() + rad) * float(bound))


def group_bad_bound(per_image: pd.DataFrame, *, alpha_group: float, beta_group: float, confidence: float) -> tuple[float, int, int]:
    group = (
        per_image.groupby("seq", sort=False)
        .agg(n_gt=("n_gt", "sum"), misses=("misses", "sum"), fp=("fp", "sum"), n_images=("img_id", "size"))
        .reset_index()
    )
    group["risk"] = np.where(group["n_gt"] > 0, group["misses"] / group["n_gt"], 0.0)
    group["fp_img"] = group["fp"] / group["n_images"].clip(lower=1)
    bad = (group["risk"] > float(alpha_group)) | (group["fp_img"] > float(beta_group))
    return cp_upper(int(bad.sum()), int(len(group)), confidence), int(bad.sum()), int(len(group))


def select_threshold(
    cal_objects: pd.DataFrame,
    cal_scores: np.ndarray,
    cal_pred: pd.DataFrame,
    *,
    alpha: float,
    confidence: float,
    grid_size: int,
) -> dict:
    selected = None
    rows = []
    for threshold in fixed_thresholds(grid_size):
        result = evaluate_object_threshold(cal_objects, cal_scores, threshold)
        upper = cp_upper(result.misses, result.n_gt, confidence)
        row = {
            "threshold": float(threshold),
            "cal_cp": upper,
            "cal_risk": result.miss_risk,
            "cal_misses": result.misses,
            "cal_n_gt": result.n_gt,
            "cal_tp": result.tp,
            "cal_fp": result.fp,
            "cal_precision": result.precision,
        }
        rows.append(row)
        if upper <= float(alpha):
            selected = row
    if selected is None:
        selected = min(rows, key=lambda r: (r["cal_cp"], -r["threshold"]))
        selected = {**selected, "threshold_feasible": False}
    else:
        selected = {**selected, "threshold_feasible": True}
    return selected


def run_split(
    *,
    dataset: str,
    cache: Path,
    cal_manifest: Path,
    eval_manifest: Path,
    split_name: str,
    alpha: float,
    alpha_select: float,
    beta_fp: float,
    confidence: float,
    family_correct: bool,
    iou: float,
    grid_size: int,
    fp_bound_cap: float,
    family: tuple[Contract, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt, pred, meta = load_cache(cache, None)
    cal_gt, _, cal_meta_raw = filter_manifest(gt, pred, meta, cal_manifest)
    eval_gt, _, eval_meta_raw = filter_manifest(gt, pred, meta, eval_manifest)
    m = len(family)
    threshold_count = len(fixed_thresholds(grid_size))
    if family_correct:
        risk_conf = 1.0 - (1.0 - float(confidence)) / (2.0 * float(m))
        fp_conf = 1.0 - (1.0 - float(confidence)) / (2.0 * float(m) * float(threshold_count))
    else:
        risk_conf = float(confidence)
        fp_conf = float(confidence)
    per_conf = min(risk_conf, fp_conf)
    candidate_rows = []
    eval_rows = []
    per_image_rows = []

    for contract in family:
        ppred = apply_contract(pred, meta, contract)
        cal_gt_c, cal_pred, cal_meta = filter_manifest(gt, ppred, meta, cal_manifest)
        eval_gt_c, eval_pred, eval_meta = filter_manifest(gt, ppred, meta, eval_manifest)
        cal_objects, cal_scores = object_match_table(cal_gt_c, cal_pred, cal_meta, iou_threshold=iou, class_aware=True)
        eval_objects, eval_scores = object_match_table(eval_gt_c, eval_pred, eval_meta, iou_threshold=iou, class_aware=True)
        selected = select_threshold(
            cal_objects,
            cal_scores,
            cal_pred,
            alpha=alpha_select,
            confidence=risk_conf,
            grid_size=grid_size,
        )
        threshold = float(selected["threshold"])
        cal_pi = per_image_fp(cal_gt_c, cal_pred, cal_meta, threshold=threshold, iou=iou)
        cal_fp_mean = float(cal_pi["fp"].mean()) if len(cal_pi) else float("nan")
        cal_fp_upper = hoeffding_upper_bounded(cal_pi["fp"].to_numpy(), fp_conf, fp_bound_cap)
        eval_result = evaluate_object_threshold(eval_objects, eval_scores, threshold)
        eval_pi = per_image_fp(eval_gt_c, eval_pred, eval_meta, threshold=threshold, iou=iou)
        eval_fp_upper = hoeffding_upper_bounded(eval_pi["fp"].to_numpy(), confidence, fp_bound_cap)
        gb_upper, gb_bad, gb_n = group_bad_bound(
            eval_pi,
            alpha_group=alpha,
            beta_group=beta_fp,
            confidence=confidence,
        )
        row = {
            "dataset": dataset,
            "split": split_name,
            "contract": contract.name,
            "family_size": m,
            "threshold_grid_size": threshold_count,
            "family_correct": family_correct,
            "confidence": confidence,
            "per_candidate_confidence": per_conf,
            "risk_confidence": risk_conf,
            "fp_confidence": fp_conf,
            "alpha": alpha,
            "alpha_select": alpha_select,
            "beta_fp": beta_fp,
            "iou": iou,
            "threshold": threshold,
            "threshold_feasible": bool(selected["threshold_feasible"]),
            "cal_cp": float(selected["cal_cp"]),
            "cal_risk": float(selected["cal_risk"]),
            "cal_misses": int(selected["cal_misses"]),
            "cal_n_gt": int(selected["cal_n_gt"]),
            "cal_tp": int(selected["cal_tp"]),
            "cal_fp": int(selected["cal_fp"]),
            "cal_precision": float(selected["cal_precision"]),
            "cal_images": int(len(cal_meta)),
            "cal_fp_img": cal_fp_mean,
            "cal_fp_upper": cal_fp_upper,
            "eval_risk": eval_result.miss_risk,
            "eval_misses": int(eval_result.misses),
            "eval_n_gt": int(eval_result.n_gt),
            "eval_tp": int(eval_result.tp),
            "eval_fp": int(eval_result.fp),
            "eval_cp": cp_upper(eval_result.misses, eval_result.n_gt, confidence),
            "eval_precision": eval_result.precision,
            "eval_images": int(len(eval_meta)),
            "eval_fp_img": float(eval_result.fp / len(eval_meta)) if len(eval_meta) else float("nan"),
            "eval_fp_upper": eval_fp_upper,
            "eval_pass": bool(eval_result.miss_risk <= alpha),
            "group_bad_cp": gb_upper,
            "group_bad": gb_bad,
            "group_n": gb_n,
            "contract_record": str(asdict(contract)),
        }
        row["cal_feasible"] = bool(row["threshold_feasible"] and row["cal_cp"] <= alpha_select)
        row["utility_feasible"] = bool(row["cal_fp_upper"] <= beta_fp)
        row["joint_feasible"] = bool(row["cal_feasible"] and row["utility_feasible"])
        candidate_rows.append(row)
        eval_rows.append(row)
        eval_pi = eval_pi.assign(dataset=dataset, split=split_name, contract=contract.name, threshold=threshold)
        per_image_rows.append(eval_pi)

    candidates = pd.DataFrame(candidate_rows)
    feasible = candidates.loc[candidates["joint_feasible"]].copy()
    if len(feasible):
        selected_idx = feasible.sort_values(
            ["cal_fp_upper", "cal_cp", "threshold"],
            ascending=[True, True, False],
            kind="mergesort",
        ).index[0]
    else:
        risk_feasible = candidates.loc[candidates["cal_feasible"]].copy()
        if len(risk_feasible):
            selected_idx = risk_feasible.sort_values(
                ["cal_fp_img", "cal_cp", "threshold"],
                ascending=[True, True, False],
                kind="mergesort",
            ).index[0]
        else:
            selected_idx = candidates.sort_values(
                ["cal_cp", "cal_fp_img"],
                ascending=[True, True],
                kind="mergesort",
            ).index[0]
    candidates["selected"] = False
    candidates.loc[selected_idx, "selected"] = True
    selected = candidates.loc[[selected_idx]].copy()
    selected["selection_mode"] = "joint_feasible_min_fp_upper" if bool(selected["joint_feasible"].iloc[0]) else (
        "risk_feasible_min_empirical_fp" if bool(selected["cal_feasible"].iloc[0]) else "no_risk_feasible_min_cp"
    )
    return candidates, selected, pd.concat(per_image_rows, ignore_index=True)


def manifest_for(dataset: str, split: str) -> tuple[Path, Path]:
    if dataset == "uavdt":
        if split.startswith("random"):
            idx = split.replace("random", "")
            return (
                UAVDT_MANIFEST_DIR / f"uavdt_val_random_half_{idx}_cal153.csv",
                UAVDT_MANIFEST_DIR / f"uavdt_val_random_half_{idx}_eval152.csv",
            )
        if split == "image_lockbox":
            return (TABLE_DIR / "uavdt_revision_image_lockbox_cal.csv", TABLE_DIR / "uavdt_revision_image_lockbox_eval.csv")
        if split == "sequence_lockbox":
            return (TABLE_DIR / "uavdt_revision_lockbox_cal.csv", TABLE_DIR / "uavdt_revision_lockbox_eval.csv")
        if split == "sequence":
            return (UAVDT_MANIFEST_DIR / "uavdt_val_seq_cal153.csv", UAVDT_MANIFEST_DIR / "uavdt_val_seq_eval152.csv")
    if dataset == "aitod":
        if split in {"val", "whole", "aitod_val_cache"}:
            return (AITOD_MANIFEST_DIR / "aitod_val_all.csv", AITOD_MANIFEST_DIR / "aitod_val_all.csv")
        if split in {"train", "aitod_train_cache"}:
            return (AITOD_MANIFEST_DIR / "aitod_train_all.csv", AITOD_MANIFEST_DIR / "aitod_train_all.csv")
    if dataset == "visdrone":
        if split == "image_lockbox":
            return (TABLE_DIR / "visdrone_revision_image_lockbox_cal.csv", TABLE_DIR / "visdrone_revision_image_lockbox_eval.csv")
        if split == "sequence":
            return (VISDRONE_MANIFEST_DIR / "visdrone_val_seq_cal274.csv", VISDRONE_MANIFEST_DIR / "visdrone_val_seq_eval274.csv")
    raise ValueError(f"unknown dataset/split: {dataset}/{split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uavdt", "aitod", "visdrone"], default="uavdt")
    parser.add_argument("--cache", type=Path, default=None, help="Optional cache override with gt_rows/pred_rows/image_meta.")
    parser.add_argument("--splits", default="random1,random2,random3,random4,random5,image_lockbox,sequence_lockbox")
    parser.add_argument("--alpha", type=float, default=0.16)
    parser.add_argument(
        "--alpha-select",
        type=float,
        default=None,
        help="Calibration CP target used for selection. Defaults to --alpha; set below --alpha for a margin variant.",
    )
    parser.add_argument("--beta-fp", type=float, default=150.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--family-correct", action="store_true")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=161)
    parser.add_argument("--fp-bound-cap", type=float, default=300.0)
    parser.add_argument("--out-prefix", default="post_selection_family")
    parser.add_argument("--family-profile", choices=["base", "track", "track_only"], default="base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha_select = float(args.alpha if args.alpha_select is None else args.alpha_select)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache is not None:
        cache = args.cache
    elif args.dataset == "uavdt":
        cache = UAVDT_CACHE
    elif args.dataset == "aitod":
        cache = AITOD_CACHE
    else:
        cache = VISDRONE_CACHE
    family = contract_family(args.family_profile)
    all_candidates = []
    all_selected = []
    all_per_image = []
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        cal_manifest, eval_manifest = manifest_for(args.dataset, split)
        candidates, selected, per_image = run_split(
            dataset=args.dataset,
            cache=cache,
            cal_manifest=cal_manifest,
            eval_manifest=eval_manifest,
            split_name=split,
            alpha=args.alpha,
            alpha_select=alpha_select,
            beta_fp=args.beta_fp,
            confidence=args.confidence,
            family_correct=args.family_correct,
            iou=args.iou,
            grid_size=args.grid_size,
            fp_bound_cap=args.fp_bound_cap,
            family=family,
        )
        all_candidates.append(candidates)
        all_selected.append(selected)
        all_per_image.append(per_image)

    suffix = f"{args.out_prefix}_{args.dataset}_a{args.alpha:g}"
    if abs(alpha_select - float(args.alpha)) > 1e-12:
        suffix += f"_sel{alpha_select:g}"
    suffix += f"_iou{args.iou:g}"
    if args.family_profile != "base":
        suffix += f"_{args.family_profile}"
    if args.family_correct:
        suffix += "_family"
    candidates = pd.concat(all_candidates, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    per_image = pd.concat(all_per_image, ignore_index=True)
    candidates.to_csv(TABLE_DIR / f"{suffix}_candidates.csv", index=False)
    selected.to_csv(TABLE_DIR / f"{suffix}_selected.csv", index=False)
    per_image.to_csv(TABLE_DIR / f"{suffix}_per_image.csv", index=False)
    print("Selected contracts:")
    cols = [
        "dataset",
        "split",
        "contract",
        "selection_mode",
        "threshold",
        "cal_cp",
        "risk_confidence",
        "fp_confidence",
        "alpha_select",
        "cal_misses",
        "cal_n_gt",
        "cal_fp_img",
        "cal_fp_upper",
        "eval_misses",
        "eval_n_gt",
        "eval_risk",
        "eval_precision",
        "eval_fp_img",
        "group_bad_cp",
        "group_bad",
        "group_n",
    ]
    print(selected.loc[:, cols].round(4).to_string(index=False))
    print(f"Wrote {TABLE_DIR / (suffix + '_selected.csv')}")


if __name__ == "__main__":
    main()
