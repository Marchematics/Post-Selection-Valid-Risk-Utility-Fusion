#!/usr/bin/env python3
"""Build positive evidence tables for risk-guarded UAV fusion.

The script is intentionally read-only with respect to detector caches.  It
combines existing random-half audit outputs into paired method-evidence tables:
raw RT-DETR-L/960 versus fused and guarded post-processing variants.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"


def split_name(value: str) -> str:
    value = str(value)
    return value.replace("random", "R")


def main() -> None:
    raw = pd.read_csv(TABLE_DIR / "uavdt_rtdetrl960_iou25_alpha16_random_summary.csv")
    union = pd.read_csv(TABLE_DIR / "uavdt_combined640960_iou25_alpha16_random_summary.csv")
    guarded = pd.read_csv(TABLE_DIR / "uavdt_fusion_guardrail_final_refine_splits.csv")
    capped = pd.read_csv(TABLE_DIR / "uavdt_guarded_fusion_capped_contract_splits.csv")

    guarded = guarded.loc[
        (guarded["nms_iou"].astype(str) == "0.4")
        & (guarded["max_det"].astype(str) == "none")
        & (guarded["alpha_select"].astype(float).round(3) == 0.152)
    ].copy()
    capped = capped.loc[capped["split"].str.startswith("random")].copy()

    frames = []
    frames.append(
        raw.assign(
            split=raw["split"].map(split_name),
            variant="Raw 960",
            risk=raw["eval_empirical_miss_risk"],
            precision=raw["eval_precision"],
            fp_per_image=raw["eval_fp"] / 152.0,
            cp=raw["cal_cp_upper"],
        ).loc[:, ["split", "variant", "cp", "risk", "precision", "fp_per_image"]]
    )
    frames.append(
        union.assign(
            split=union["split"].map(split_name),
            variant="640+960 union",
            risk=union["eval_risk"],
            precision=union["eval_precision"],
            fp_per_image=union["eval_fp"] / 152.0,
            cp=union["cal_cp_upper"],
        ).loc[:, ["split", "variant", "cp", "risk", "precision", "fp_per_image"]]
    )
    frames.append(
        guarded.assign(
            split=guarded["split"].map(split_name),
            variant="Guarded NMS",
            risk=guarded["eval_risk"],
            precision=guarded["eval_precision"],
            fp_per_image=guarded["eval_fp_per_image"],
            cp=guarded["cal_cp_upper"],
        ).loc[:, ["split", "variant", "cp", "risk", "precision", "fp_per_image"]]
    )
    frames.append(
        capped.assign(
            split=capped["split"].map(split_name),
            variant="Guarded NMS+cap",
            risk=capped["eval_risk"],
            precision=capped["eval_precision"],
            fp_per_image=capped["eval_fp_per_image"],
            cp=capped["cal_cp_upper"],
        ).loc[:, ["split", "variant", "cp", "risk", "precision", "fp_per_image"]]
    )

    paired = pd.concat(frames, ignore_index=True)
    raw_by_split = paired.loc[paired["variant"] == "Raw 960"].set_index("split")
    paired["delta_fp_pct_vs_raw"] = paired.apply(
        lambda row: 100.0
        * (row["fp_per_image"] - raw_by_split.loc[row["split"], "fp_per_image"])
        / raw_by_split.loc[row["split"], "fp_per_image"],
        axis=1,
    )
    paired["delta_precision_pct_vs_raw"] = paired.apply(
        lambda row: 100.0
        * (row["precision"] - raw_by_split.loc[row["split"], "precision"])
        / raw_by_split.loc[row["split"], "precision"],
        axis=1,
    )

    summary = (
        paired.groupby("variant", sort=False)
        .agg(
            cp=("cp", "mean"),
            risk=("risk", "mean"),
            worst_risk=("risk", "max"),
            precision=("precision", "mean"),
            fp_per_image=("fp_per_image", "mean"),
            delta_fp_pct_vs_raw=("delta_fp_pct_vs_raw", "mean"),
            delta_precision_pct_vs_raw=("delta_precision_pct_vs_raw", "mean"),
            split_risk_pass=("risk", lambda x: int((x <= 0.16).sum())),
            n_splits=("risk", "size"),
        )
        .reset_index()
    )

    capped_rows = paired.loc[paired["variant"] == "Guarded NMS+cap"].set_index("split")
    raw_rows = raw_by_split
    paired_gain = pd.DataFrame(
        {
            "split": capped_rows.index,
            "raw_precision": raw_rows.loc[capped_rows.index, "precision"].to_numpy(),
            "capped_precision": capped_rows["precision"].to_numpy(),
            "precision_gain_pct": capped_rows["delta_precision_pct_vs_raw"].to_numpy(),
            "raw_fp_per_image": raw_rows.loc[capped_rows.index, "fp_per_image"].to_numpy(),
            "capped_fp_per_image": capped_rows["fp_per_image"].to_numpy(),
            "fp_reduction_pct": -capped_rows["delta_fp_pct_vs_raw"].to_numpy(),
            "raw_risk": raw_rows.loc[capped_rows.index, "risk"].to_numpy(),
            "capped_risk": capped_rows["risk"].to_numpy(),
        }
    )

    local_robust = pd.read_csv(TABLE_DIR / "uavdt_fusion_guardrail_final_refine_summary.csv")
    local_robust = local_robust.loc[
        (local_robust["eval_pass_rate"] == 1.0)
        & (local_robust["alpha_select"].astype(float).between(0.150, 0.152))
    ].copy()
    local_robust = local_robust.sort_values(
        ["mean_fp_per_image", "mean_eval_risk", "mean_precision"],
        ascending=[True, True, False],
    ).head(5)

    paired.to_csv(TABLE_DIR / "positive_method_evidence_by_split.csv", index=False)
    summary.to_csv(TABLE_DIR / "positive_method_evidence_summary.csv", index=False)
    paired_gain.to_csv(TABLE_DIR / "positive_method_paired_gain.csv", index=False)
    local_robust.to_csv(TABLE_DIR / "positive_method_local_robustness.csv", index=False)

    print("Summary")
    print(summary.round(4).to_string(index=False))
    print("\nPaired gain: raw 960 -> guarded NMS+cap")
    print(paired_gain.round(4).to_string(index=False))
    print("\nLocal pass-rate-1.0 variants")
    cols = [
        "nms_iou",
        "alpha_select",
        "mean_eval_risk",
        "worst_eval_risk",
        "mean_precision",
        "mean_fp_per_image",
    ]
    print(local_robust.loc[:, cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
