#!/usr/bin/env python3
"""Repair public result-table metadata without changing metric values.

Some AITOD audit tables were generated through the original UAVDT-oriented
script path and therefore wrote ``dataset=uavdt`` even though their split names
and cache paths point to AITOD. This utility makes AITOD a first-class metadata
value in the released CSV/Parquet artifacts and records every changed file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"
AUDIT_PATH = TABLE_DIR / "metadata_corrections_audit.csv"


def is_aitod_artifact(path: Path, frame: pd.DataFrame) -> bool:
    if "dataset" not in frame.columns:
        return False
    text_parts = [path.name]
    for column in ("split", "cache", "cal_cache", "eval_cache", "eval_manifest", "cal_manifest"):
        if column in frame.columns:
            text_parts.extend(frame[column].astype(str).head(20).tolist())
    text = " ".join(text_parts).lower()
    return "aitod" in text and set(frame["dataset"].astype(str)) == {"uavdt"}


def update_frame(path: Path, kind: str) -> dict | None:
    if kind == "csv":
        frame = pd.read_csv(path)
    elif kind == "parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(kind)
    if not is_aitod_artifact(path, frame):
        return None
    before = sorted(frame["dataset"].astype(str).unique())
    frame = frame.copy()
    frame["dataset"] = "aitod"
    if kind == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return {
        "file": str(path.relative_to(ROOT)),
        "kind": kind,
        "rows": int(len(frame)),
        "dataset_before": "|".join(before),
        "dataset_after": "aitod",
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in sorted(TABLE_DIR.glob("*.csv")):
        changed = update_frame(path, "csv")
        if changed:
            rows.append(changed)
    for path in sorted(TABLE_DIR.glob("*.parquet")):
        changed = update_frame(path, "parquet")
        if changed:
            rows.append(changed)
    audit = pd.DataFrame(rows, columns=["file", "kind", "rows", "dataset_before", "dataset_after"])
    audit.to_csv(AUDIT_PATH, index=False)
    print(f"metadata files updated: {len(audit)}")
    print(f"wrote {AUDIT_PATH}")


if __name__ == "__main__":
    main()
