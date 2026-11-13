#!/usr/bin/env python3
"""Merge raw bench CSV files into data/final.csv with unique labels.

The original CSV files are only read. The only file written is:
    data/final.csv

Each output row keeps all original columns, plus:
    label        unique experiment label for plotting/grouping
    source_file  original CSV file name

All repeated measurements are preserved; no median is computed here.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "final.csv"

# Source file -> unique label.
# T2/T3 both use variant "block", and all T6 files use variant "omp",
# so the label column is required to keep them distinct.
SOURCES = [
    ("T0_naive.csv", "T0_naive"),
    ("T1_ikj.csv", "T1_ikj"),
    ("T2_block.csv", "T2_block"),
    ("T3_block.csv", "T3_block"),
    ("T4_pack.csv", "T4_pack"),
    ("T5_simd.csv", "T5_simd"),
    ("T6_omp_t1.csv", "T6_omp_t1"),
    ("T6_omp_t2.csv", "T6_omp_t2"),
    ("T6_omp_t4.csv", "T6_omp_t4"),
    ("T6_omp_t8.csv", "T6_omp_t8"),
    # best_t4 is the default 64^3 run; best_b64_t4 is the sweep run.
    ("best_t4.csv", "best_default_t4"),
    ("best_t1.csv", "best_t1"),
    ("best_b32_t4.csv", "best_b32_t4"),
    ("best_b64_t4.csv", "best_b64_t4"),
    ("best_b128_t4.csv", "best_b128_t4"),
    ("best_b256_t4.csv", "best_b256_t4"),
]


def main() -> int:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for fname, label in SOURCES:
        path = DATA_DIR / fname
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"missing header: {path}", file=sys.stderr)
                return 1

            original_fields = list(reader.fieldnames)
            if fieldnames is None:
                fieldnames = ["label", "source_file", *original_fields]
            elif original_fields != fieldnames[2:]:
                print(f"header mismatch: {path}", file=sys.stderr)
                print(f"  expected: {fieldnames[2:]}", file=sys.stderr)
                print(f"  found:    {original_fields}", file=sys.stderr)
                return 1

            for row in reader:
                row["label"] = label
                row["source_file"] = fname
                rows.append(row)

    if fieldnames is None or not rows:
        print("no input rows", file=sys.stderr)
        return 1

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label"] for row in rows)
    print(f"wrote {OUT_PATH} ({len(rows)} data rows)")
    for _, label in SOURCES:
        print(f"  {label:18s} {counts[label]:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
