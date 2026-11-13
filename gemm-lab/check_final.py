#!/usr/bin/env python3
"""Validate data/final.csv and print a per-label summary.

Checks:
  - required columns exist
  - numeric fields parse correctly
  - every expected experiment label is present
  - rel_err <= K * FLT_EPSILON for every row
  - recorded GFLOPS agrees with 2*M*N*K/time_s

Only reads data/final.csv; writes nothing.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "final.csv"

EPS_FLOAT = 1.1920928955078125e-07
REQUIRED_COLUMNS = {
    "label", "source_file", "M", "N", "K", "threads",
    "block_m", "block_n", "block_k", "time_s", "gflops",
    "max_abs_err", "rel_err",
}

EXPECTED_LABELS = [
    "T0_naive",
    "T1_ikj",
    "T2_block",
    "T3_block",
    "T4_pack",
    "T5_simd",
    "T6_omp_t1",
    "T6_omp_t2",
    "T6_omp_t4",
    "T6_omp_t8",
    "best_default_t4",
    "best_t1",
    "best_b32_t4",
    "best_b64_t4",
    "best_b128_t4",
    "best_b256_t4",
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


def main() -> int:
    if not CSV_PATH.is_file():
        fail(f"missing file: {CSV_PATH}")
        return 1

    rows: list[dict[str, object]] = []
    errors: list[str] = []

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            fail("CSV has no header")
            return 1

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            fail(f"missing columns: {', '.join(sorted(missing))}")
            return 1

        for lineno, raw in enumerate(reader, start=2):
            try:
                m = int(raw["M"])
                n = int(raw["N"])
                k = int(raw["K"])
                time_s = float(raw["time_s"])
                gflops = float(raw["gflops"])
                rel_err = float(raw["rel_err"])
            except (TypeError, ValueError) as exc:
                errors.append(f"line {lineno}: bad numeric field ({exc})")
                continue

            if m <= 0 or n <= 0 or k <= 0:
                errors.append(f"line {lineno}: non-positive matrix size")
                continue
            if not math.isfinite(time_s) or time_s <= 0:
                errors.append(f"line {lineno}: invalid time_s={time_s}")
                continue
            if not math.isfinite(rel_err):
                errors.append(f"line {lineno}: invalid rel_err={rel_err}")
                continue

            threshold = k * EPS_FLOAT
            expected_gflops = 2.0 * m * n * k / (time_s * 1e9)
            gflops_diff = abs(gflops - expected_gflops) / expected_gflops

            rows.append({
                "label": raw["label"],
                "source_file": raw["source_file"],
                "n": m,
                "threads": raw["threads"],
                "block": (raw["block_m"], raw["block_n"], raw["block_k"]),
                "time_s": time_s,
                "gflops": gflops,
                "rel_err": rel_err,
                "threshold": threshold,
                "ratio": rel_err / threshold,
                "gflops_diff": gflops_diff,
            })

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    if not rows:
        fail("no data rows")
        return 1

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["label"])].append(row)

    actual_labels = set(groups)
    expected_labels = set(EXPECTED_LABELS)
    missing_labels = sorted(expected_labels - actual_labels)
    extra_labels = sorted(actual_labels - expected_labels)

    print(f"file:  {CSV_PATH}")
    print(f"rows:  {len(rows)}")
    print(f"labels: {len(groups)}")
    print()

    header = (
        f"{'label':18s} {'rows':>5s} {'n-values':>27s} "
        f"{'max_rel':>10s} {'max_ratio':>10s} {'gflops_err':>10s} {'status':>6s}"
    )
    print(header)
    print("-" * len(header))

    bad_rows = 0
    for label in EXPECTED_LABELS:
        if label not in groups:
            continue
        items = groups[label]
        ns = sorted({int(row["n"]) for row in items})
        ns_text = ",".join(str(n) for n in ns)
        max_rel = max(float(row["rel_err"]) for row in items)
        max_ratio = max(float(row["ratio"]) for row in items)
        max_gflops_diff = max(float(row["gflops_diff"]) for row in items)
        ok = max_ratio <= 1.0 and max_gflops_diff <= 0.02
        if not ok:
            bad_rows += 1
        print(
            f"{label:18s} {len(items):5d} {ns_text:>27s} "
            f"{max_rel:10.2e} {max_ratio:10.3f} {max_gflops_diff:10.2%} "
            f"{'OK' if ok else 'FAIL':>6s}"
        )

    print()
    if missing_labels:
        fail(f"missing labels: {', '.join(missing_labels)}")
    if extra_labels:
        print(f"WARNING: unexpected labels: {', '.join(extra_labels)}", file=sys.stderr)
    if bad_rows:
        fail(f"{bad_rows} label(s) failed correctness/GFLOPS checks")
        return 1
    if missing_labels:
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
