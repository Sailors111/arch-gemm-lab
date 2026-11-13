#!/usr/bin/env python3
"""Plot the merged experiment data in data/final.csv.

Figures written to figures/:
    gflops_vs_n.png           throughput of the main variants
    speedup_vs_n.png          speedup relative to T0 naive
    t6_thread_scaling.png     OpenMP thread scaling and parallel efficiency
    best_block_size.png       block-size sensitivity of the best kernel
    throughput_n4096.png      ranked throughput at n = 4096

All series are medians over the repeated measurements in final.csv.
Usage:
    python3 plot_final.py [--csv data/final.csv] [--outdir figures]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

BASE_DIR = Path(__file__).resolve().parent

# Reference categorical palette (light surface), fixed slot order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
# Ordinal blue ramp for ordered categories (threads, block sizes).
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

LABELS = {
    "T0_naive": "T0 naive",
    "T1_ikj": "T1 i-k-j",
    "T2_block": "T2 block (kb=K)",
    "T3_block": "T3 block",
    "T4_pack": "T4 packing",
    "T5_simd": "T5 AVX2/FMA",
    "T6_omp_t1": "T6 OpenMP 1T",
    "T6_omp_t2": "T6 OpenMP 2T",
    "T6_omp_t4": "T6 OpenMP 4T",
    "T6_omp_t8": "T6 OpenMP 8T",
    "best_default_t4": "best (64^3, 4T)",
    "best_t1": "best (1T)",
    "best_b32_t4": "best block 32",
    "best_b64_t4": "best block 64",
    "best_b128_t4": "best block 128",
    "best_b256_t4": "best block 256",
}

MAIN_SERIES = [
    ("T0_naive", 0),
    ("T1_ikj", 1),
    ("T2_block", 2),
    ("T3_block", 3),
    ("T4_pack", 4),
    ("T5_simd", 5),
    ("T6_omp_t4", 6),
    ("best_default_t4", 7),
]


def load(csv_path: Path) -> tuple[dict, dict]:
    times: dict[tuple[str, int], list[float]] = defaultdict(list)
    gflops: dict[tuple[str, int], list[float]] = defaultdict(list)

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                label = row["label"]
                n = int(row["M"])
                t = float(row["time_s"])
                g = float(row["gflops"])
            except (KeyError, TypeError, ValueError):
                continue
            if t > 0 and g > 0:
                times[(label, n)].append(t)
                gflops[(label, n)].append(g)

    med_t = {k: statistics.median(v) for k, v in times.items()}
    med_g = {k: statistics.median(v) for k, v in gflops.items()}
    return med_t, med_g


def require(med: dict, label: str, ns: list[int]) -> None:
    missing = [n for n in ns if (label, n) not in med]
    if missing:
        raise SystemExit(f"{label}: missing matrix sizes {missing}")


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor", axis="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)


def x_axis(ax, ns: list[int], xlim=(110, 5600)) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(*xlim)
    ax.set_xlabel("matrix size n", color=INK2, fontsize=10)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def legend_row_major(ax, ncol: int, **kwargs) -> None:
    """Make a multi-column legend read left-to-right, top-to-bottom.

    Matplotlib fills legend columns top-to-bottom, which scrambles the
    experiment order. Reorder the handles so each visual row is sequential.
    """
    handles, labels = ax.get_legend_handles_labels()
    n = len(labels)
    nrows = (n + ncol - 1) // ncol
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < n
    ]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        ncol=ncol,
        **kwargs,
    )


def plot_gflops(med_g: dict, ns: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
    style_axes(ax)

    for label, slot in MAIN_SERIES:
        require(med_g, label, ns)
        ax.plot(
            ns,
            [med_g[(label, n)] for n in ns],
            color=CATEGORICAL[slot],
            marker=MARKERS[slot],
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linewidth=2.0,
            label=LABELS[label],
            zorder=3,
        )

    ax.set_yscale("log")
    yticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.15, 200)
    ax.set_ylabel("GFLOPS (log scale)", color=INK2, fontsize=10)
    ax.set_title("GEMM throughput vs matrix size", color=INK,
                 fontsize=12, fontweight="semibold", loc="left")

    # Selective end labels: the fastest and the baseline.
    for label, slot in (("best_default_t4", 7), ("T0_naive", 0)):
        y = med_g[(label, ns[-1])]
        ax.annotate(
            f"{y:.0f}" if y >= 10 else f"{y:.2f}",
            xy=(ns[-1], y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK2,
        )

    legend_row_major(
        ax, 4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.4,
        labelspacing=0.4,
    )
    x_axis(ax, ns)
    fig.subplots_adjust(bottom=0.24, top=0.9, left=0.1, right=0.97)
    save(fig, out / "gflops_vs_n.png")


def plot_speedup(med_t: dict, ns: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
    style_axes(ax)

    require(med_t, "T0_naive", ns)
    base = {n: med_t[("T0_naive", n)] for n in ns}

    series = [(label, slot) for label, slot in MAIN_SERIES if label != "T0_naive"]
    for label, slot in series:
        require(med_t, label, ns)
        ax.plot(
            ns,
            [base[n] / med_t[(label, n)] for n in ns],
            color=CATEGORICAL[slot],
            marker=MARKERS[slot],
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linewidth=2.0,
            label=LABELS[label],
            zorder=3,
        )

    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.annotate("1x (T0 baseline)", xy=(ns[0], 1.0), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color=MUTED)

    ax.set_yscale("log")
    yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v}x" for v in yticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.9, 700)
    ax.set_ylabel("speedup vs T0 naive (log scale)", color=INK2, fontsize=10)
    ax.set_title("Speedup relative to the naive i-j-k baseline", color=INK,
                 fontsize=12, fontweight="semibold", loc="left")

    y = base[ns[-1]] / med_t[("best_default_t4", ns[-1])]
    ax.annotate(f"{y:.0f}x", xy=(ns[-1], y), xytext=(8, 0),
                textcoords="offset points", va="center",
                fontsize=8.5, color=INK2)

    legend_row_major(
        ax, 4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.4,
        labelspacing=0.4,
    )
    x_axis(ax, ns)
    fig.subplots_adjust(bottom=0.24, top=0.9, left=0.1, right=0.97)
    save(fig, out / "speedup_vs_n.png")


def plot_threads(med_t: dict, med_g: dict, ns: list[int], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.8), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    for ax in (ax1, ax2):
        style_axes(ax)
        x_axis(ax, ns)
        # 3072 and 4096 crowd each other in the narrower panels.
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], rotation=35, ha="right",
                           fontsize=8)

    thread_specs = [
        ("T6_omp_t1", 1, ORDINAL_BLUE[0]),
        ("T6_omp_t2", 2, ORDINAL_BLUE[1]),
        ("T6_omp_t4", 4, ORDINAL_BLUE[2]),
        ("T6_omp_t8", 8, ORDINAL_BLUE[3]),
    ]
    for label, threads, color in thread_specs:
        require(med_g, label, ns)
        ax1.plot(
            ns,
            [med_g[(label, n)] for n in ns],
            color=color,
            marker="o",
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linewidth=2.0,
            label=f"{threads} thread{'s' if threads > 1 else ''}",
            zorder=3,
        )

    ax1.set_ylim(0, 88)
    ax1.set_ylabel("GFLOPS", color=INK2, fontsize=10)
    ax1.set_title("OpenMP thread scaling", color=INK,
                  fontsize=11.5, fontweight="semibold", loc="left")
    ax1.legend(loc="upper right", frameon=False, fontsize=8.5,
               handlelength=2.2, labelspacing=0.4)

    require(med_t, "T6_omp_t1", ns)
    for label, p, color in [
        ("T6_omp_t2", 2, ORDINAL_BLUE[1]),
        ("T6_omp_t4", 4, ORDINAL_BLUE[2]),
        ("T6_omp_t8", 8, ORDINAL_BLUE[3]),
    ]:
        require(med_t, label, ns)
        eff = [
            med_t[("T6_omp_t1", n)] / med_t[(label, n)] / p
            for n in ns
        ]
        ax2.plot(
            ns,
            eff,
            color=color,
            marker="o",
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            linewidth=2.0,
            label=f"{p} threads",
            zorder=3,
        )

    ax2.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax2.annotate("ideal", xy=(ns[0], 1.0), xytext=(0, 5),
                 textcoords="offset points", fontsize=8, color=MUTED)
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax2.set_ylabel("parallel efficiency vs 1 thread", color=INK2, fontsize=10)
    ax2.set_title("Parallel efficiency", color=INK,
                  fontsize=11.5, fontweight="semibold", loc="left")
    ax2.legend(loc="upper right", frameon=False, fontsize=8.5,
               handlelength=2.2, labelspacing=0.4)

    fig.suptitle("T6 OpenMP: 4 physical cores / 8 logical threads",
                 color=INK, fontsize=12, fontweight="semibold", x=0.01,
                 ha="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.25, wspace=0.25)
    save(fig, out / "t6_thread_scaling.png")


def plot_blocks(med_g: dict, out: Path) -> None:
    sizes = [1024, 2048, 4096]
    blocks = [
        ("best_b32_t4", "32", ORDINAL_BLUE[0]),
        ("best_b64_t4", "64", ORDINAL_BLUE[1]),
        ("best_b128_t4", "128", ORDINAL_BLUE[2]),
        ("best_b256_t4", "256", ORDINAL_BLUE[3]),
    ]
    for label, _, _ in blocks:
        require(med_g, label, sizes)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(False, axis="x")

    x = list(range(len(sizes)))
    width = 0.18
    offsets = [-0.30, -0.10, 0.10, 0.30]

    series_vals: list[list[float]] = []
    for (label, bname, color), off in zip(blocks, offsets):
        vals = [med_g[(label, n)] for n in sizes]
        series_vals.append(vals)
        ax.bar(
            [xi + off for xi in x],
            vals,
            width=width,
            color=color,
            label=f"block {bname}",
            zorder=3,
        )

    # Label only the fastest block at each matrix size.
    for i, xi in enumerate(x):
        best_j = max(range(len(blocks)), key=lambda j: series_vals[j][i])
        val = series_vals[best_j][i]
        ax.annotate(
            f"{val:.0f}",
            xy=(xi + offsets[best_j], val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("matrix size n", color=INK2, fontsize=10)
    ax.set_ylabel("GFLOPS", color=INK2, fontsize=10)
    ax.set_ylim(0, 165)
    ax.set_title("Best kernel: block-size sensitivity (4 threads)", color=INK,
                 fontsize=12, fontweight="semibold", loc="left")
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8.5,
              handlelength=1.4, columnspacing=1.0, labelspacing=0.4)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.14)
    save(fig, out / "best_block_size.png")


def plot_throughput_4096(med_g: dict, out: Path) -> None:
    n = 4096
    series = [
        ("T0_naive", 0),
        ("T2_block", 2),
        ("T1_ikj", 1),
        ("T5_simd", 5),
        ("T3_block", 3),
        ("T6_omp_t4", 6),
        ("T4_pack", 4),
        ("best_default_t4", 7),
    ]
    for label, _ in series:
        require(med_g, label, [n])

    data = sorted(
        [(LABELS[label], med_g[(label, n)], slot) for label, slot in series],
        key=lambda item: item[1],
    )
    names = [item[0] for item in data]
    values = [item[1] for item in data]

    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(False, axis="y")

    y = list(range(len(data)))
    ax.barh(y, values, height=0.32, color=CATEGORICAL[0], zorder=3)
    for yi, val in zip(y, values):
        ax.annotate(
            f"{val:.2f}" if val < 10 else f"{val:.1f}",
            xy=(val, yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=9.5)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("GFLOPS (median over repetitions)", color=INK2, fontsize=10)
    ax.set_title("Throughput at n = 4096 (higher is better)", color=INK,
                 fontsize=12, fontweight="semibold", loc="left")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.9, bottom=0.14)
    save(fig, out / "throughput_n4096.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(BASE_DIR / "data" / "final.csv"))
    parser.add_argument("--outdir", default=str(BASE_DIR / "figures"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    out = Path(args.outdir)
    if not out.is_absolute():
        out = BASE_DIR / out

    if not csv_path.is_file():
        print(f"missing input: {csv_path}", file=sys.stderr)
        return 1

    med_t, med_g = load(csv_path)
    if not med_t:
        print(f"no usable rows in {csv_path}", file=sys.stderr)
        return 1

    ns = [128, 256, 512, 1024, 2048, 3072, 4096]
    plot_gflops(med_g, ns, out)
    plot_speedup(med_t, ns, out)
    plot_threads(med_t, med_g, ns, out)
    plot_blocks(med_g, out)
    plot_throughput_4096(med_g, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
