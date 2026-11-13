#!/usr/bin/env python3
"""读取 bench 输出的 CSV，按中位数汇总并画图。

用法：python3 plot.py [data/raw.csv]
输出：figures/speedup_vs_n.png、figures/gflops_vs_n.png
"""
import csv
import os
import statistics
import sys


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["n"] = int(r["M"])
                r["time_s"] = float(r["time_s"])
                r["gflops"] = float(r["gflops"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw.csv"
    rows = load(path)
    if not rows:
        sys.exit(f"no usable rows in {path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib 未安装：sudo apt install python3-matplotlib")

    times, gflops = {}, {}
    for r in rows:
        key = (r["variant"], r["n"])
        times.setdefault(key, []).append(r["time_s"])
        gflops.setdefault(key, []).append(r["gflops"])
    med_t = {k: statistics.median(v) for k, v in times.items()}
    med_g = {k: statistics.median(v) for k, v in gflops.items()}

    variants = sorted({k[0] for k in med_t})
    ns = sorted({k[1] for k in med_t})
    base = {n: med_t.get(("naive", n)) for n in ns}

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        xs = [n for n in ns if (v, n) in med_t and base.get(n)]
        ys = [base[n] / med_t[(v, n)] for n in xs]
        if xs:
            ax.plot(xs, ys, marker="o", label=v)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("speedup vs naive")
    ax.set_title("GEMM speedup vs naive baseline")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/speedup_vs_n.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        xs = [n for n in ns if (v, n) in med_g]
        ys = [med_g[(v, n)] for n in xs]
        if xs:
            ax.plot(xs, ys, marker="o", label=v)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("GFLOPS")
    ax.set_title("GEMM throughput")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/gflops_vs_n.png", dpi=150)
    plt.close(fig)

    print("wrote figures/speedup_vs_n.png, figures/gflops_vs_n.png")


if __name__ == "__main__":
    main()
