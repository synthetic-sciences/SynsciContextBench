#!/usr/bin/env python3
"""Generate a single benchmark chart in BixBench style from results.json."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "benchmarks" / "results" / "results.json").read_text())
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# ── BixBench-style palette ─────────────────────────────────────────────
BG        = "#F5F0E8"
BAR_1     = "#B8420F"   # dark red-brown
BAR_2     = "#A89070"   # warm tan
TXT_DARK  = "#2C2418"
TXT_MID   = "#6B5D4F"
GRID_CLR  = "#DDD6C8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": BG,
    "axes.labelcolor": TXT_DARK,
    "text.color": TXT_DARK,
    "xtick.color": TXT_DARK,
    "ytick.color": TXT_MID,
    "font.family": "sans-serif",
    "font.size": 13,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.6,
    "savefig.dpi": 200,
    "axes.grid": False,
})


def main():
    engines = ["Delphie", "Context7", "Nia"]

    # Metric 1: Retrieval MRR (custom benchmark, %)
    retrieval_mrr = [
        RESULTS["retrieval"]["synsc-context"]["mrr"] * 100,
        RESULTS["retrieval"]["context7"]["mrr"] * 100,
        RESULTS["retrieval"]["nia"]["mrr"] * 100,
    ]

    # Metric 2: Code QA Accuracy (custom benchmark, %)
    code_qa = [
        RESULTS["code_qa"]["synsc-context"]["accuracy"] * 100,
        RESULTS["code_qa"]["context7"]["accuracy"] * 100,
        RESULTS["code_qa"]["nia"]["accuracy"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.arange(len(engines))
    total_group_width = 0.52
    bar_width = total_group_width / 2
    gap = 0.04

    bars1 = ax.bar(
        x - (bar_width + gap) / 2, retrieval_mrr, bar_width,
        color=BAR_1, zorder=3, label="Retrieval MRR"
    )
    bars2 = ax.bar(
        x + (bar_width + gap) / 2, code_qa, bar_width,
        color=BAR_2, zorder=3, label="Code QA Accuracy"
    )

    # value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.1f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=TXT_DARK
            )

    # y-axis
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=12)
    for tick_val in [20, 40, 60, 80, 100]:
        ax.axhline(y=tick_val, color=GRID_CLR, linewidth=0.7, zorder=1)

    # x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(engines, fontsize=15, fontweight="bold")

    # spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    # title
    fig.text(0.06, 0.95, "SynSci Context Bench", fontsize=26, fontweight="bold",
             color=TXT_DARK, ha="left", va="top",
             fontfamily="sans-serif")
    fig.text(0.06, 0.90, "Code Context Engine Comparison \u2014 Custom Benchmarks",
             fontsize=14, color=TXT_MID, ha="left", va="top")

    # legend
    legend = ax.legend(
        loc="upper right", frameon=False, fontsize=12,
        handlelength=1.2, handleheight=0.9, labelspacing=0.4
    )
    for text in legend.get_texts():
        text.set_color(TXT_DARK)

    fig.savefig(OUT / "results.png", facecolor=BG)
    plt.close(fig)
    print("  OK  results.png")


if __name__ == "__main__":
    print("Generating chart...")
    main()
    print(f"Done! Saved to {OUT}/results.png")
