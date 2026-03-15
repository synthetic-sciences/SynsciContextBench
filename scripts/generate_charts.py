#!/usr/bin/env python3
"""Generate a single benchmark comparison chart from results.json."""

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

# ── Theme ──────────────────────────────────────────────────────────────
BG       = "#0d1117"
CARD_BG  = "#161b22"
GRID     = "#21262d"
TXT      = "#e6edf3"
MUTED    = "#8b949e"
BORDER   = "#30363d"
DELPHIE  = "#58a6ff"
CTX7     = "#f78166"
NIA      = "#d2a8ff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": BORDER, "axes.labelcolor": TXT,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.35,
    "text.color": TXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "sans-serif", "font.size": 11,
    "legend.facecolor": CARD_BG, "legend.edgecolor": BORDER,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.5, "savefig.dpi": 200,
})

def clean_spine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def val_label(ax, bars, fmt=".2f", fs=8):
    for b in bars:
        h = b.get_height()
        if h > 0.01:
            ax.text(b.get_x() + b.get_width()/2, h + ax.get_ylim()[1]*0.015,
                    f"{h:{fmt}}", ha="center", va="bottom",
                    fontsize=fs, color=TXT, fontweight="bold")


def main():
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(BG)

    fig.suptitle("SynSci Context Bench",
                 fontsize=24, fontweight="bold", color=TXT, y=0.98)
    fig.text(0.5, 0.935,
             "Delphie  vs  Context7  vs  Nia   |   ~2,000 queries  \u00b7  8 benchmark suites  \u00b7  Claude Sonnet 4.6 judge",
             ha="center", fontsize=11, color=MUTED)

    gs = fig.add_gridspec(1, 3, wspace=0.28, top=0.88, bottom=0.10, left=0.05, right=0.97)

    # ── LEFT: Custom benchmarks (all 3 engines) ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cats = ["Retrieval\nMRR", "Multi-Hop\nCoverage", "Code QA\nAccuracy", "Adversarial\nAccuracy"]
    d = [RESULTS["retrieval"]["synsc-context"]["mrr"],
         RESULTS["multihop"]["synsc-context"]["hop_coverage"],
         RESULTS["code_qa"]["synsc-context"]["accuracy"],
         RESULTS["adversarial"]["synsc-context"]["accuracy"]]
    c = [RESULTS["retrieval"]["context7"]["mrr"],
         RESULTS["multihop"]["context7"]["hop_coverage"],
         RESULTS["code_qa"]["context7"]["accuracy"],
         RESULTS["adversarial"]["context7"]["accuracy"]]
    n = [RESULTS["retrieval"]["nia"]["mrr"],
         RESULTS["multihop"]["nia"]["hop_coverage"],
         RESULTS["code_qa"]["nia"]["accuracy"],
         RESULTS["adversarial"]["nia"]["accuracy"]]

    x = np.arange(len(cats))
    w = 0.24
    b1 = ax1.bar(x - w, d, w, color=DELPHIE, label="Delphie", alpha=0.92, zorder=3)
    b2 = ax1.bar(x,     c, w, color=CTX7,    label="Context7", alpha=0.92, zorder=3)
    b3 = ax1.bar(x + w, n, w, color=NIA,     label="Nia",      alpha=0.92, zorder=3)
    val_label(ax1, b1, ".2f", 7)
    val_label(ax1, b2, ".2f", 7)
    val_label(ax1, b3, ".2f", 7)
    ax1.set_xticks(x); ax1.set_xticklabels(cats, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Custom Benchmarks", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(fontsize=8, loc="upper right")
    clean_spine(ax1)

    # ── CENTER: Enhanced Judge 4D (debiased) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    dims = ["avg_relevance", "avg_completeness", "avg_specificity", "avg_faithfulness"]
    labels = ["Relevance", "Completeness", "Specificity", "Faithfulness"]
    ej = RESULTS["enhanced_judge"]["codesearchnet"]
    dv = [ej["synsc-context"][k] for k in dims]
    cv = [ej["context7"][k] for k in dims]

    x2 = np.arange(len(dims))
    w2 = 0.33
    b4 = ax2.bar(x2 - w2/2, dv, w2, color=DELPHIE, label="Delphie", alpha=0.92, zorder=3)
    b5 = ax2.bar(x2 + w2/2, cv, w2, color=CTX7,    label="Context7", alpha=0.92, zorder=3)
    val_label(ax2, b4); val_label(ax2, b5)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 2.8)
    ax2.set_title("CodeSearchNet \u2014 Debiased 4D Judge (497q)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("Score (0\u20133)", fontsize=10, color=MUTED)
    ax2.legend(fontsize=8, loc="upper right")
    clean_spine(ax2)

    # ── RIGHT: Win rates ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    csn = RESULTS["enhanced_judge"]["codesearchnet"]
    cos = RESULTS["enhanced_judge"]["cosqa"]
    labels_wr = ["CodeSearchNet\n(497q)", "CoSQA\n(500q)"]

    sw = [csn["synsc-context"]["win_count"], cos["synsc-context"]["win_count"]]
    cw = [csn["context7"]["win_count"],      cos["context7"]["win_count"]]
    tw = [csn["synsc-context"]["tie_count"], cos["synsc-context"]["tie_count"]]
    tots = [a+b+c for a,b,c in zip(sw, cw, tw)]
    sp = [a/t*100 for a,t in zip(sw, tots)]
    tp = [a/t*100 for a,t in zip(tw, tots)]
    cp = [a/t*100 for a,t in zip(cw, tots)]

    y = np.arange(len(labels_wr))
    ht = 0.45
    ax3.barh(y, sp, ht, color=DELPHIE, label="Delphie wins", zorder=3)
    ax3.barh(y, tp, ht, left=sp, color=GRID, label="Ties", zorder=3)
    ax3.barh(y, cp, ht, left=[a+b for a,b in zip(sp, tp)], color=CTX7, label="Context7 wins", zorder=3)
    for i,(s,t,c) in enumerate(zip(sp, tp, cp)):
        if s > 10: ax3.text(s/2, i, f"{s:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        if c > 10: ax3.text(s+t+c/2, i, f"{c:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax3.set_yticks(y); ax3.set_yticklabels(labels_wr, fontsize=10)
    ax3.set_xlim(0, 100)
    ax3.set_title("Head-to-Head Win Rates", fontsize=14, fontweight="bold", pad=12)
    ax3.legend(fontsize=8, loc="lower right")
    clean_spine(ax3)
    ax3.grid(axis="y", visible=False)

    fig.savefig(OUT / "results.png", facecolor=BG)
    plt.close(fig)
    print("  OK  results.png")


if __name__ == "__main__":
    print("Generating chart...")
    main()
    print(f"Done! Saved to {OUT}/results.png")
