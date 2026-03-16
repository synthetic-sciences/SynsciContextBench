#!/usr/bin/env python3
"""Generate benchmark results chart from final results."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────
BG        = "#F5F0E8"
CLR_DELPH = "#B8420F"   # dark red-brown  — Delphi
CLR_CTX7  = "#A89070"   # warm tan        — Context7
CLR_NIA   = "#5C7A5E"   # muted green     — Nia
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
    "font.size": 12,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.6,
    "savefig.dpi": 200,
    "axes.grid": False,
})


def main():
    # Results from results_final (100 queries per engine per phase)
    metrics = [
        "Retrieval\nMRR",
        "CodeSearchNet\nMRR",
        "CoSQA\nMRR",
        "Adversarial\nDiscrimination",
        "Hallucination\nAvoidance",
    ]

    delphi  = [0.962, 0.865, 0.703, 0.560, 0.61]   # halluc avoidance = 1 - 0.39
    context7 = [0.790, 0.010, 0.110, 0.170, 0.55]   # 1 - 0.45
    nia      = [0.728, 0.040, 0.298, 0.140, 0.49]   # 1 - 0.51

    engines = ["Delphi", "Context7", "Nia"]
    colors  = [CLR_DELPH, CLR_CTX7, CLR_NIA]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(metrics))
    n_engines = 3
    total_width = 0.72
    bar_width = total_width / n_engines
    gap = 0.02

    for i, (engine, vals, color) in enumerate(zip(engines, [delphi, context7, nia], colors)):
        offset = (i - 1) * (bar_width + gap)
        bars = ax.bar(
            x + offset, [v * 100 for v in vals], bar_width,
            color=color, zorder=3, label=engine,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 1.2,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=TXT_DARK,
                )

    # y-axis
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=11)
    for tick_val in [20, 40, 60, 80, 100]:
        ax.axhline(y=tick_val, color=GRID_CLR, linewidth=0.7, zorder=1)

    # x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")

    # spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    # title
    fig.text(0.06, 0.96, "SynSci Context Bench", fontsize=26, fontweight="bold",
             color=TXT_DARK, ha="left", va="top")
    fig.text(0.06, 0.91,
             "3 engines \u00b7 8 phases \u00b7 100 queries/engine/phase \u00b7 LLM judge (Claude Sonnet 4.6)",
             fontsize=12, color=TXT_MID, ha="left", va="top")

    # legend
    legend = ax.legend(
        loc="upper right", frameon=False, fontsize=13,
        handlelength=1.2, handleheight=0.9, labelspacing=0.4,
        ncol=3,
    )
    for text in legend.get_texts():
        text.set_color(TXT_DARK)

    # footnote
    fig.text(0.06, -0.02,
             "All differences statistically significant (p<0.0001, Holm-corrected). "
             "Hallucination Avoidance = 1 \u2212 Hallucination Rate.",
             fontsize=9, color=TXT_MID, ha="left", va="top", style="italic")

    fig.savefig(OUT / "results.png", facecolor=BG)
    plt.close(fig)
    print(f"  OK  {OUT / 'results.png'}")


if __name__ == "__main__":
    print("Generating chart...")
    main()
    print("Done!")
