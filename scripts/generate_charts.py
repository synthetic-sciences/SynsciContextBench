#!/usr/bin/env python3
"""Generate 3 benchmark visuals: 1 matplotlib comparison + 2 Gemini diagrams."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from google import genai
from google.genai import types
import base64

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "benchmarks" / "results" / "results.json").read_text())
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# ── Theme ──────────────────────────────────────────────────────────────
BG = "#0d1117"
CARD_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
TEXT_MUTED = "#8b949e"
BORDER = "#30363d"
SYNSC = "#58a6ff"
CTX7 = "#f78166"
NIA = "#d2a8ff"
GREEN = "#3fb950"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "font.family": "sans-serif",
    "font.size": 11,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": BORDER,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.4,
    "savefig.dpi": 200,
})


# ═══════════════════════════════════════════════════════════════════════
# CHART 1: The One Comparison Chart — multi-panel dashboard
# ═══════════════════════════════════════════════════════════════════════
def chart_comparison():
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    # Title
    fig.suptitle("SynSci Context Bench — Engine Comparison Dashboard",
                 fontsize=22, fontweight="bold", color=TEXT_COLOR, y=0.97)
    fig.text(0.5, 0.94, "synsc-context  vs  Context7  vs  Nia  |  ~2,000 queries across 8 benchmark suites",
             ha="center", fontsize=12, color=TEXT_MUTED)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, top=0.90, bottom=0.06)

    # ── Panel 1: Enhanced Judge (4D Debiased) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    dims = ["avg_relevance", "avg_completeness", "avg_specificity", "avg_faithfulness"]
    labels = ["Relevance", "Complete-\nness", "Specificity", "Faithful-\nness"]
    ej = RESULTS["enhanced_judge"]["codesearchnet"]
    synsc_v = [ej["synsc-context"][d] for d in dims]
    ctx7_v = [ej["context7"][d] for d in dims]

    x = np.arange(len(dims))
    w = 0.35
    b1 = ax1.bar(x - w/2, synsc_v, w, color=SYNSC, label="synsc-context", alpha=0.9, zorder=3)
    b2 = ax1.bar(x + w/2, ctx7_v, w, color=CTX7, label="Context7", alpha=0.9, zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.06,
                         f"{h:.2f}", ha="center", va="bottom",
                         fontsize=8, color=TEXT_COLOR, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 3.0)
    ax1.set_title("CodeSearchNet — 4D Debiased Judge", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Score (0-3)", fontsize=10, color=TEXT_MUTED)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: CoSQA Enhanced Judge ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ej2 = RESULTS["enhanced_judge"]["cosqa"]
    synsc_v2 = [ej2["synsc-context"][d] for d in dims]
    ctx7_v2 = [ej2["context7"][d] for d in dims]

    b3 = ax2.bar(x - w/2, synsc_v2, w, color=SYNSC, label="synsc-context", alpha=0.9, zorder=3)
    b4 = ax2.bar(x + w/2, ctx7_v2, w, color=CTX7, label="Context7", alpha=0.9, zorder=3)
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.06,
                         f"{h:.2f}", ha="center", va="bottom",
                         fontsize=8, color=TEXT_COLOR, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 3.0)
    ax2.set_title("CoSQA — 4D Debiased Judge", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Score (0-3)", fontsize=10, color=TEXT_MUTED)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3: Win Rates (stacked horizontal) ───────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    datasets_wr = ["CodeSearchNet\n(497q)", "CoSQA\n(500q)"]

    csn = RESULTS["enhanced_judge"]["codesearchnet"]
    cos = RESULTS["enhanced_judge"]["cosqa"]

    sw = [csn["synsc-context"]["win_count"], cos["synsc-context"]["win_count"]]
    cw = [csn["context7"]["win_count"], cos["context7"]["win_count"]]
    tw = [csn["synsc-context"]["tie_count"], cos["synsc-context"]["tie_count"]]
    totals = [s+c+t for s,c,t in zip(sw, cw, tw)]
    sw_pct = [s/t*100 for s,t in zip(sw, totals)]
    cw_pct = [c/t*100 for c,t in zip(cw, totals)]
    tw_pct = [t_/t*100 for t_,t in zip(tw, totals)]

    y = np.arange(len(datasets_wr))
    h = 0.45
    ax3.barh(y, sw_pct, h, color=SYNSC, label="synsc wins", zorder=3)
    ax3.barh(y, tw_pct, h, left=sw_pct, color=GRID_COLOR, label="Ties", zorder=3)
    ax3.barh(y, cw_pct, h, left=[s+t for s,t in zip(sw_pct, tw_pct)], color=CTX7, label="Context7 wins", zorder=3)

    for i, (s, t, c) in enumerate(zip(sw_pct, tw_pct, cw_pct)):
        if s > 12:
            ax3.text(s/2, i, f"{s:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        if c > 12:
            ax3.text(s + t + c/2, i, f"{c:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    ax3.set_yticks(y)
    ax3.set_yticklabels(datasets_wr, fontsize=10)
    ax3.set_xlim(0, 100)
    ax3.set_title("Head-to-Head Win Rates", fontsize=13, fontweight="bold", pad=10)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", visible=False)

    # ── Panel 4: Custom Benchmarks (3-engine) ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    benchmarks = ["Retrieval\nMRR", "Multi-Hop\nCoverage", "Code QA\nAccuracy", "Adversarial\nAccuracy"]
    synsc_custom = [
        RESULTS["retrieval"]["synsc-context"]["mrr"],
        RESULTS["multihop"]["synsc-context"]["hop_coverage"],
        RESULTS["code_qa"]["synsc-context"]["accuracy"],
        RESULTS["adversarial"]["synsc-context"]["accuracy"],
    ]
    ctx7_custom = [
        RESULTS["retrieval"]["context7"]["mrr"],
        RESULTS["multihop"]["context7"]["hop_coverage"],
        RESULTS["code_qa"]["context7"]["accuracy"],
        RESULTS["adversarial"]["context7"]["accuracy"],
    ]
    nia_custom = [
        RESULTS["retrieval"]["nia"]["mrr"],
        RESULTS["multihop"]["nia"]["hop_coverage"],
        RESULTS["code_qa"]["nia"]["accuracy"],
        RESULTS["adversarial"]["nia"]["accuracy"],
    ]

    x4 = np.arange(len(benchmarks))
    w4 = 0.25
    b5 = ax4.bar(x4 - w4, synsc_custom, w4, color=SYNSC, label="synsc-context", alpha=0.9, zorder=3)
    b6 = ax4.bar(x4, ctx7_custom, w4, color=CTX7, label="Context7", alpha=0.9, zorder=3)
    b7 = ax4.bar(x4 + w4, nia_custom, w4, color=NIA, label="Nia", alpha=0.9, zorder=3)
    for bars in [b5, b6, b7]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax4.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                         f"{h:.2f}", ha="center", va="bottom",
                         fontsize=7, color=TEXT_COLOR, fontweight="bold")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(benchmarks, fontsize=9)
    ax4.set_ylim(0, 1.15)
    ax4.set_title("Custom Benchmarks — 3 Engines", fontsize=13, fontweight="bold", pad=10)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ── Panel 5: Validated IR (CodeSearchNet + CoSQA MRR) ─────────────
    ax5 = fig.add_subplot(gs[1, 1])
    datasets_val = ["CodeSearchNet\nMRR", "CodeSearchNet\nNDCG@10", "CoSQA\nMRR", "CoSQA\nNDCG@10"]
    vr = RESULTS["validated_retrieval"]
    synsc_val = [
        vr["codesearchnet"]["synsc-context"]["mrr"],
        vr["codesearchnet"]["synsc-context"]["ndcg_at_10"],
        vr["cosqa"]["synsc-context"]["mrr"],
        vr["cosqa"]["synsc-context"]["ndcg_at_10"],
    ]
    nia_val = [
        vr["codesearchnet"]["nia"]["mrr"],
        vr["codesearchnet"]["nia"]["ndcg_at_10"],
        vr["cosqa"]["nia"]["mrr"],
        vr["cosqa"]["nia"]["ndcg_at_10"],
    ]
    ctx7_val = [
        vr["codesearchnet"]["context7"]["mrr"],
        vr["codesearchnet"]["context7"]["ndcg_at_10"],
        vr["cosqa"]["context7"]["mrr"],
        vr["cosqa"]["context7"]["ndcg_at_10"],
    ]

    x5 = np.arange(len(datasets_val))
    w5 = 0.25
    b8 = ax5.bar(x5 - w5, synsc_val, w5, color=SYNSC, label="synsc-context", alpha=0.9, zorder=3)
    b9 = ax5.bar(x5, ctx7_val, w5, color=CTX7, label="Context7", alpha=0.9, zorder=3)
    b10 = ax5.bar(x5 + w5, nia_val, w5, color=NIA, label="Nia", alpha=0.9, zorder=3)
    for bars in [b8]:
        for bar in bars:
            h_val = bar.get_height()
            if h_val > 0.05:
                ax5.text(bar.get_x() + bar.get_width()/2, h_val + 0.02,
                         f"{h_val:.3f}", ha="center", va="bottom",
                         fontsize=7, color=TEXT_COLOR, fontweight="bold")
    ax5.set_xticks(x5)
    ax5.set_xticklabels(datasets_val, fontsize=8)
    ax5.set_ylim(0, 1.15)
    ax5.set_title("Validated Retrieval (Industry Datasets)", fontsize=13, fontweight="bold", pad=10)
    ax5.legend(fontsize=8, loc="upper right")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # ── Panel 6: Latency + Hallucination combo ────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])

    # Latency bars
    lat_benchmarks = ["Retrieval", "Multi-Hop", "Code QA", "Adversarial"]
    synsc_lat = [
        RESULTS["retrieval"]["synsc-context"]["avg_latency_ms"] / 1000,
        RESULTS["multihop"]["synsc-context"]["avg_latency_ms"] / 1000,
        RESULTS["code_qa"]["synsc-context"]["avg_latency_ms"] / 1000,
        RESULTS["adversarial"]["synsc-context"]["avg_latency_ms"] / 1000,
    ]
    ctx7_lat = [
        RESULTS["retrieval"]["context7"]["avg_latency_ms"] / 1000,
        RESULTS["multihop"]["context7"]["avg_latency_ms"] / 1000,
        RESULTS["code_qa"]["context7"]["avg_latency_ms"] / 1000,
        RESULTS["adversarial"]["context7"]["avg_latency_ms"] / 1000,
    ]
    nia_lat = [
        RESULTS["retrieval"]["nia"]["avg_latency_ms"] / 1000,
        RESULTS["multihop"]["nia"]["avg_latency_ms"] / 1000,
        RESULTS["code_qa"]["nia"]["avg_latency_ms"] / 1000,
        RESULTS["adversarial"]["nia"]["avg_latency_ms"] / 1000,
    ]

    x6 = np.arange(len(lat_benchmarks))
    w6 = 0.25
    ax6.bar(x6 - w6, synsc_lat, w6, color=SYNSC, label="synsc-context", alpha=0.9, zorder=3)
    ax6.bar(x6, ctx7_lat, w6, color=CTX7, label="Context7", alpha=0.9, zorder=3)
    ax6.bar(x6 + w6, nia_lat, w6, color=NIA, label="Nia", alpha=0.9, zorder=3)
    ax6.set_xticks(x6)
    ax6.set_xticklabels(lat_benchmarks, fontsize=9)
    ax6.set_ylim(0, 14)
    ax6.set_ylabel("Latency (seconds)", fontsize=10, color=TEXT_MUTED)
    ax6.set_title("Avg Latency by Benchmark", fontsize=13, fontweight="bold", pad=10)
    ax6.legend(fontsize=8, loc="upper right")
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig.savefig(OUT / "comparison_dashboard.png", facecolor=BG)
    plt.close(fig)
    print("  OK  comparison_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════
# CHART 2 & 3: Gemini-generated diagrams
# ═══════════════════════════════════════════════════════════════════════
def generate_gemini_diagrams():
    client = genai.Client(api_key="REDACTED_API_KEY")

    # ── Diagram 1: Benchmark Pipeline ──────────────────────────────────
    print("  ... generating benchmark pipeline diagram with Gemini...")
    pipeline_prompt = """Create a clean, modern, visually striking infographic diagram on a dark background (#0d1117) showing the 4-phase evaluation pipeline of a code context benchmark. The design should be minimal, sleek, and use a dark tech aesthetic with blue (#58a6ff), orange (#f78166), and purple (#d2a8ff) accent colors against the dark background. White text.

The 4 phases should flow left to right or top to bottom:

Phase 1: "Custom Benchmarks" - icon of a code bracket. 55 hand-crafted queries. Tests: Retrieval, Multi-Hop, Code QA, Adversarial, Hallucination.

Phase 2: "Validated Datasets" - icon of a database. ~1000 queries from CoSQA + CodeSearchNet (industry-standard). Content-matched IR metrics.

Phase 3: "LLM-as-Judge" - icon of a brain/AI. Blind 3D scoring (Relevance, Completeness, Specificity). Claude Sonnet 4.6 judge.

Phase 4: "Enhanced Judge" - icon of a shield/checkmark. Position-debiased 4D scoring + Faithfulness. RAGAS metrics. The gold standard.

Show arrows between phases with labels like "address bias" between them. At the bottom, show the 3 engines being tested: synsc-context (blue), Context7 (orange), Nia (purple). Make it look like a premium tech company's documentation diagram. Clean lines, no clutter, modern font styling."""

    resp1 = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=pipeline_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in resp1.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img_bytes = part.inline_data.data
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            (OUT / "benchmark_pipeline.png").write_bytes(img_bytes)
            print("  OK  benchmark_pipeline.png")
            break

    # ── Diagram 2: How synsc-context works ─────────────────────────────
    print("  ... generating architecture diagram with Gemini...")
    arch_prompt = """Create a clean, modern technical architecture diagram on a dark background (#0d1117) showing how "synsc-context" (a code context engine for AI agents) works. Use a dark tech aesthetic with glowing blue (#58a6ff) accents and white text. Minimal, sleek, premium feel.

Show the flow in a clear pipeline:

LEFT SIDE - "Indexing Pipeline":
1. "Git Repository" (icon: folder/git) -> Arrow ->
2. "AST Parsing" (icon: tree structure) - Extracts functions, classes, symbols ->
3. "Smart Chunking" - Code-aware chunks that respect syntax boundaries ->
4. "Context Enrichment" - Adds file paths, function signatures, docstrings, scope info ->
5. "Embedding + Storage" (icon: database) - Vector embeddings stored in Supabase

RIGHT SIDE - "Search & Retrieval":
1. "Agent Query" (icon: robot/AI) - "Find where auth middleware validates tokens" ->
2. "Hybrid Search" - Combines vector similarity + full-text + symbol lookup ->
3. "Post-Retrieval Enrichment" - Adds enclosing function signature, docstring, preceding context ->
4. "Ranked Results" - Top-K relevant code chunks returned to the agent

Connect both sides with the database/storage in the middle. Show it as a cohesive system. Make it look like a diagram from a YC startup's pitch deck - clean, impressive, professional. No clutter."""

    resp2 = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=arch_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in resp2.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img_bytes = part.inline_data.data
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            (OUT / "architecture.png").write_bytes(img_bytes)
            print("  OK  architecture.png")
            break


if __name__ == "__main__":
    print("Generating charts...")
    chart_comparison()
    generate_gemini_diagrams()
    print(f"\nDone! 3 charts saved to {OUT}/")
