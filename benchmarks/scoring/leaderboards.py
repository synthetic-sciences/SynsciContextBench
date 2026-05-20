"""Per-category leaderboards.

The single-line "Engine X wins" reporting buries the actual story: an engine
can win scoped code retrieval but lose Atlas workflow context entirely. The
diagnosis explicitly asked for separate leaderboards per category, with each
engine's strongest axis surfaced rather than a global average.

This module derives a small set of category leaderboards from a populated
``BenchmarkReport`` dict:

- ``code_retrieval``     — MRR over hand-curated retrieval + validated CSN
- ``docs_lookup``        — MRR over CoSQA + StackOverflow QA (real user
                           queries, often answered by documentation)
- ``paper_qa``           — Composite over Atlas ``paper_qa`` and ``synthesis``
                           categories (paper-flavored cases)
- ``atlas_graph``       — Composite over Atlas ``graph_memory``,
                           ``prior_decision``, ``avoid_repeat``, ``multi_turn``
- ``tool_contract``      — Composite over Atlas ``tool_contract`` category
- ``swe_patch``          — SWE-Agent judge composite (and pass-rate)
- ``hallucination``      — Lower is better; reported as 1 - hallucination_rate
                           so the leaderboard is "higher = better" consistent.

A leaderboard is only included if at least one engine produced data for it.
"""

from __future__ import annotations

from typing import Any


def _safe(d: dict | None, *keys, default: Any = 0.0) -> Any:
    """Walk a dict, returning `default` if any segment is missing."""
    cur: Any = d or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _entry(engine: str, score: float, **meta) -> dict:
    return {"engine": engine, "score": round(float(score), 4), **meta}


def _validated_dataset(report: dict, engine: str, dataset_suffix: str) -> dict | None:
    """Locate a per-dataset validated result for an engine."""
    for key, val in (report.get("validated") or {}).items():
        if not isinstance(val, dict):
            continue
        if val.get("engine") == engine and key.endswith(dataset_suffix):
            return val
    return None


def _build_code_retrieval(report: dict, engines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for e in engines:
        sources: list[tuple[str, float]] = []
        retrieval = (report.get("retrieval") or {}).get(e)
        if retrieval:
            sources.append(("retrieval", float(retrieval.get("avg_mrr", 0.0))))
        csn = _validated_dataset(report, e, "codesearchnet_benchmark")
        if csn:
            sources.append(("codesearchnet", float(csn.get("avg_mrr", 0.0))))
        advtest = _validated_dataset(report, e, "advtest_benchmark")
        if advtest:
            sources.append(("advtest", float(advtest.get("avg_mrr", 0.0))))
        if not sources:
            continue
        avg = sum(s for _, s in sources) / len(sources)
        rows.append(_entry(e, avg, components={k: round(v, 4) for k, v in sources}))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _build_docs_lookup(report: dict, engines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for e in engines:
        sources: list[tuple[str, float]] = []
        cosqa = _validated_dataset(report, e, "cosqa_benchmark")
        if cosqa:
            sources.append(("cosqa", float(cosqa.get("avg_mrr", 0.0))))
        so = _validated_dataset(report, e, "stackoverflow_qa_benchmark")
        if so:
            sources.append(("stackoverflow_qa", float(so.get("avg_mrr", 0.0))))
        if not sources:
            continue
        avg = sum(s for _, s in sources) / len(sources)
        rows.append(_entry(e, avg, components={k: round(v, 4) for k, v in sources}))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _atlas_category_avg(report: dict, engine: str, cats: list[str]) -> float | None:
    eng_rep = (report.get("atlas") or {}).get(engine)
    if not eng_rep:
        return None
    available = eng_rep.get("categories") or {}
    scores: list[float] = []
    for cat in cats:
        cat_rep = available.get(cat)
        if cat_rep:
            scores.append(float(cat_rep.get("avg_composite", 0.0)))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _build_paper_qa(report: dict, engines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for e in engines:
        score = _atlas_category_avg(report, e, ["paper_qa", "synthesis"])
        if score is None:
            continue
        rows.append(_entry(e, score))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _build_atlas_graph(report: dict, engines: list[str]) -> list[dict]:
    rows: list[dict] = []
    cats = ["graph_memory", "prior_decision", "avoid_repeat", "multi_turn"]
    for e in engines:
        score = _atlas_category_avg(report, e, cats)
        if score is None:
            continue
        rows.append(_entry(e, score))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _build_tool_contract(report: dict, engines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for e in engines:
        score = _atlas_category_avg(report, e, ["tool_contract"])
        if score is None:
            continue
        rows.append(_entry(e, score))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _build_swe_patch(report: dict, engines: list[str]) -> list[dict]:
    swe = report.get("swe_agent") or {}
    rows: list[dict] = []
    for e in engines:
        eng_data = swe.get(e)
        if not eng_data:
            continue
        # The Phase 9 aggregator stores judge_composite + criteria_pass_rate.
        score = float(eng_data.get("judge_composite", 0.0))
        rows.append(_entry(
            e, score,
            criteria_pass_rate=float(eng_data.get("criteria_pass_rate", 0.0)),
            context_utilization=float(eng_data.get("context_utilization_score", 0.0)),
        ))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _build_hallucination(report: dict, engines: list[str]) -> list[dict]:
    rows: list[dict] = []
    hal = report.get("hallucination") or {}
    for e in engines:
        # Take any model row matching this engine, prefer the lowest rate
        # (best case) to avoid penalizing for a bad model.
        rates: list[float] = []
        for key, val in hal.items():
            if not isinstance(val, dict):
                continue
            if val.get("engine") == e:
                # 'true_hallucination_rate' is the corrected number from the
                # earlier metric fix; fall back to overall if missing.
                tr = val.get("true_hallucination_rate")
                if tr is None:
                    tr = val.get("overall_rate", 0.0)
                rates.append(float(tr))
        if not rates:
            continue
        best_rate = min(rates)
        rows.append(_entry(e, 1.0 - best_rate, raw_rate=round(best_rate, 4)))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def _build_context_utilization(report: dict, engines: list[str]) -> list[dict]:
    """Surfaces SWE-Agent context_utilization separately.

    The diagnosis singled out the utilization gap (Delphi 0.12 vs Nia 0.21)
    as the most important real-world signal that single-number SWE composite
    hides.
    """
    swe = report.get("swe_agent") or {}
    rows: list[dict] = []
    for e in engines:
        eng_data = swe.get(e)
        if not eng_data:
            continue
        score = float(eng_data.get("context_utilization_score", 0.0))
        rows.append(_entry(e, score))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def build_leaderboards(report: dict) -> dict[str, list[dict]]:
    """Return ``{category: [ranked_rows]}``."""
    engines = list(report.get("engines") or [])
    out: dict[str, list[dict]] = {}

    def _add(name: str, rows: list[dict]) -> None:
        if rows:
            out[name] = rows

    _add("code_retrieval", _build_code_retrieval(report, engines))
    _add("docs_lookup", _build_docs_lookup(report, engines))
    _add("paper_qa", _build_paper_qa(report, engines))
    _add("atlas_graph", _build_atlas_graph(report, engines))
    _add("tool_contract", _build_tool_contract(report, engines))
    _add("swe_patch", _build_swe_patch(report, engines))
    _add("context_utilization", _build_context_utilization(report, engines))
    _add("hallucination_inverted", _build_hallucination(report, engines))

    return out


def print_leaderboards(boards: dict[str, list[dict]]) -> None:
    """Pretty-print category leaderboards."""
    if not boards:
        return
    print("\n=== Per-Category Leaderboards ===")
    print("  (each category scored on its own; no single 'winner')")
    for name, rows in boards.items():
        print(f"\n  [{name}]")
        for rank, row in enumerate(rows, 1):
            extra = ""
            if "raw_rate" in row:
                extra = f"  (raw hallucination rate: {row['raw_rate']:.3f})"
            elif "components" in row:
                extra = "  components: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in row["components"].items()
                )
            elif "criteria_pass_rate" in row:
                extra = (
                    f"  criteria_pass={row['criteria_pass_rate']:.2f}, "
                    f"util={row['context_utilization']:.2f}"
                )
            print(f"    {rank}. {row['engine']:<18} score={row['score']:.3f}{extra}")
