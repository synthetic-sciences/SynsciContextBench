"""Phase 10 — Atlas-workflow benchmark.

Real Atlas usage is not function-level code retrieval. It is the orchestration
of tool contracts, graph state, artifacts, paper citations, and prior decisions
across multiple turns. Pure CodeSearchNet/CoSQA-style benchmarks miss this
entirely, which is why an engine can win retrieval and still feel worse than
its competitors in a real Atlas session.

This module evaluates each engine on eight Atlas-shaped categories
(tool_contract, graph_memory, artifact, paper_qa, multi_turn, prior_decision,
avoid_repeat, synthesis). It reuses `search_code` and `search_papers` from the
existing adapter contract, so it works against any engine without requiring
new endpoints — engines that ship a Atlas-native retrieval path simply do
better on these tasks because they surface the right evidence.

Scoring per case:

- ``evidence_recall`` — share of expected evidence keywords found across the
  retrieved chunks.
- ``anchor_hit`` — 1 if any retrieved chunk references one of the case's
  expected anchors (e.g., a graph node id, a paper key, a tool-contract path),
  otherwise 0.
- ``judge_score`` — optional LLM rubric score in [0, 1] derived from the case
  rubric and the engine's top retrieved chunks. Only computed when the
  benchmark is given LLM creds; otherwise skipped and the composite uses the
  structural signals alone.
- ``composite`` — weighted blend (0.4 anchor_hit + 0.4 evidence_recall +
  0.2 judge_score). When no judge is configured the judge weight is folded
  proportionally back into the structural terms.
- ``hallucination_signals`` — count of negative-signal phrases matched in
  the engine's surfaced content. Used to flag engines that fabricate.

Aggregate metrics are reported per-category, so the report explicitly
distinguishes "good at tool contracts" from "good at paper QA" — the
diagnosis flagged single-number leaderboards as the main reporting bug.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..adapters.base import ContextEngineAdapter, SearchResult
from ..infra.logging_config import get_logger

logger = get_logger("atlas")


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------


@dataclass
class AtlasCase:
    """One Atlas-workflow test case."""

    id: str
    category: str
    difficulty: str
    question: str
    expected_evidence: list[str]
    expected_anchors: list[str]
    judge_rubric: str = ""
    negative_signals: list[str] = field(default_factory=list)


@dataclass
class AtlasResult:
    """Per-case result for one engine."""

    case_id: str
    category: str
    difficulty: str
    engine: str
    question: str

    # Structural scores
    anchor_hit: int = 0           # 0/1
    evidence_recall: float = 0.0  # share of expected_evidence terms surfaced
    hallucination_signals: int = 0
    judge_score: float = 0.0      # 0..1, only if LLM judge enabled
    composite: float = 0.0        # weighted blend

    # Diagnostics
    num_chunks: int = 0
    chunks_with_evidence: int = 0
    surfaces_paper: bool = False
    surfaces_artifact: bool = False
    surfaces_tool_contract: bool = False
    surfaces_graph_node: bool = False

    latency_ms: float = 0.0
    error: str = ""


@dataclass
class AtlasCategoryReport:
    """Per-category aggregate for one engine."""

    category: str
    num_cases: int = 0
    avg_anchor_hit: float = 0.0
    avg_evidence_recall: float = 0.0
    avg_composite: float = 0.0
    avg_hallucination_signals: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class AtlasEngineReport:
    """Full Atlas-benchmark report for one engine."""

    engine: str
    num_cases: int = 0
    avg_composite: float = 0.0
    avg_anchor_hit: float = 0.0
    avg_evidence_recall: float = 0.0
    avg_hallucination_signals: float = 0.0
    avg_latency_ms: float = 0.0
    categories: dict[str, AtlasCategoryReport] = field(default_factory=dict)
    per_case: list[AtlasResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_atlas_cases(path: str | Path) -> list[AtlasCase]:
    """Load cases from JSON. Robust to extra fields."""
    with open(path) as f:
        data = json.load(f)
    out: list[AtlasCase] = []
    for raw in data.get("test_cases", []):
        out.append(
            AtlasCase(
                id=raw["id"],
                category=raw.get("category", "unknown"),
                difficulty=raw.get("difficulty", "medium"),
                question=raw["question"],
                expected_evidence=list(raw.get("expected_evidence", [])),
                expected_anchors=list(raw.get("expected_anchors", [])),
                judge_rubric=raw.get("judge_rubric", ""),
                negative_signals=list(raw.get("negative_signals", [])),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-chunk classification
# ---------------------------------------------------------------------------


def _classify_chunk(chunk: SearchResult) -> dict[str, bool]:
    """Heuristic surface-type detection for a single retrieved chunk."""
    text = (chunk.content or "")
    text_lower = text.lower()
    path_lower = (chunk.file_path or "").lower()

    is_paper = any(
        marker in path_lower
        for marker in (".pdf", "paper", "arxiv", "section ", "abstract")
    ) or any(m in text_lower for m in ("abstract\n", "et al.", "doi:", "arxiv:"))

    is_artifact = any(
        marker in path_lower
        for marker in ("artifact", "results", "plots", "tables", ".csv", ".json")
    ) or any(m in text_lower for m in ("mrr=", "recall=", "table ", "figure "))

    is_tool_contract = any(
        marker in path_lower
        for marker in ("mcp", "tool_contracts", "tools.py", "schema")
    ) or any(m in text_lower for m in ("tool_call", "input_schema", "parameters", "mcp tool"))

    is_graph_node = any(
        marker in path_lower
        for marker in ("graph/", "atlas", "node_", "hypothesis", "branch")
    ) or any(m in text_lower for m in ("node id", "hypothesis", "committed", "campaign"))

    return {
        "paper": is_paper,
        "artifact": is_artifact,
        "tool_contract": is_tool_contract,
        "graph_node": is_graph_node,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _evidence_recall(chunks: list[SearchResult], expected: list[str]) -> float:
    """Fraction of expected evidence terms that appear somewhere in the chunks."""
    if not expected:
        return 0.0
    haystack = "\n".join((c.content or "") + " " + (c.file_path or "") for c in chunks).lower()
    hits = sum(1 for term in expected if term.lower() in haystack)
    return hits / len(expected)


def _anchor_hit(chunks: list[SearchResult], anchors: list[str]) -> int:
    """1 if any expected anchor appears in chunk paths or content."""
    if not anchors:
        return 0
    for c in chunks:
        text = ((c.file_path or "") + "\n" + (c.content or "")).lower()
        for a in anchors:
            if a.lower() in text:
                return 1
    return 0


def _hallucination_signals(chunks: list[SearchResult], negatives: list[str]) -> int:
    """Count negative-signal phrases matched in the surfaced content."""
    if not negatives:
        return 0
    haystack = "\n".join((c.content or "") for c in chunks).lower()
    return sum(1 for n in negatives if n.lower() in haystack)


_ATLAS_JUDGE_SYSTEM = (
    "You are evaluating whether retrieved context answers a Atlas-workflow "
    "question. Score a single number in [0,1] for how well the context grounds "
    "an answer that satisfies the rubric. 0 means useless. 1 means a downstream "
    "agent could answer the question correctly using only this context. "
    "Reward citing specific anchors (node ids, artifact paths, paper sections). "
    "Penalize fabricated identifiers."
)


async def _judge_score(
    case: AtlasCase,
    chunks: list[SearchResult],
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> float:
    """Call the LLM judge for one case. Returns a float in [0, 1]."""
    from ..judges.llm_judge import _call_llm_judge_raw, _safe_parse_json

    context_blob = "\n\n---\n\n".join(
        f"[{c.file_path or c.id}]\n{(c.content or '')[:600]}" for c in chunks[:8]
    )
    user_prompt = (
        f"## Question\n{case.question}\n\n"
        f"## Rubric\n{case.judge_rubric}\n\n"
        f"## Retrieved context (top {min(len(chunks), 8)} chunks)\n{context_blob}\n\n"
        f'Respond with ONLY JSON: {{"score": <float between 0 and 1>}}'
    )
    try:
        text = await _call_llm_judge_raw(
            system_prompt=_ATLAS_JUDGE_SYSTEM,
            user_prompt=user_prompt,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
        data = _safe_parse_json(text, defaults={"score": 0.0})
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as e:  # pragma: no cover - judge failures are tolerated
        logger.warning("Atlas judge failed for %s: %s", case.id, e)
        return 0.0


def _compose(anchor_hit: int, evidence_recall: float, judge_score: float, has_judge: bool) -> float:
    """Blend structural and judge signals into a single composite."""
    if has_judge:
        return 0.4 * anchor_hit + 0.4 * evidence_recall + 0.2 * judge_score
    # Re-allocate the judge weight to the structural terms so composites
    # remain comparable in magnitude across configurations.
    return 0.5 * anchor_hit + 0.5 * evidence_recall


# ---------------------------------------------------------------------------
# Per-engine runner
# ---------------------------------------------------------------------------


async def _retrieve_for_case(
    engine: ContextEngineAdapter,
    case: AtlasCase,
    top_k: int,
) -> tuple[list[SearchResult], float, str]:
    """Run retrieval for one case. Paper-flavored cases also hit search_papers."""
    chunks: list[SearchResult] = []
    total_latency = 0.0
    error = ""
    try:
        res, lat = await engine.search_code(query=case.question, top_k=top_k)
        chunks.extend(res)
        total_latency += lat
    except Exception as e:
        error = f"search_code: {type(e).__name__}: {e}"
        logger.warning("Atlas search_code failed [%s/%s]: %s", engine.name, case.id, e)

    if case.category in ("paper_qa", "synthesis"):
        try:
            pres, plat = await engine.search_papers(query=case.question, top_k=top_k)
            chunks.extend(pres)
            total_latency += plat
        except Exception as e:
            logger.info(
                "Atlas search_papers fell back for %s (%s): %s",
                engine.name, case.id, e,
            )

    return chunks, total_latency, error


async def run_atlas_benchmark(
    engine: ContextEngineAdapter,
    dataset_path: str | Path,
    top_k: int = 10,
    max_cases: int | None = None,
    seed: int = 0,
    llm_provider: str = "",
    llm_model: str = "",
    llm_api_key: str = "",
) -> AtlasEngineReport:
    """Run the Atlas benchmark for a single engine.

    The LLM judge is optional. Without it, the composite uses the structural
    anchor_hit + evidence_recall signals only, which still discriminates
    engines that surface the right evidence from those that do not.
    """
    from ..infra.sampling import stratified_sample

    cases = load_atlas_cases(dataset_path)
    cases = stratified_sample(cases, max_cases, key=lambda c: c.category, seed=seed)

    has_judge = bool(llm_api_key and llm_provider and llm_model)
    per_case: list[AtlasResult] = []

    for case in tqdm(cases, desc=f"  {engine.name} atlas", unit="q"):
        t0 = time.perf_counter()
        chunks, retrieval_latency, error = await _retrieve_for_case(engine, case, top_k)
        anchor = _anchor_hit(chunks, case.expected_anchors)
        recall = _evidence_recall(chunks, case.expected_evidence)
        halls = _hallucination_signals(chunks, case.negative_signals)

        judge = 0.0
        if has_judge and chunks:
            judge = await _judge_score(case, chunks, llm_provider, llm_model, llm_api_key)
            await asyncio.sleep(0.2)  # avoid bursty judge calls

        classification_counts = {"paper": 0, "artifact": 0, "tool_contract": 0, "graph_node": 0}
        chunks_with_evidence = 0
        for c in chunks:
            cls = _classify_chunk(c)
            for k, v in cls.items():
                if v:
                    classification_counts[k] += 1
            if any(term.lower() in ((c.content or "") + " " + (c.file_path or "")).lower()
                   for term in case.expected_evidence):
                chunks_with_evidence += 1

        result = AtlasResult(
            case_id=case.id,
            category=case.category,
            difficulty=case.difficulty,
            engine=engine.name,
            question=case.question,
            anchor_hit=anchor,
            evidence_recall=recall,
            hallucination_signals=halls,
            judge_score=judge,
            composite=_compose(anchor, recall, judge, has_judge),
            num_chunks=len(chunks),
            chunks_with_evidence=chunks_with_evidence,
            surfaces_paper=classification_counts["paper"] > 0,
            surfaces_artifact=classification_counts["artifact"] > 0,
            surfaces_tool_contract=classification_counts["tool_contract"] > 0,
            surfaces_graph_node=classification_counts["graph_node"] > 0,
            latency_ms=(time.perf_counter() - t0) * 1000 if retrieval_latency == 0 else retrieval_latency,
            error=error,
        )
        per_case.append(result)

    return _aggregate(engine.name, per_case)


def _aggregate(engine_name: str, per_case: list[AtlasResult]) -> AtlasEngineReport:
    if not per_case:
        return AtlasEngineReport(engine=engine_name)

    by_cat: dict[str, list[AtlasResult]] = {}
    for r in per_case:
        by_cat.setdefault(r.category, []).append(r)

    cats: dict[str, AtlasCategoryReport] = {}
    for name, rs in by_cat.items():
        n = len(rs)
        cats[name] = AtlasCategoryReport(
            category=name,
            num_cases=n,
            avg_anchor_hit=sum(r.anchor_hit for r in rs) / n,
            avg_evidence_recall=sum(r.evidence_recall for r in rs) / n,
            avg_composite=sum(r.composite for r in rs) / n,
            avg_hallucination_signals=sum(r.hallucination_signals for r in rs) / n,
            avg_latency_ms=sum(r.latency_ms for r in rs) / n,
        )

    n_total = len(per_case)
    return AtlasEngineReport(
        engine=engine_name,
        num_cases=n_total,
        avg_composite=sum(r.composite for r in per_case) / n_total,
        avg_anchor_hit=sum(r.anchor_hit for r in per_case) / n_total,
        avg_evidence_recall=sum(r.evidence_recall for r in per_case) / n_total,
        avg_hallucination_signals=sum(r.hallucination_signals for r in per_case) / n_total,
        avg_latency_ms=sum(r.latency_ms for r in per_case) / n_total,
        categories=cats,
        per_case=per_case,
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def print_atlas_summary(report: AtlasEngineReport) -> None:
    """Console summary for one engine."""
    print(f"  Engine:                {report.engine}")
    print(f"  Cases:                 {report.num_cases}")
    print(f"  Composite (0-1):       {report.avg_composite:.3f}")
    print(f"  Anchor hit rate:       {report.avg_anchor_hit:.3f}")
    print(f"  Evidence recall:       {report.avg_evidence_recall:.3f}")
    print(f"  Hallucination signals: {report.avg_hallucination_signals:.2f}/case")
    print(f"  Avg latency:           {report.avg_latency_ms:.0f}ms")
    print("  Per-category composite:")
    for name in sorted(report.categories):
        cr = report.categories[name]
        print(f"    - {name:<16} n={cr.num_cases:<3} composite={cr.avg_composite:.3f} "
              f"anchor={cr.avg_anchor_hit:.2f} recall={cr.avg_evidence_recall:.2f}")
