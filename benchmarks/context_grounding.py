"""Context utilization, citation grounding, and 'did context change the answer?'.

The diagnosis singled out four signals the benchmark was previously missing:

1. **Citation grounding** — does the final answer cite the retrieved context?
   Without this, an engine that returns perfect chunks looks identical to one
   that returns junk, as long as the LLM happens to know the right answer
   already.
2. **Context utilization** — what fraction of facts in the retrieved chunks
   actually appear in the answer. (Phase 9 ``swe_agent`` already computes
   one variant of this; we expose it as a shared helper for other phases.)
3. **Context-changed-the-answer** — diff the answer with and without
   context; non-trivial deltas indicate the context did real work.
4. **Context-prevented-hallucination** — did context cause a known
   library/API symbol to be used correctly that was previously fabricated?
   Computed as the *reduction* in hallucination signals between the
   no-context and with-context outputs.

These helpers are deliberately small and stateless so any benchmark module
can call them on its own answer/context pairs.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class GroundingMetrics:
    """All four context-quality signals for a single answer/context pair."""

    citation_count: int = 0          # number of explicit citation patterns
    cited_chunks: int = 0            # number of retrieved chunks referenced
    citation_share: float = 0.0      # cited_chunks / total_chunks
    utilization: float = 0.0         # facts_in_context_used / facts_available
    facts_available: int = 0
    facts_used: int = 0
    answer_change: float = 0.0       # 0=identical, 1=completely different
    hallucination_reduction: int = 0 # negative-signal hits removed by context


# ---------------------------------------------------------------------------
# Citation detection
# ---------------------------------------------------------------------------

_CITATION_PATTERNS = (
    # numeric brackets common in our judge prompts:  [1], [2], [3..]
    re.compile(r"\[(\d+)\]"),
    # explicit "chunk N" / "source N" / "context N"
    re.compile(r"\b(?:chunk|source|context|reference)\s+(\d+)\b", re.IGNORECASE),
    # explicit "according to <file>" / "see <file>"
    re.compile(r"\b(?:according to|see|cited from|per)\s+([^\.,;\n]+\.(?:py|md|rst|txt|json|yaml|yml|pdf))",
               re.IGNORECASE),
)


def detect_citations(answer: str) -> tuple[int, set[str]]:
    """Return (citation_count, set_of_distinct_targets)."""
    hits: list[str] = []
    targets: set[str] = set()
    for pattern in _CITATION_PATTERNS:
        for m in pattern.findall(answer or ""):
            hits.append(m)
            if isinstance(m, str):
                targets.add(m.strip().lower())
            else:
                targets.add(str(m))
    return len(hits), targets


def cited_chunk_share(answer: str, chunks: list[str]) -> tuple[int, float]:
    """How many chunks does the answer reference by content overlap?

    Heuristic: a chunk counts as cited if any of its unique identifying tokens
    (rare-ish identifiers, paths, paper section labels) appears in the answer.
    """
    if not chunks:
        return 0, 0.0
    cited = 0
    answer_lower = (answer or "").lower()
    for chunk in chunks:
        signature_tokens = _signature_tokens(chunk)
        if not signature_tokens:
            continue
        if any(tok in answer_lower for tok in signature_tokens):
            cited += 1
    return cited, cited / len(chunks)


def _signature_tokens(chunk: str) -> list[str]:
    """Pull out 1-4 'rare' identifying tokens from a chunk."""
    if not chunk:
        return []
    candidates: list[str] = []
    # snake_case / CamelCase identifiers
    for m in re.findall(r"\b([A-Z][A-Za-z0-9_]{4,}|[a-z]+_[a-z][A-Za-z0-9_]+)\b", chunk):
        candidates.append(m.lower())
    # file paths
    for m in re.findall(r"\b[\w/\\\-.]+\.(?:py|md|rst|json|yaml|yml|pdf)\b", chunk):
        candidates.append(m.lower())
    # quoted strings
    for m in re.findall(r'"([^"\n]{6,40})"', chunk):
        candidates.append(m.lower())
    # de-dupe, keep first 4
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= 4:
            break
    return out


# ---------------------------------------------------------------------------
# Utilization (re-export the swe_agent variant if available)
# ---------------------------------------------------------------------------


def fact_utilization(answer: str, chunks: list[str]) -> tuple[int, int, float]:
    """Compute (facts_available, facts_used, utilization).

    Defers to ``swe_agent._compute_context_utilization`` when available so we
    keep one canonical implementation; falls back to a minimal in-module
    extractor otherwise.
    """
    try:
        from .swe_agent import _compute_context_utilization  # type: ignore
        return _compute_context_utilization(answer or "", chunks)
    except Exception:
        return _fallback_utilization(answer or "", chunks)


def _fallback_utilization(answer: str, chunks: list[str]) -> tuple[int, int, float]:
    facts: set[str] = set()
    for chunk in chunks:
        for m in re.finditer(r"def\s+(\w+)\s*\(", chunk):
            facts.add(m.group(1))
        for m in re.finditer(r"class\s+(\w+)", chunk):
            facts.add(m.group(1))
        for m in re.finditer(r"\.(\w+)\s*\(", chunk):
            facts.add(m.group(1))
    facts = {f for f in facts if len(f) > 2}
    if not facts:
        return 0, 0, 0.0
    used = sum(1 for f in facts if f in answer)
    return len(facts), used, used / len(facts)


# ---------------------------------------------------------------------------
# Answer-change-with-context
# ---------------------------------------------------------------------------


def answer_change(no_context_answer: str, with_context_answer: str) -> float:
    """Lower is more similar. Uses SequenceMatcher ratio (0..1)."""
    a = (no_context_answer or "").strip()
    b = (with_context_answer or "").strip()
    if not a and not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a[:4000], b[:4000]).ratio()
    return round(1.0 - ratio, 4)


# ---------------------------------------------------------------------------
# Hallucination reduction
# ---------------------------------------------------------------------------


def hallucination_reduction(
    no_context_answer: str,
    with_context_answer: str,
    negative_signals: Iterable[str],
) -> int:
    """Negative-signal hits in no_context minus hits in with_context.

    Positive value means context prevented some hallucinations. Zero or
    negative means context introduced/maintained them.
    """
    sig_list = [s for s in negative_signals if s]
    before = sum(1 for s in sig_list if s.lower() in (no_context_answer or "").lower())
    after = sum(1 for s in sig_list if s.lower() in (with_context_answer or "").lower())
    return before - after


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def grounding_metrics(
    *,
    answer: str,
    chunks: list[str],
    no_context_answer: str = "",
    negative_signals: Iterable[str] = (),
) -> GroundingMetrics:
    """Compute all four signals for one answer/context pair."""
    citation_count, _ = detect_citations(answer)
    cited_chunks, citation_share = cited_chunk_share(answer, chunks)
    facts_available, facts_used, util = fact_utilization(answer, chunks)
    change = answer_change(no_context_answer, answer) if no_context_answer else 0.0
    reduction = hallucination_reduction(no_context_answer, answer, negative_signals) if no_context_answer else 0

    return GroundingMetrics(
        citation_count=citation_count,
        cited_chunks=cited_chunks,
        citation_share=round(citation_share, 4),
        utilization=round(util, 4),
        facts_available=facts_available,
        facts_used=facts_used,
        answer_change=change,
        hallucination_reduction=reduction,
    )
