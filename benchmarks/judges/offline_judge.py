"""Deterministic, dependency-free judge for offline runs and tests.

Every judged phase calls a paid LLM. That makes the benchmark non-reproducible
without keys and budget, and makes the judge-dependent scorers untestable. This
module provides a deterministic stand-in that grades relevance by token overlap
between the retrieved result and the ground truth. It is NOT a substitute for
the LLM judge in published numbers — it exists so the harness can be exercised
end-to-end in CI and so scorer logic can be unit-tested with known inputs.

Grade scale matches the LLM judge (0–3):
  3 fully answers, 2 partial, 1 tangential, 0 irrelevant.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def offline_relevance_grade(query: str, ground_truth: str, result: str) -> int:
    """Deterministic 0–3 relevance grade by token overlap.

    Compares the result against the union of the query and ground-truth tokens
    (Jaccard-style), bucketed into the 0–3 grade scale.
    """
    target = _tokens(query) | _tokens(ground_truth)
    got = _tokens(result)
    if not target or not got:
        return 0
    overlap = len(target & got) / len(target)
    if overlap >= 0.6:
        return 3
    if overlap >= 0.35:
        return 2
    if overlap >= 0.15:
        return 1
    return 0


def offline_judge_match(query: str, ground_truth: str, result: str) -> tuple[bool, int]:
    """Mirror of ``validated_eval._llm_judge_match`` for offline use.

    Returns ``(is_relevant, grade)`` where relevance is ``grade >= 2``.
    """
    grade = offline_relevance_grade(query, ground_truth, result)
    return grade >= 2, grade
