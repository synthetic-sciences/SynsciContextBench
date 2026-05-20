"""Statistical significance analysis for benchmark comparisons.

Implements research-standard methods for IR evaluation:
- Paired t-tests (Sakai 2006, recommended by Urbano et al. 2019 SIGIR)
- Wilcoxon signed-rank tests (non-parametric alternative)
- Bootstrap confidence intervals (Efron & Tibshirani 1993)
- Effect size measures (Cohen's d, Cliff's delta)
- Bonferroni correction for multiple comparisons

References:
    Urbano, Marrero & Martín (2019). "On the Measurement of Test Collection
    Reliability." SIGIR.
    Sakai (2006). "Evaluating Evaluation Metrics based on the Bootstrap." SIGIR.
    Sakai (2014). "Statistical Reform in Information Retrieval?" SIGIR Forum.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PairedTestResult:
    """Result of a paired statistical significance test."""

    test_name: str
    metric: str
    engine_a: str
    engine_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    statistic: float
    p_value: float
    significant: bool  # at the given alpha
    alpha: float = 0.05
    effect_size: float = 0.0
    effect_size_label: str = ""  # negligible / small / medium / large
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_queries: int = 0


@dataclass
class BootstrapCIResult:
    """Result of bootstrap confidence interval estimation."""

    metric: str
    engine: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float  # e.g., 0.95
    n_bootstrap: int
    n_samples: int
    std_error: float = 0.0


@dataclass
class EffectSizeResult:
    """Effect size between two systems."""

    metric: str
    engine_a: str
    engine_b: str
    cohens_d: float
    cohens_d_label: str  # negligible / small / medium / large
    cliffs_delta: float
    cliffs_delta_label: str  # negligible / small / medium / large
    n: int


@dataclass
class MultipleComparisonResult:
    """Result with multiple comparison correction applied."""

    raw_results: list[PairedTestResult]
    correction_method: str  # "bonferroni" | "holm"
    num_comparisons: int
    corrected_alpha: float
    significant_pairs: list[tuple[str, str, str]]  # (metric, engine_a, engine_b)


# ---------------------------------------------------------------------------
# Core statistical functions (no scipy dependency — pure Python)
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _std(xs: Sequence[float], ddof: int = 1) -> float:
    if len(xs) <= ddof:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def _t_cdf_approx(t: float, df: int) -> float:
    """Approximate the CDF of the t-distribution using the normal approximation.

    For df > 30, the t-distribution is well-approximated by the standard normal.
    For smaller df, uses a simple correction factor.
    """
    if df <= 0:
        return 0.5
    # Cornish-Fisher approximation for the t-distribution
    x = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
    # Standard normal CDF via error function approximation
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_ppf(p: float) -> float:
    """Approximate the inverse CDF (quantile function) of the standard normal.

    Uses the Beasley-Springer-Moro algorithm.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    # Rational approximation
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    metric: str,
    engine_a: str,
    engine_b: str,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Two-sided paired t-test (Sakai 2006; Urbano et al. 2019).

    The paired t-test is the recommended significance test for IR evaluation
    (Urbano et al. 2019, SIGIR). It compares per-query metric differences
    between two systems.

    Args:
        scores_a: Per-query scores for engine A
        scores_b: Per-query scores for engine B (same length, same queries)
        metric: Name of the metric being compared
        engine_a, engine_b: Engine names
        alpha: Significance level (default 0.05)

    Returns:
        PairedTestResult with test statistic, p-value, and effect size
    """
    n = len(scores_a)
    assert n == len(scores_b), f"Score lists must have equal length: {n} vs {len(scores_b)}"
    assert n >= 2, f"Need at least 2 paired observations, got {n}"

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = _mean(diffs)
    std_diff = _std(diffs)
    se = std_diff / math.sqrt(n) if std_diff > 0 else 1e-10

    t_stat = mean_diff / se
    df = n - 1

    # Two-sided p-value
    p_value = 2 * (1 - _t_cdf_approx(abs(t_stat), df))
    p_value = max(0.0, min(1.0, p_value))

    # Cohen's d effect size
    d = abs(mean_diff) / std_diff if std_diff > 0 else 0.0
    d_label = _cohens_d_label(d)

    # 95% CI for the mean difference
    t_crit = abs(_normal_ppf(alpha / 2))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return PairedTestResult(
        test_name="paired_t_test",
        metric=metric,
        engine_a=engine_a,
        engine_b=engine_b,
        mean_a=_mean(scores_a),
        mean_b=_mean(scores_b),
        mean_diff=mean_diff,
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=d,
        effect_size_label=d_label,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_queries=n,
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
    metric: str,
    engine_a: str,
    engine_b: str,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Wilcoxon signed-rank test (non-parametric alternative).

    Suitable when normality of score differences cannot be assumed.
    Uses normal approximation for n > 20.
    """
    n = len(scores_a)
    assert n == len(scores_b)

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    # Remove zero differences
    nonzero = [(abs(d), d) for d in diffs if d != 0]
    nr = len(nonzero)

    if nr == 0:
        return PairedTestResult(
            test_name="wilcoxon_signed_rank",
            metric=metric,
            engine_a=engine_a,
            engine_b=engine_b,
            mean_a=_mean(scores_a),
            mean_b=_mean(scores_b),
            mean_diff=0.0,
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            n_queries=n,
        )

    # Rank absolute differences
    nonzero.sort(key=lambda x: x[0])
    ranks = []
    i = 0
    while i < nr:
        j = i + 1
        while j < nr and nonzero[j][0] == nonzero[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average rank for ties
        for k in range(i, j):
            ranks.append((avg_rank, nonzero[k][1]))
        i = j

    # W+ = sum of ranks for positive differences
    w_plus = sum(rank for rank, d in ranks if d > 0)
    w_minus = sum(rank for rank, d in ranks if d < 0)
    w = min(w_plus, w_minus)

    # Normal approximation for p-value (n > 20)
    mu = nr * (nr + 1) / 4
    sigma = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24)
    if sigma > 0:
        z = (w - mu) / sigma
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        p_value = 1.0

    p_value = max(0.0, min(1.0, p_value))

    # Cliff's delta as effect size for non-parametric test
    cd = cliffs_delta(scores_a, scores_b)

    return PairedTestResult(
        test_name="wilcoxon_signed_rank",
        metric=metric,
        engine_a=engine_a,
        engine_b=engine_b,
        mean_a=_mean(scores_a),
        mean_b=_mean(scores_b),
        mean_diff=_mean(diffs),
        statistic=w,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=cd,
        effect_size_label=_cliffs_delta_label(cd),
        n_queries=n,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: list[float],
    metric: str,
    engine: str,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
    statistic_fn=None,
) -> BootstrapCIResult:
    """Compute bootstrap confidence intervals (Efron & Tibshirani 1993).

    Uses the percentile method: sample with replacement, compute the statistic
    on each resample, and take the alpha/2 and 1-alpha/2 percentiles.

    Args:
        scores: Per-query metric scores
        metric: Metric name
        engine: Engine name
        n_bootstrap: Number of bootstrap resamples (default: 10,000)
        ci_level: Confidence level (default: 0.95)
        seed: Random seed for reproducibility
        statistic_fn: Function to compute the statistic (default: mean)
    """
    if statistic_fn is None:
        statistic_fn = _mean

    n = len(scores)
    point_estimate = statistic_fn(scores)

    rng = random.Random(seed)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(statistic_fn(resample))

    bootstrap_stats.sort()
    alpha = 1 - ci_level
    lower_idx = max(0, int(alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

    ci_lower = bootstrap_stats[lower_idx]
    ci_upper = bootstrap_stats[upper_idx]

    std_error = _std(bootstrap_stats, ddof=1)

    return BootstrapCIResult(
        metric=metric,
        engine=engine,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        std_error=std_error,
    )


# ---------------------------------------------------------------------------
# Effect size measures
# ---------------------------------------------------------------------------


def _cohens_d_label(d: float) -> str:
    """Interpret Cohen's d (Cohen 1988)."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def cliffs_delta(scores_a: list[float], scores_b: list[float]) -> float:
    """Cliff's delta — non-parametric effect size.

    Returns value in [-1, 1] where:
    - 1.0 means all values in A are greater than all in B
    - 0.0 means distributions are identical
    - -1.0 means all values in B are greater than all in A
    """
    n_a, n_b = len(scores_a), len(scores_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    count = 0
    for a in scores_a:
        for b in scores_b:
            if a > b:
                count += 1
            elif a < b:
                count -= 1
    return count / (n_a * n_b)


def _cliffs_delta_label(d: float) -> str:
    """Interpret Cliff's delta (Romano et al. 2006)."""
    d = abs(d)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def compute_effect_sizes(
    scores_a: list[float],
    scores_b: list[float],
    metric: str,
    engine_a: str,
    engine_b: str,
) -> EffectSizeResult:
    """Compute both Cohen's d and Cliff's delta effect sizes."""
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = _mean(diffs)
    std_diff = _std(diffs)

    d = abs(mean_diff) / std_diff if std_diff > 0 else 0.0
    cd = cliffs_delta(scores_a, scores_b)

    return EffectSizeResult(
        metric=metric,
        engine_a=engine_a,
        engine_b=engine_b,
        cohens_d=d,
        cohens_d_label=_cohens_d_label(d),
        cliffs_delta=cd,
        cliffs_delta_label=_cliffs_delta_label(cd),
        n=n,
    )


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------


def bonferroni_correction(
    results: list[PairedTestResult],
) -> MultipleComparisonResult:
    """Apply Bonferroni correction for multiple comparisons.

    Divides the significance level by the number of comparisons.
    Conservative but widely accepted (Sakai 2014).
    """
    k = len(results)
    if k == 0:
        return MultipleComparisonResult(
            raw_results=[],
            correction_method="bonferroni",
            num_comparisons=0,
            corrected_alpha=0.05,
            significant_pairs=[],
        )

    alpha = results[0].alpha
    corrected_alpha = alpha / k

    significant = []
    for r in results:
        if r.p_value < corrected_alpha:
            significant.append((r.metric, r.engine_a, r.engine_b))

    return MultipleComparisonResult(
        raw_results=results,
        correction_method="bonferroni",
        num_comparisons=k,
        corrected_alpha=corrected_alpha,
        significant_pairs=significant,
    )


def holm_correction(
    results: list[PairedTestResult],
) -> MultipleComparisonResult:
    """Apply Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni — rejects more hypotheses
    while still controlling family-wise error rate.
    """
    k = len(results)
    if k == 0:
        return MultipleComparisonResult(
            raw_results=[],
            correction_method="holm",
            num_comparisons=0,
            corrected_alpha=0.05,
            significant_pairs=[],
        )

    alpha = results[0].alpha
    sorted_results = sorted(results, key=lambda r: r.p_value)

    significant = []
    for i, r in enumerate(sorted_results):
        adjusted_alpha = alpha / (k - i)
        if r.p_value < adjusted_alpha:
            significant.append((r.metric, r.engine_a, r.engine_b))
        else:
            break  # Holm is a step-down procedure — stop at first non-rejection

    return MultipleComparisonResult(
        raw_results=results,
        correction_method="holm",
        num_comparisons=k,
        corrected_alpha=alpha / k,  # most conservative threshold
        significant_pairs=significant,
    )


# ---------------------------------------------------------------------------
# Full pairwise analysis
# ---------------------------------------------------------------------------


def run_pairwise_significance(
    per_query_scores: dict[str, dict[str, list[float]]],
    alpha: float = 0.05,
    correction: str = "holm",
) -> tuple[list[PairedTestResult], list[BootstrapCIResult], MultipleComparisonResult]:
    """Run full pairwise significance analysis across engines and metrics.

    Args:
        per_query_scores: {metric_name: {engine_name: [per_query_scores]}}
        alpha: Significance level
        correction: "bonferroni" or "holm"

    Returns:
        (paired_tests, bootstrap_cis, corrected_result)
    """
    all_tests: list[PairedTestResult] = []
    all_cis: list[BootstrapCIResult] = []

    for metric, engine_scores in per_query_scores.items():
        engines = list(engine_scores.keys())

        # Bootstrap CIs for each engine
        for eng in engines:
            scores = engine_scores[eng]
            if len(scores) >= 2:
                ci = bootstrap_ci(scores, metric, eng)
                all_cis.append(ci)

        # Pairwise tests between all engine pairs
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                eng_a, eng_b = engines[i], engines[j]
                scores_a = engine_scores[eng_a]
                scores_b = engine_scores[eng_b]

                if len(scores_a) < 2 or len(scores_b) < 2:
                    continue
                if len(scores_a) != len(scores_b):
                    # Truncate to minimum length (queries must be paired)
                    min_n = min(len(scores_a), len(scores_b))
                    scores_a = scores_a[:min_n]
                    scores_b = scores_b[:min_n]

                # Paired t-test (primary)
                t_result = paired_t_test(scores_a, scores_b, metric, eng_a, eng_b, alpha)
                all_tests.append(t_result)

                # Wilcoxon as robustness check
                w_result = wilcoxon_signed_rank(scores_a, scores_b, metric, eng_a, eng_b, alpha)
                all_tests.append(w_result)

    # Apply multiple comparison correction
    # Only correct the primary tests (paired t-tests)
    t_tests = [r for r in all_tests if r.test_name == "paired_t_test"]
    if correction == "bonferroni":
        corrected = bonferroni_correction(t_tests)
    else:
        corrected = holm_correction(t_tests)

    return all_tests, all_cis, corrected


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_significance_summary(
    tests: list[PairedTestResult],
    cis: list[BootstrapCIResult],
    correction: MultipleComparisonResult,
) -> None:
    """Print a formatted significance analysis summary."""
    print("\n=== Statistical Significance Analysis ===")
    print(f"  Correction method: {correction.correction_method}")
    print(f"  Number of comparisons: {correction.num_comparisons}")
    print(f"  Corrected alpha: {correction.corrected_alpha:.4f}")

    # Group by metric
    metrics_seen: dict[str, list[PairedTestResult]] = {}
    for t in tests:
        metrics_seen.setdefault(t.metric, []).append(t)

    for metric, metric_tests in metrics_seen.items():
        t_tests = [t for t in metric_tests if t.test_name == "paired_t_test"]
        if not t_tests:
            continue

        print(f"\n  --- {metric} ---")
        for t in t_tests:
            sig = "*" if t.significant else ""
            corrected_sig = ""
            if (metric, t.engine_a, t.engine_b) in correction.significant_pairs:
                corrected_sig = " [sig. after correction]"
            print(
                f"  {t.engine_a} vs {t.engine_b}: "
                f"diff={t.mean_diff:+.4f}, t={t.statistic:.3f}, "
                f"p={t.p_value:.4f}{sig}, "
                f"d={t.effect_size:.3f} ({t.effect_size_label}), "
                f"95% CI [{t.ci_lower:.4f}, {t.ci_upper:.4f}]"
                f"{corrected_sig}"
            )

    # Bootstrap CIs
    if cis:
        print("\n  --- Bootstrap 95% Confidence Intervals ---")
        for ci in cis:
            print(
                f"  {ci.engine} {ci.metric}: "
                f"{ci.point_estimate:.4f} "
                f"[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}] "
                f"(SE={ci.std_error:.4f}, n={ci.n_samples})"
            )

    # Summary of significant pairs
    if correction.significant_pairs:
        print(f"\n  Significant differences (after {correction.correction_method} correction):")
        for metric, eng_a, eng_b in correction.significant_pairs:
            print(f"    {metric}: {eng_a} vs {eng_b}")
    else:
        print(f"\n  No significant differences after {correction.correction_method} correction.")
