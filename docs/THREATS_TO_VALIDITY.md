# Threats to Validity

A benchmark is only as trustworthy as its account of how it could be wrong. This
document states the threats to the validity of SynSci Context Bench and the
mitigations in place, so results can be read with the right amount of
skepticism. It is written to the standard expected of an empirical
evaluation paper.

## 1. Internal validity — does the harness measure what it claims?

**Scorer correctness.** The scoring code is now covered by a deterministic test
suite (`tests/`) that runs offline (no API keys, no network) against a
`MockAdapter` and hand-computed expected values. This guards the IR metrics
(MRR, nDCG, MAP, P@K, R@K, R-Precision, Success@K), the diff-aware composite, and
the session-replay aggregation.

**Known prior defect (fixed).** Before this revision, the diff-aware phase called
`engine.search` / `engine.asearch` — methods no adapter implements (the contract
is `search_code(...) -> (list[SearchResult], latency)`). Every engine therefore
received an empty result list and floored at correctness `1/3`. The phase also
never re-indexed between commit A and commit B, so it could not measure freshness
even in principle. Both are fixed: the phase now uses the adapter contract and
drives a real `A -> B` re-index via `index_repository_at_commit`, scored only for
engines that support commit pinning (others are reported as *unsupported*, not
floored). Regression tests assert a fresh engine scores 1.0 and a frozen index
floors at 1/3.

**Near-zero validated MRR is a coverage artifact, not a metric bug.**
`tests/test_validated_eval_offline.py` shows that when the relevant document is
actually retrieved, MRR is ~1.0. The near-zero validated MRRs in the published
run therefore indicate that the dataset corpus was not fully indexed into the
engines under test (retrieval could not surface ground truth), **not** an error
in the metric. The fix is operational: index each validated dataset's corpus
into every engine before scoring, and assert non-empty intersection between
returned ids and the dataset corpus.

## 2. Construct validity — does retrieval quality measure context value?

The headline retrieval metrics (MRR/nDCG) measure ranking quality, but the thing
that matters is whether the context **improves downstream generation**. The
SWE-Agent phase (Phase 9) is the construct-validity check, and it is informative
precisely because it does **not** always agree with retrieval rank: an engine can
win MRR yet produce context the model utilizes less. Conclusions about "better
context" must cite Phase 9, not retrieval rank alone. We recommend treating
retrieval phases as *necessary-but-not-sufficient* signals.

## 3. External validity — do results generalize?

- **Library/domain coverage.** The curated and diff-aware corpora are Python
  libraries (FastAPI, httpx, pydantic, SQLAlchemy, polars, ...). Results may not
  transfer to other languages or to application (vs. library) code. Extending the
  corpus to Go/Rust/TS/Java is open work.
- **Query distribution.** Curated queries are author-written; validated datasets
  (CodeSearchNet, CoSQA, AdvTest) bring an external query distribution. Report
  both separately (see §5) and weight external datasets when claiming generality.

## 4. Data contamination / leakage (critical for an LLM-era benchmark)

CodeSearchNet (2019) and CoSQA (2021) predate the training cutoffs of every
judge and most engines. Two leakage paths exist: (a) the **judge** may have
memorized canonical answers; (b) an **engine's** underlying model may have seen
the corpus. Mitigations and recommendations:

- Prefer **post-cutoff** evidence. The diff-aware (Phase 10) and session-replay
  (Phase 11) phases use commits and incidents drawn from recent, dated repository
  history; pinning cases to commits after a stated cutoff makes them
  contamination-resistant and is the strongest part of the suite for a 2027
  submission.
- Publish a **contamination audit**: for each validated dataset, report the
  fraction of items whose ground-truth answer the judge reproduces verbatim with
  no context (a leakage proxy).
- Report curated vs. validated results separately and never average across them.

## 5. Vendor bias

This benchmark is authored by the maintainers of one of the engines under test
(Delphi). That is a real conflict of interest. Mitigations:

- **Open source + offline-reproducible.** The harness, datasets, scorers, and
  (with this revision) a keyless offline mode are public; anyone can re-run and
  audit. The test suite pins scorer behavior.
- **Format-agnostic LLM judge** replaces file-path matching, which would favor
  whichever engine returns repo-native chunks.
- **Report losses.** The suite reports phases where competitors win and does not
  suppress them.
- **Separate self-owned from external data.** Curated datasets (this repo) and
  validated public datasets are reported separately; headline generality claims
  should lead with the public datasets and the contamination-resistant phases.
- **Recommended:** a third-party replication and pre-registration of the metric
  set and hypotheses before the next full run.

## 6. Judge reliability

LLM-as-judge introduces its own error. Controls in place / recommended:

- **Position debiasing** — each item scored twice with swapped chunk order;
  report position-consistency.
- **Multi-judge ensemble** (Gemini / Claude / GPT) for the headline judged
  phases; report inter-judge agreement (Cohen's / Fleiss' kappa).
- **Human-validation subset** — sample N judged items, collect human labels, and
  report judge–human agreement. This is the single most valuable addition for
  publication and is currently the largest open gap.

## 7. Statistical-conclusion validity

- Several decision-relevant phases have small N (SWE-Agent n=25, diff-aware n=15,
  session-replay n=10). Differences of a few points on these are very unlikely to
  be significant. Always report bootstrap CIs and effect sizes alongside point
  estimates on these phases, and avoid bolding sub-CI gaps.
- Run multiple seeds (`--num-seeds`) and aggregate; report mean ± CI over seeds,
  not a single draw.

## Summary of the most important open items for publication

1. Human-validation subset for judge reliability (judge–human kappa).
2. Contamination audit for the validated datasets.
3. Fix corpus coverage so validated MRRs are meaningful, then re-run.
4. Larger N + CIs on Phases 9–11; lead generality claims with public/post-cutoff
   data.
5. Independent third-party replication.
