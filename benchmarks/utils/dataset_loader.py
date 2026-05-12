"""Download and convert validated benchmark datasets from HuggingFace.

Supported datasets:
1. CodeSearchNet — 2M (comment, code) pairs across 6 languages (GitHub)
   - Used in: MTEB, CodeBERT, GraphCodeBERT evaluations
   - HF: code-search-net/code_search_net
   - Schema: func_documentation_string (query), func_code_string (code)

2. CoSQA — 20,604 human-annotated (web query, code) pairs (Python only)
   - Used in: CodeBERT, UniXcoder, CodeRetriever evaluations
   - HF: CoIR-Retrieval/cosqa (BEIR-style: separate queries, corpus, qrels configs)
   - Schema: queries(_id, text), corpus(_id, text), qrels(query-id, corpus-id, score)

3. CodeSearchNet Challenge — curated queries with expert relevance judgments
   - The "gold standard" for code retrieval evaluation
   - HF: irds/codesearchnet_challenge

Each downloader produces our internal benchmark JSON format so it plugs
directly into the existing runner.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Downloads land under benchmarks/datasets/validated/, side-by-side with the
# in-repo hand-curated cases in benchmarks/datasets/curated/.
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "validated"


def _ensure_datasets_lib():
    """Import HuggingFace datasets, with a helpful error if missing."""
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` library required. Install with: uv pip install datasets"
        )


# ---------------------------------------------------------------------------
# CodeSearchNet
# ---------------------------------------------------------------------------

def download_codesearchnet(
    languages: list[str] | None = None,
    max_per_language: int = 500,
    seed: int = 42,
) -> Path:
    """Download CodeSearchNet and convert to retrieval benchmark format.

    Creates query-code pairs where the query is the function's docstring
    and the code is the function body. This is the standard evaluation
    protocol from the CodeSearchNet paper.

    Args:
        languages: Which languages to include (default: all 6)
        max_per_language: Max samples per language (to keep manageable)
        seed: Random seed for sampling

    Returns:
        Path to the generated JSON file
    """
    datasets = _ensure_datasets_lib()

    all_languages = languages or ["python", "javascript", "java", "go", "ruby", "php"]
    random.seed(seed)

    queries = []
    corpus = []
    qrels = []
    query_id = 0

    for lang in all_languages:
        print(f"  Loading CodeSearchNet/{lang}...")
        try:
            ds = datasets.load_dataset(
                "code-search-net/code_search_net",
                lang,
                split="test",
            )
        except Exception as e:
            print(f"  [!] Failed to load {lang}: {e}")
            continue

        # Sample if dataset is larger than max
        indices = list(range(len(ds)))
        if len(indices) > max_per_language:
            random.shuffle(indices)
            indices = indices[:max_per_language]

        for idx in indices:
            row = ds[idx]
            docstring = (row.get("func_documentation_string") or "").strip()
            code = (row.get("func_code_string") or "").strip()
            func_name = row.get("func_name", "")
            repo = row.get("repository_name", "")
            path = row.get("func_path_in_repository", "")

            if not docstring or not code or len(docstring) < 10:
                continue

            doc_id = f"csn_{lang}_{idx}"
            qid = f"q_{query_id}"

            queries.append({
                "id": qid,
                "query": docstring,
                "language": lang,
                "source": "codesearchnet",
            })

            corpus.append({
                "id": doc_id,
                "content": code,
                "language": lang,
                "func_name": func_name,
                "repo": repo,
                "file_path": path,
            })

            # Each query has exactly 1 relevant doc (its paired code)
            qrels.append({
                "query_id": qid,
                "doc_id": doc_id,
                "relevance": 1,
            })

            query_id += 1

    output = {
        "_description": "CodeSearchNet benchmark — docstring queries matched to function code",
        "_source": "code-search-net/code_search_net (HuggingFace)",
        "_citation": "Husain et al., 2019. CodeSearchNet Challenge: Evaluating the State of Semantic Code Search",
        "languages": all_languages,
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels,
    }

    out_path = DATASETS_DIR / "codesearchnet_benchmark.json"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(queries)} queries across {len(all_languages)} languages -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CoSQA (Code Search and Question Answering)
# ---------------------------------------------------------------------------

def download_cosqa(
    max_samples: int = 500,
    seed: int = 42,
) -> Path:
    """Download CoSQA and convert to retrieval benchmark format.

    CoSQA is a BEIR-style dataset on HuggingFace with three configs:
    - 'queries' (split='queries'): _id, text (natural language query)
    - 'corpus' (split='corpus'): _id, text (Python code)
    - default (split='test'): query-id, corpus-id, score (qrels)

    We load all three and join them into our benchmark format.

    Args:
        max_samples: Max number of query-code pairs to include
        seed: Random seed for sampling

    Returns:
        Path to the generated JSON file
    """
    datasets = _ensure_datasets_lib()
    random.seed(seed)

    print("  Loading CoSQA queries...")
    queries_ds = datasets.load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
    print(f"  Loaded {len(queries_ds)} queries")

    print("  Loading CoSQA corpus...")
    corpus_ds = datasets.load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
    print(f"  Loaded {len(corpus_ds)} corpus docs")

    print("  Loading CoSQA qrels...")
    qrels_ds = datasets.load_dataset("CoIR-Retrieval/cosqa", split="test")
    print(f"  Loaded {len(qrels_ds)} qrels")

    # Build lookup maps
    query_map = {}
    for row in queries_ds:
        qid = row["_id"]
        query_map[qid] = row["text"]

    corpus_map = {}
    for row in corpus_ds:
        did = row["_id"]
        corpus_map[did] = row["text"]

    # Sample qrels
    qrel_indices = list(range(len(qrels_ds)))
    if len(qrel_indices) > max_samples:
        random.shuffle(qrel_indices)
        qrel_indices = qrel_indices[:max_samples]

    queries = []
    corpus = []
    qrels = []
    seen_queries = set()
    seen_docs = set()

    for idx in qrel_indices:
        row = qrels_ds[idx]
        qid = row["query-id"]
        did = row["corpus-id"]
        score = row["score"]

        if qid not in query_map or did not in corpus_map:
            continue
        if score <= 0:
            continue

        if qid not in seen_queries:
            queries.append({
                "id": qid,
                "query": query_map[qid],
                "language": "python",
                "source": "cosqa",
            })
            seen_queries.add(qid)

        if did not in seen_docs:
            corpus.append({
                "id": did,
                "content": corpus_map[did],
                "language": "python",
            })
            seen_docs.add(did)

        qrels.append({
            "query_id": qid,
            "doc_id": did,
            "relevance": score,
        })

    output = {
        "_description": "CoSQA benchmark — real web queries annotated by 3+ humans for Python code search",
        "_source": "CoIR-Retrieval/cosqa (HuggingFace)",
        "_citation": "Huang et al., 2021. CoSQA: 20,000+ Web Queries for Code Search and Question Answering",
        "languages": ["python"],
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels,
    }

    out_path = DATASETS_DIR / "cosqa_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(queries)} queries, {len(corpus)} docs, {len(qrels)} qrels -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CodeSearchNet Challenge (expert-annotated relevance judgments)
# ---------------------------------------------------------------------------

def download_codesearchnet_challenge() -> Path:
    """Download CodeSearchNet Challenge with expert relevance labels.

    This is the "gold standard" subset — 99 queries with expert-annotated
    relevance judgments (0-3 graded relevance per query-code pair).
    Much smaller but higher quality than the full CSN dataset.

    Returns:
        Path to the generated JSON file
    """
    datasets = _ensure_datasets_lib()

    print("  Loading CodeSearchNet Challenge...")
    try:
        # irds dataset has specific split names matching config names
        queries_ds = datasets.load_dataset(
            "irds/codesearchnet_challenge", "queries", split="queries",
        )
        print(f"  Loaded {len(queries_ds)} queries")
    except Exception as e:
        print(f"  [!] Challenge dataset not available ({e}), skipping")
        return DATASETS_DIR / "codesearchnet_challenge_benchmark.json"

    try:
        qrels_ds = datasets.load_dataset(
            "irds/codesearchnet_challenge", "qrels", split="qrels",
        )
        print(f"  Loaded {len(qrels_ds)} qrels")
    except Exception:
        # Try alternate split name
        qrels_ds = datasets.load_dataset(
            "irds/codesearchnet_challenge", "qrels", split="train",
        )

    try:
        docs_ds = datasets.load_dataset(
            "irds/codesearchnet_challenge", "docs", split="docs",
        )
        print(f"  Loaded {len(docs_ds)} docs")
    except Exception:
        docs_ds = datasets.load_dataset(
            "irds/codesearchnet_challenge", "docs", split="train",
        )

    # Build lookup maps
    doc_map = {}
    for row in docs_ds:
        doc_id = str(row.get("doc_id", row.get("_id", "")))
        doc_map[doc_id] = {
            "id": doc_id,
            "content": row.get("code", row.get("text", "")),
            "language": row.get("language", "python"),
            "func_name": row.get("func_name", ""),
            "repo": row.get("repo", ""),
            "file_path": row.get("path", ""),
        }

    queries = []
    for row in queries_ds:
        qid = str(row.get("query_id", row.get("_id", "")))
        query_text = row.get("text", row.get("query", ""))
        queries.append({
            "id": qid,
            "query": query_text,
            "language": "python",
            "source": "codesearchnet_challenge",
        })

    qrels = []
    relevant_doc_ids = set()
    for row in qrels_ds:
        qid = str(row.get("query_id", ""))
        doc_id = str(row.get("doc_id", ""))
        rel = row.get("relevance", row.get("score", 0))
        if rel > 0:
            qrels.append({
                "query_id": qid,
                "doc_id": doc_id,
                "relevance": rel,
            })
            relevant_doc_ids.add(doc_id)

    # Only include docs that appear in qrels
    corpus = [doc_map[did] for did in relevant_doc_ids if did in doc_map]

    output = {
        "_description": "CodeSearchNet Challenge — 99 queries with expert graded relevance (0-3)",
        "_source": "irds/codesearchnet_challenge (HuggingFace)",
        "_citation": "Husain et al., 2019. CodeSearchNet Challenge",
        "_note": "Gold standard — expert relevance judgments, not docstring matching",
        "languages": ["python"],
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "num_qrels": len(qrels),
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels,
    }

    out_path = DATASETS_DIR / "codesearchnet_challenge_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(queries)} queries, {len(corpus)} docs, {len(qrels)} judgments -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# AdvTest (CodeXGLUE adversarial code search)
# ---------------------------------------------------------------------------

def download_advtest(
    max_samples: int = 500,
    seed: int = 42,
) -> Path:
    """Download AdvTest and convert to retrieval benchmark format.

    AdvTest is the adversarial code search dataset from CodeXGLUE. It is
    derived from CodeSearchNet but with function names normalized (replaced
    with generic identifiers) to prevent trivial keyword matching. This
    tests whether a model truly understands code semantics rather than
    relying on name overlap between docstrings and function names.

    Args:
        max_samples: Max number of query-code pairs to include
        seed: Random seed for sampling

    Returns:
        Path to the generated JSON file
    """
    datasets = _ensure_datasets_lib()
    random.seed(seed)

    print("  Loading AdvTest (CodeXGLUE NL-Code-Search-Adv)...")
    try:
        ds = datasets.load_dataset(
            "google/code_x_glue_tc_nl_code_search_adv",
            split="test",
        )
    except Exception as e:
        print(f"  [!] Failed to load AdvTest: {e}")
        return DATASETS_DIR / "advtest_benchmark.json"

    print(f"  Loaded {len(ds)} test samples")

    indices = list(range(len(ds)))
    if len(indices) > max_samples:
        random.shuffle(indices)
        indices = indices[:max_samples]

    queries = []
    corpus = []
    qrels = []

    for i, idx in enumerate(indices):
        row = ds[idx]
        docstring = (row.get("docstring") or "").strip()
        code = (row.get("code") or "").strip()
        func_name = row.get("func_name", "")
        repo = row.get("repo", row.get("nwo", ""))
        path = row.get("path", "")

        if not docstring or not code or len(docstring) < 10:
            continue

        qid = f"advtest_q_{i}"
        doc_id = f"advtest_d_{idx}"

        queries.append({
            "id": qid,
            "query": docstring,
            "language": "python",
            "source": "advtest",
        })

        corpus.append({
            "id": doc_id,
            "content": code,
            "language": "python",
            "func_name": func_name,
            "repo": repo,
            "file_path": path,
        })

        qrels.append({
            "query_id": qid,
            "doc_id": doc_id,
            "relevance": 1,
        })

    output = {
        "_description": "AdvTest benchmark — adversarial code search with normalized function names",
        "_source": "google/code_x_glue_tc_nl_code_search_adv (HuggingFace)",
        "_citation": "Lu et al., 2021. CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation",
        "_note": "Function names replaced with generic identifiers to prevent trivial keyword matching",
        "languages": ["python"],
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels,
    }

    out_path = DATASETS_DIR / "advtest_benchmark.json"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(queries)} queries -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Download all
# ---------------------------------------------------------------------------

def download_all(
    csn_max_per_lang: int = 500,
    cosqa_max: int = 500,
    seed: int = 42,
) -> dict[str, Path]:
    """Download all validated benchmark datasets.

    Returns dict of dataset_name -> file_path.
    """
    paths = {}

    print("\n=== Downloading CodeSearchNet ===")
    paths["codesearchnet"] = download_codesearchnet(
        max_per_language=csn_max_per_lang, seed=seed
    )

    print("\n=== Downloading CoSQA ===")
    paths["cosqa"] = download_cosqa(max_samples=cosqa_max, seed=seed)

    print("\n=== Downloading CodeSearchNet Challenge ===")
    paths["codesearchnet_challenge"] = download_codesearchnet_challenge()

    print(f"\n=== All datasets downloaded to {DATASETS_DIR} ===")
    return paths
