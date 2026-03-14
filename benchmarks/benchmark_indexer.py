"""Direct-to-DB benchmark corpus indexer.

Bypasses the repo clone/file-walk/chunk pipeline and inserts benchmark
corpus snippets directly into the database as code_chunks with embeddings.

Each corpus snippet becomes exactly one chunk — perfect boundary alignment
for fair retrieval evaluation.

Usage:
    python -m benchmarks.benchmark_indexer --dataset cosqa
    python -m benchmarks.benchmark_indexer --dataset codesearchnet
    python -m benchmarks.benchmark_indexer --dataset all
    python -m benchmarks.benchmark_indexer --cleanup   # remove benchmark data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any synsc imports
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
import structlog
from sqlalchemy import text

from synsc.database.connection import get_session
from synsc.embeddings.generator import EmbeddingGenerator

logger = structlog.get_logger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"
MANIFEST_PATH = Path(__file__).parent / "benchmark_manifest.json"

# Repo name used for all benchmark data
BENCHMARK_REPO_NAME = "benchmark_context"
BENCHMARK_OWNER = "__benchmark__"


def _save_to_manifest(dataset_name: str, repo_id: str) -> None:
    """Save dataset -> repo_id mapping for the benchmark runner."""
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    manifest[dataset_name] = repo_id
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest() -> dict[str, str]:
    """Load the benchmark manifest (dataset_name -> repo_id)."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def _load_corpus(dataset_path: Path) -> tuple[list[dict], dict]:
    """Load corpus and metadata from a benchmark dataset JSON.

    Returns:
        (corpus_docs, metadata) where each doc has 'id', 'content', 'language'.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    corpus = data["corpus"]
    metadata = {
        "description": data.get("_description", dataset_path.stem),
        "num_queries": data.get("num_queries", 0),
        "num_corpus": len(corpus),
        "languages": data.get("languages", []),
    }
    return corpus, metadata


def _create_repository(session, dataset_name: str, metadata: dict) -> str:
    """Create a dummy repository row for the benchmark corpus.

    Returns the repo_id.
    """
    repo_id = str(uuid.uuid4())

    # Collect language stats from corpus
    languages = {}
    for lang in metadata.get("languages", []):
        languages[lang] = 1.0 / max(len(metadata.get("languages", [])), 1)

    session.execute(
        text("""
            INSERT INTO repositories (
                repo_id, url, owner, name, branch, commit_sha,
                is_public,
                files_count, chunks_count, total_lines, total_tokens,
                languages, indexed_at
            ) VALUES (
                :repo_id, :url, :owner, :name, 'main', :commit_sha,
                TRUE,
                :files_count, :chunks_count, 0, 0,
                :languages, NOW()
            )
        """),
        {
            "repo_id": repo_id,
            "url": f"benchmark://{dataset_name}",
            "owner": BENCHMARK_OWNER,
            "name": f"{BENCHMARK_REPO_NAME}_{dataset_name}",
            "commit_sha": f"bench_{dataset_name}_{int(time.time())}",
            "files_count": 1,
            "chunks_count": metadata["num_corpus"],
            "languages": json.dumps(languages),
        },
    )

    return repo_id


def _create_file(session, repo_id: str, dataset_name: str, language: str) -> str:
    """Create a dummy repository_file row.

    Returns the file_id.
    """
    file_id = str(uuid.uuid4())

    ext_map = {
        "python": "py", "javascript": "js", "java": "java",
        "go": "go", "ruby": "rb", "php": "php",
    }
    ext = ext_map.get(language, "txt")

    session.execute(
        text("""
            INSERT INTO repository_files (
                file_id, repo_id, file_path, file_name,
                language, line_count, token_count, size_bytes
            ) VALUES (
                :file_id, :repo_id, :file_path, :file_name,
                :language, 0, 0, 0
            )
        """),
        {
            "file_id": file_id,
            "repo_id": repo_id,
            "file_path": f"corpus/{dataset_name}.{ext}",
            "file_name": f"{dataset_name}.{ext}",
            "language": language,
        },
    )

    return file_id


def _insert_chunks_and_embeddings(
    session,
    repo_id: str,
    file_ids: dict[str, str],
    corpus: list[dict],
    embedding_generator: EmbeddingGenerator,
    batch_size: int = 100,
) -> int:
    """Insert corpus docs as code_chunks and generate + insert embeddings.

    Args:
        session: DB session (caller manages transaction).
        repo_id: Repository ID.
        file_ids: Map of language -> file_id.
        corpus: List of corpus docs with 'id', 'content', 'language'.
        embedding_generator: Gemini embedding generator.
        batch_size: Batch size for embedding generation.

    Returns:
        Number of chunks inserted.
    """
    total = len(corpus)
    inserted = 0

    for batch_start in range(0, total, batch_size):
        batch = corpus[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        # Create chunk rows
        chunk_ids = []
        contents = []
        for i, doc in enumerate(batch):
            chunk_id = str(uuid.uuid4())
            language = doc.get("language", "python")
            file_id = file_ids.get(language, list(file_ids.values())[0])

            session.execute(
                text("""
                    INSERT INTO code_chunks (
                        chunk_id, repo_id, file_id, chunk_index,
                        content, start_line, end_line,
                        chunk_type, language, token_count
                    ) VALUES (
                        :chunk_id, :repo_id, :file_id, :chunk_index,
                        :content, 1, :end_line,
                        'code', :language, :token_count
                    )
                """),
                {
                    "chunk_id": chunk_id,
                    "repo_id": repo_id,
                    "file_id": file_id,
                    "chunk_index": batch_start + i,
                    "content": doc["content"],
                    "end_line": doc["content"].count("\n") + 1,
                    "language": language,
                    "token_count": len(doc["content"].split()),
                },
            )

            chunk_ids.append(chunk_id)
            contents.append(doc["content"])

        # Generate embeddings for this batch
        max_retries = 5
        for attempt in range(max_retries):
            try:
                embeddings = embedding_generator.generate(contents)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        # Insert embeddings
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            session.execute(
                text("""
                    INSERT INTO chunk_embeddings (chunk_id, repo_id, embedding)
                    VALUES (:chunk_id, :repo_id, :embedding)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding
                """),
                {
                    "chunk_id": chunk_id,
                    "repo_id": repo_id,
                    "embedding": str(embedding.tolist()),
                },
            )

        inserted += len(batch)
        print(f"    Batch {batch_num}/{total_batches}: {inserted}/{total} chunks indexed")

    return inserted


def _add_to_user_collection(session, repo_id: str, user_id: str) -> None:
    """Add the benchmark repo to a user's collection."""
    session.execute(
        text("""
            INSERT INTO user_repositories (user_id, repo_id)
            VALUES (:user_id, :repo_id)
            ON CONFLICT (user_id, repo_id) DO NOTHING
        """),
        {"user_id": user_id, "repo_id": repo_id},
    )


def index_benchmark_dataset(
    dataset_path: Path,
    dataset_name: str,
    user_id: str | None = None,
) -> str:
    """Index a benchmark dataset directly into the database.

    Args:
        dataset_path: Path to the benchmark JSON file.
        dataset_name: Short name (e.g., 'cosqa', 'codesearchnet').
        user_id: Optional user ID to add repo to their collection.

    Returns:
        The repo_id of the created benchmark repository.
    """
    print(f"\n{'='*60}")
    print(f"Indexing benchmark: {dataset_name}")
    print(f"{'='*60}")

    corpus, metadata = _load_corpus(dataset_path)
    print(f"  Corpus size: {len(corpus)} snippets")
    print(f"  Languages: {metadata['languages']}")

    embedding_generator = EmbeddingGenerator()
    start_time = time.time()

    with get_session() as session:
        # Set long timeout for large corpora
        session.execute(text("SET LOCAL statement_timeout = '600s'"))

        # Check if already indexed
        existing = session.execute(
            text("""
                SELECT repo_id FROM repositories
                WHERE owner = :owner AND name = :name
            """),
            {
                "owner": BENCHMARK_OWNER,
                "name": f"{BENCHMARK_REPO_NAME}_{dataset_name}",
            },
        ).fetchone()

        if existing:
            repo_id = str(existing[0])
            print(f"  Already indexed as repo_id={repo_id}")
            if user_id:
                _add_to_user_collection(session, repo_id, user_id)
                session.commit()
                print(f"  Added to user {user_id}'s collection")
            return repo_id

        # Create repository
        repo_id = _create_repository(session, dataset_name, metadata)
        print(f"  Created repo: {repo_id}")

        # Create one file per language
        languages = set(doc.get("language", "python") for doc in corpus)
        file_ids = {}
        for lang in languages:
            file_ids[lang] = _create_file(session, repo_id, dataset_name, lang)
        print(f"  Created {len(file_ids)} file(s): {list(languages)}")

        session.flush()

        # Insert chunks + embeddings
        inserted = _insert_chunks_and_embeddings(
            session=session,
            repo_id=repo_id,
            file_ids=file_ids,
            corpus=corpus,
            embedding_generator=embedding_generator,
        )

        # Update repo stats
        session.execute(
            text("""
                UPDATE repositories
                SET chunks_count = :count
                WHERE repo_id = :repo_id
            """),
            {"count": inserted, "repo_id": repo_id},
        )

        # Add to user collection if specified
        if user_id:
            _add_to_user_collection(session, repo_id, user_id)

        session.commit()

    elapsed = time.time() - start_time
    print(f"\n  Done! {inserted} chunks indexed in {elapsed:.1f}s")
    print(f"  Repo ID: {repo_id}")

    # Save repo_id to manifest for benchmark runner
    _save_to_manifest(dataset_name, repo_id)

    return repo_id


def cleanup_benchmark_data() -> None:
    """Remove all benchmark repos, chunks, and embeddings from the DB."""
    print("\nCleaning up benchmark data...")

    with get_session() as session:
        # Find all benchmark repos
        rows = session.execute(
            text("SELECT repo_id FROM repositories WHERE owner = :owner"),
            {"owner": BENCHMARK_OWNER},
        ).fetchall()

        if not rows:
            print("  No benchmark data found.")
            return

        repo_ids = [str(r[0]) for r in rows]
        print(f"  Found {len(repo_ids)} benchmark repo(s)")

        for repo_id in repo_ids:
            session.execute(
                text("DELETE FROM chunk_embeddings WHERE repo_id = :id"),
                {"id": repo_id},
            )
            session.execute(
                text("DELETE FROM code_chunks WHERE repo_id = :id"),
                {"id": repo_id},
            )
            session.execute(
                text("DELETE FROM repository_files WHERE repo_id = :id"),
                {"id": repo_id},
            )
            session.execute(
                text("DELETE FROM user_repositories WHERE repo_id = :id"),
                {"id": repo_id},
            )
            session.execute(
                text("DELETE FROM repositories WHERE repo_id = :id"),
                {"id": repo_id},
            )
            print(f"  Deleted repo {repo_id}")

        session.commit()

    print("  Cleanup complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index benchmark datasets directly into the database",
    )
    parser.add_argument(
        "--dataset",
        choices=["cosqa", "codesearchnet", "all"],
        default="cosqa",
        help="Which dataset to index (default: cosqa)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID to add benchmark repo to their collection",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove all benchmark data from the database",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cleanup:
        cleanup_benchmark_data()
        return 0

    datasets_to_index = []
    if args.dataset in ("cosqa", "all"):
        path = DATASETS_DIR / "cosqa_benchmark.json"
        if path.exists():
            datasets_to_index.append((path, "cosqa"))
        else:
            print(f"[!] {path} not found — run --download-datasets first")

    if args.dataset in ("codesearchnet", "all"):
        path = DATASETS_DIR / "codesearchnet_benchmark.json"
        if path.exists():
            datasets_to_index.append((path, "codesearchnet"))
        else:
            print(f"[!] {path} not found — run --download-datasets first")

    if not datasets_to_index:
        print("[ERROR] No datasets to index.")
        return 1

    for path, name in datasets_to_index:
        index_benchmark_dataset(path, name, user_id=args.user_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
