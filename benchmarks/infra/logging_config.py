"""Structured logging and tracing for benchmark runs.

Provides:
- Structured JSON logging (file) + human-readable console output
- Per-query trace records with full request/response data
- Run metadata (git hash, system info, config snapshot)
- CSV export of query-level results for statistical analysis
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

@dataclass
class RunMetadata:
    """Captures environment and config for reproducibility."""

    run_id: str = ""
    timestamp: str = ""
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    python_version: str = ""
    platform: str = ""
    platform_version: str = ""
    cpu_count: int = 0
    hostname: str = ""
    engines: list[str] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    benchmark_version: str = "1.0.0"

    @classmethod
    def capture(cls, engines: list[str] | None = None, config: Any = None) -> RunMetadata:
        meta = cls(
            run_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version.split()[0],
            platform=platform.system(),
            platform_version=platform.release(),
            cpu_count=os.cpu_count() or 0,
            hostname=platform.node(),
            engines=engines or [],
        )
        # Git info
        try:
            meta.git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()[:12]
            meta.git_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            meta.git_dirty = bool(subprocess.run(
                ["git", "diff", "--quiet"],
                capture_output=True, timeout=5,
            ).returncode)
        except Exception:
            pass

        if config is not None:
            sensitive = {"api_key", "api_secret", "token", "password"}
            meta.config_snapshot = {}
            for k, v in asdict(config).items():
                if k in ("model_matrix",) or isinstance(v, Path):
                    continue
                if any(s in k.lower() for s in sensitive):
                    meta.config_snapshot[k] = "***REDACTED***"
                elif isinstance(v, Path):
                    meta.config_snapshot[k] = str(v)
                else:
                    meta.config_snapshot[k] = v

        return meta


# ---------------------------------------------------------------------------
# Query trace
# ---------------------------------------------------------------------------

@dataclass
class QueryTrace:
    """Full trace of a single query execution."""

    trace_id: str = ""
    run_id: str = ""
    timestamp: str = ""
    engine: str = ""
    benchmark_type: str = ""  # retrieval, multihop, code_qa, adversarial, etc.
    query_id: str = ""
    query_text: str = ""
    query_metadata: dict = field(default_factory=dict)

    # Request
    request_params: dict = field(default_factory=dict)

    # Response
    num_results: int = 0
    results: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""
    error_category: str = ""  # timeout, auth, no_results, api_error, parse_error

    # Context / repo breakdown
    repos_in_results: list[str] = field(default_factory=list)
    languages_in_results: list[str] = field(default_factory=list)
    primary_repo: str = ""
    primary_language: str = ""

    # Token efficiency
    total_tokens_returned: int = 0
    relevant_tokens_returned: int = 0
    token_efficiency: float = 0.0  # relevant / total (0-1)

    # Evaluation
    relevance_judgments: list[dict] = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # mrr, precision@k, ndcg@k, etc.

    @classmethod
    def create(cls, run_id: str, engine: str, benchmark_type: str) -> QueryTrace:
        return cls(
            trace_id=uuid.uuid4().hex[:12],
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            engine=engine,
            benchmark_type=benchmark_type,
        )


# ---------------------------------------------------------------------------
# Trace store
# ---------------------------------------------------------------------------

class TraceStore:
    """Collects and persists query traces and run metadata."""

    def __init__(self, results_dir: Path):
        self.base_results_dir = results_dir
        self.results_dir = results_dir  # updated in start_run to per-run subdir
        self.traces: list[QueryTrace] = []
        self.metadata: RunMetadata | None = None
        self._run_id: str = ""
        self._start_time: float = 0.0
        self._benchmark_timings: dict[str, dict[str, float]] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def start_run(self, engines: list[str], config: Any = None) -> RunMetadata:
        self.metadata = RunMetadata.capture(engines=engines, config=config)
        self._run_id = self.metadata.run_id
        self._start_time = time.perf_counter()
        self.traces = []
        self._benchmark_timings = {}

        # Create per-run directory with categorized subfolders
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_results_dir / f"run_{ts}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("traces", "logs", "reports", "data"):
            (self.results_dir / sub).mkdir(exist_ok=True)

        # Move log file handler into logs/ subfolder
        relocate_log_handler(self.results_dir / "logs")

        return self.metadata

    def start_benchmark(self, benchmark_type: str) -> None:
        self._benchmark_timings[benchmark_type] = {
            "start": time.perf_counter(),
        }

    def end_benchmark(self, benchmark_type: str) -> float:
        if benchmark_type in self._benchmark_timings:
            elapsed = time.perf_counter() - self._benchmark_timings[benchmark_type]["start"]
            self._benchmark_timings[benchmark_type]["duration_s"] = elapsed
            return elapsed
        return 0.0

    def add_trace(self, trace: QueryTrace) -> None:
        trace.run_id = self._run_id
        self.traces.append(trace)

    def total_duration_s(self) -> float:
        return time.perf_counter() - self._start_time

    def save(self, report: dict | None = None) -> dict[str, Path]:
        """Save all trace data into categorized subfolders. Returns paths to saved files."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, Path] = {}

        # 1. Full traces (JSONL) → traces/
        traces_path = self.results_dir / "traces" / f"traces_{timestamp}.jsonl"
        with open(traces_path, "w") as f:
            for trace in self.traces:
                f.write(json.dumps(asdict(trace), default=str) + "\n")
        paths["traces"] = traces_path

        # 2. Run manifest → reports/
        manifest = {
            "metadata": asdict(self.metadata) if self.metadata else {},
            "total_duration_s": self.total_duration_s(),
            "benchmark_timings": {
                k: v.get("duration_s", 0.0)
                for k, v in self._benchmark_timings.items()
            },
            "trace_count": len(self.traces),
            "trace_file": str(traces_path.name),
            "engines_summary": self._engine_summary(),
        }
        if report:
            manifest["report"] = report
        manifest_path = self.results_dir / "reports" / f"manifest_{timestamp}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        paths["manifest"] = manifest_path

        # 3. CSV export → data/
        csv_path = self._export_csv(timestamp)
        if csv_path:
            paths["csv"] = csv_path

        return paths

    def _engine_summary(self) -> dict:
        """Aggregate trace stats per engine."""
        summary: dict[str, dict] = {}
        for t in self.traces:
            if t.engine not in summary:
                summary[t.engine] = {
                    "total_queries": 0,
                    "errors": 0,
                    "error_categories": {},
                    "avg_latency_ms": 0.0,
                    "latencies": [],
                    "benchmark_types": set(),
                    "token_efficiencies": [],
                    "per_repo": {},
                    "per_language": {},
                }
            s = summary[t.engine]
            s["total_queries"] += 1
            if t.error:
                s["errors"] += 1
                cat = t.error_category or "unknown"
                s["error_categories"][cat] = s["error_categories"].get(cat, 0) + 1
            s["latencies"].append(t.latency_ms)
            s["benchmark_types"].add(t.benchmark_type)
            if t.token_efficiency > 0:
                s["token_efficiencies"].append(t.token_efficiency)

            # Per-repo aggregation
            if t.primary_repo:
                repo = t.primary_repo
                if repo not in s["per_repo"]:
                    s["per_repo"][repo] = {"count": 0, "mrr_sum": 0.0, "latency_sum": 0.0}
                s["per_repo"][repo]["count"] += 1
                s["per_repo"][repo]["mrr_sum"] += t.scores.get("mrr", 0)
                s["per_repo"][repo]["latency_sum"] += t.latency_ms

            # Per-language aggregation
            if t.primary_language:
                lang = t.primary_language
                if lang not in s["per_language"]:
                    s["per_language"][lang] = {"count": 0, "mrr_sum": 0.0}
                s["per_language"][lang]["count"] += 1
                s["per_language"][lang]["mrr_sum"] += t.scores.get("mrr", 0)

        # Finalize
        for engine, s in summary.items():
            lats = s.pop("latencies")
            s["avg_latency_ms"] = sum(lats) / len(lats) if lats else 0.0
            s["p50_latency_ms"] = sorted(lats)[len(lats) // 2] if lats else 0.0
            s["p95_latency_ms"] = sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0
            s["min_latency_ms"] = min(lats) if lats else 0.0
            s["max_latency_ms"] = max(lats) if lats else 0.0
            s["benchmark_types"] = sorted(s["benchmark_types"])

            # Token efficiency
            effs = s.pop("token_efficiencies")
            s["avg_token_efficiency"] = sum(effs) / len(effs) if effs else 0.0

            # Finalize per-repo (avg MRR per repo)
            for repo, stats in s["per_repo"].items():
                stats["avg_mrr"] = stats.pop("mrr_sum") / stats["count"]
                stats["avg_latency_ms"] = stats.pop("latency_sum") / stats["count"]

            # Finalize per-language (avg MRR per language)
            for lang, stats in s["per_language"].items():
                stats["avg_mrr"] = stats.pop("mrr_sum") / stats["count"]

        return summary

    def _export_csv(self, timestamp: int) -> Path | None:
        """Export flat CSV for statistical analysis."""
        if not self.traces:
            return None

        csv_path = self.results_dir / "data" / f"results_{timestamp}.csv"

        # Collect all score keys across traces
        all_score_keys: set[str] = set()
        for t in self.traces:
            all_score_keys.update(t.scores.keys())
        score_cols = sorted(all_score_keys)

        fieldnames = [
            "run_id", "trace_id", "engine", "benchmark_type",
            "query_id", "query_text", "num_results", "latency_ms",
            "error", "error_category",
            "primary_repo", "primary_language",
            "total_tokens_returned", "relevant_tokens_returned", "token_efficiency",
        ] + score_cols

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for t in self.traces:
                row = {
                    "run_id": t.run_id,
                    "trace_id": t.trace_id,
                    "engine": t.engine,
                    "benchmark_type": t.benchmark_type,
                    "query_id": t.query_id,
                    "query_text": t.query_text[:200],
                    "num_results": t.num_results,
                    "latency_ms": f"{t.latency_ms:.2f}",
                    "error": t.error,
                    "error_category": t.error_category,
                    "primary_repo": t.primary_repo,
                    "primary_language": t.primary_language,
                    "total_tokens_returned": t.total_tokens_returned,
                    "relevant_tokens_returned": t.relevant_tokens_returned,
                    "token_efficiency": f"{t.token_efficiency:.3f}" if t.token_efficiency else "",
                }
                for k in score_cols:
                    row[k] = t.scores.get(k, "")
                writer.writerow(row)

        return csv_path


# ---------------------------------------------------------------------------
# Structured logger setup
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Outputs structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include extra fields
        for key in ("engine", "benchmark_type", "query_id", "latency_ms",
                     "num_results", "error_type", "run_id", "trace_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Clean console output with optional color."""

    COLORS = {
        "DEBUG": "\033[36m",    # cyan
        "INFO": "\033[32m",     # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        engine = getattr(record, "engine", "")
        prefix = f"[{engine}] " if engine else ""
        return f"{color}{record.levelname[0]}{reset} {prefix}{record.getMessage()}"


def setup_logging(
    results_dir: Path,
    run_id: str = "",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure structured logging for a benchmark run.

    Returns the root benchmark logger. All benchmark modules should use:
        logger = logging.getLogger("bench.<module_name>")

    Note: The file handler initially writes to results_dir. When
    TraceStore.start_run() creates the per-run subfolder, it calls
    relocate_log_handler() to move the log file into logs/.
    """
    logger = logging.getLogger("bench")
    logger.setLevel(logging.DEBUG)  # always capture DEBUG in file
    logger.handlers.clear()

    # File handler — structured JSON (initially in base results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"bench_{ts}.jsonl"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Console handler — INFO only (DEBUG goes to log file to avoid tqdm overlap)
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(ConsoleFormatter())
        logger.addHandler(ch)

    return logger


def relocate_log_handler(logs_dir: Path) -> None:
    """Move the bench file handler to a new logs/ directory.

    Removes the initial log file from the base results dir to avoid orphan files.
    """
    logger = logging.getLogger("bench")

    from datetime import datetime
    new_log_path = logs_dir / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and not isinstance(
            handler, logging.StreamHandler
        ):
            old_path = Path(handler.baseFilename)
            handler.close()
            logger.removeHandler(handler)
            # Move into the new logs dir, keeping the same filename
            new_log_path = logs_dir / old_path.name
            if old_path.exists():
                import shutil
                shutil.move(str(old_path), str(new_log_path))

    fh = logging.FileHandler(new_log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Print path for easy tail -f
    print(f"\n  Log file: {new_log_path}")
    print(f"  tail -f {new_log_path}\n")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the bench namespace."""
    return logging.getLogger(f"bench.{name}")
