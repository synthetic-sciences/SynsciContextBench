"""Benchmark configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load benchmarks/.env.local (does not override existing env vars)
load_dotenv(Path(__file__).parent / ".env.local")


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model."""

    provider: str  # gemini | anthropic | openai
    model: str  # model ID
    tier: str  # low | mid | high
    api_key: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def short_name(self) -> str:
        return f"{self.provider}_{self.tier}"


def _get_model_matrix() -> list[LLMModelConfig]:
    """Build the multi-model matrix from environment variables.

    Reads BENCH_{PROVIDER}_{TIER}_MODEL and BENCH_{PROVIDER}_API_KEY.
    Only includes models where both the model name and API key are set.
    """
    providers = {
        "gemini": {
            "low": os.getenv("BENCH_GEMINI_LOW_MODEL", "gemini-2.0-flash-lite"),
            "mid": os.getenv("BENCH_GEMINI_MID_MODEL", "gemini-2.0-flash"),
            "high": os.getenv("BENCH_GEMINI_HIGH_MODEL", "gemini-2.5-pro"),
            "api_key": os.getenv("BENCH_GEMINI_API_KEY", ""),
        },
        "anthropic": {
            "low": os.getenv("BENCH_ANTHROPIC_LOW_MODEL", "claude-haiku-4-5-20251001"),
            "mid": os.getenv("BENCH_ANTHROPIC_MID_MODEL", "claude-sonnet-4-6"),
            "high": os.getenv("BENCH_ANTHROPIC_HIGH_MODEL", "claude-opus-4-6"),
            "api_key": os.getenv("BENCH_ANTHROPIC_API_KEY", ""),
        },
        "openai": {
            "low": os.getenv("BENCH_OPENAI_LOW_MODEL", "gpt-4o-mini"),
            "mid": os.getenv("BENCH_OPENAI_MID_MODEL", "gpt-4o"),
            "high": os.getenv("BENCH_OPENAI_HIGH_MODEL", "o3"),
            "api_key": os.getenv("BENCH_OPENAI_API_KEY", ""),
        },
    }

    models = []
    for provider, cfg in providers.items():
        api_key = cfg["api_key"]
        if not api_key:
            continue
        for tier in ("low", "mid", "high"):
            model_name = cfg[tier]
            if model_name:
                models.append(LLMModelConfig(
                    provider=provider,
                    model=model_name,
                    tier=tier,
                    api_key=api_key,
                ))

    return models


@dataclass
class BenchmarkConfig:
    """Central config for benchmark runs."""

    # --- Synsc Context ---
    synsc_api_url: str = os.getenv("SYNSC_API_URL", "http://localhost:8000")
    synsc_api_key: str = os.getenv("SYNSC_API_KEY", "")

    # --- Nia (trynia) ---
    nia_api_url: str = os.getenv("NIA_API_URL", "https://apigcp.trynia.ai")
    nia_api_key: str = os.getenv("NIA_API_KEY", "")

    # --- Context7 ---
    context7_enabled: bool = os.getenv("CONTEXT7_ENABLED", "true").lower() == "true"
    context7_api_url: str = os.getenv("CONTEXT7_API_URL", "https://context7.com")
    context7_api_key: str = os.getenv("CONTEXT7_API_KEY", "")
    context7_npx_command: str = os.getenv("CONTEXT7_NPX_COMMAND", "npx")
    context7_request_delay: float = float(os.getenv("CONTEXT7_REQUEST_DELAY", "0.5"))

    # --- Single LLM (legacy, used when --multi-model is not set) ---
    llm_provider: str = os.getenv("BENCH_LLM_PROVIDER", "gemini")
    llm_model: str = os.getenv("BENCH_LLM_MODEL", "gemini-2.0-flash")
    llm_api_key: str = os.getenv("BENCH_LLM_API_KEY", "")

    # --- Multi-model matrix (populated on demand) ---
    model_matrix: list[LLMModelConfig] = field(default_factory=list)

    # --- Evaluation ---
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    similarity_threshold: float = 0.3
    max_queries: int | None = None  # Limit queries per dataset (None = all)
    # Seed list for query sub-sampling. A single benchmark run can replay
    # itself across multiple seeds; aggregate stats then report mean ± CI
    # over seeds rather than a single deterministic draw.
    seeds: list[int] = field(default_factory=lambda: [0])
    # Cap the number of results scored by the LLM judge in `validated_eval`.
    # The previous hard-coded cap of 3 silently forced rank-4+ to be
    # irrelevant. 10 lines up with the default reporting window.
    judge_top_k: int = 10

    # --- Paths ---
    # ``datasets_dir`` is the root that contains both ``curated/`` (hand-built
    # test cases owned by this repo) and ``validated/`` (downloaded standard
    # datasets like CodeSearchNet / CoSQA / AdvTest).
    datasets_dir: Path = Path(__file__).parent / "datasets"
    results_dir: Path = Path(__file__).parent / "results"

    @property
    def curated_dir(self) -> Path:
        """Hand-curated benchmark cases (Thesis, session replay, etc.)."""
        return self.datasets_dir / "curated"

    @property
    def validated_dir(self) -> Path:
        """Downloaded validated datasets (CodeSearchNet, CoSQA, AdvTest...)."""
        return self.datasets_dir / "validated"

    def load_model_matrix(self) -> list[LLMModelConfig]:
        """Load the multi-model matrix from env vars."""
        self.model_matrix = _get_model_matrix()
        return self.model_matrix

    def validate(self) -> list[str]:
        """Return list of missing required config items."""
        missing = []
        if not self.synsc_api_key:
            missing.append("SYNSC_API_KEY")
        if not self.nia_api_key:
            missing.append("NIA_API_KEY")
        return missing
