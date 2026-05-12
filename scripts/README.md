# scripts/

Standalone scripts that aren't part of the benchmark runtime.

| File | Purpose |
|------|---------|
| `generate_charts.py` | Regenerates `assets/charts/results.png` from the latest run under `benchmarks/results/`. Re-run after a full benchmark to refresh the README chart. |

Run with `uv run python scripts/<name>.py` so the script picks up the
project's resolved dependencies (matplotlib, reportlab, etc.).
