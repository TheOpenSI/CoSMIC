# Benchmark outputs

Running `python -m agents.testing.benchmark` writes here by default:

- `summary_tier{N}.csv` — summary metrics for a specific scalability tier (accuracy, latency, tokens).
- `summary_all_tiers.csv` — combined summary metrics for tiers 3, 5, 8, and 10 (when using `--all`).
- `per_sample_tier{N}.csv` — per-row experiment log: full `response` text, routing fields, latency, tokens, and error types. Rows are **flushed after each prompt** to ensure partial results survive long runs.

Override the output directory with `--out-dir path/to/dir`.

See `[../README.md](../README.md)` for the full column list and CLI flags.
