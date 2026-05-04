# Benchmark outputs

Running `python -m agents.testing.benchmark` writes here by default:

- `metrics.md` — tabular routing metrics (one row per tier when using `--all`)
- `confusion_tier{N}.png` — confusion matrix plots (requires matplotlib)
- `predictions_tier{N}.csv` — per-row experiment log: full `response` text, routing fields, latency/tokens, and duplicated tier summary metrics (omit with `--no-per-row-csv`)

Override the directory with `--out-dir path/to/dir`.

See `[../README.md](../README.md)` for the full column list and CLI flags.
