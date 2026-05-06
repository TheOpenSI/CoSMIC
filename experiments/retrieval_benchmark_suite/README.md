# Retrieval Benchmark Suite

This module benchmarks retrieval-only service routing with three methods:

- `bi-encoder` (`sentence-transformers/all-MiniLM-L6-v2`)
- `cross-encoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `colbert` (`colbert-ir/colbertv2.0`) using late-interaction MaxSim ranking

The benchmark predicts `expected_service` from each dataset prompt using service descriptions from
`datasets/dataset_descriptions.xlsx`. It does not call OpenAI APIs and does not perform generation or tool execution.

## Run

```bash
python experiments/retrieval_benchmark_suite/cli.py \
  --method bi-encoder \
  --dataset datasets/dataset_3_services.csv
```

```bash
python experiments/retrieval_benchmark_suite/cli.py \
  --method cross-encoder \
  --dataset datasets/dataset_5_services.csv
```

```bash
python experiments/retrieval_benchmark_suite/cli.py \
  --method colbert \
  --dataset datasets/dataset_8_services.csv
```

## Output Structure

Per-run files are written to:

- `experiments/retrieval_benchmark_suite/outputs/<method>/<dataset_name>/predictions.csv`
- `experiments/retrieval_benchmark_suite/outputs/<method>/<dataset_name>/metrics_summary.csv`

Visualizations are written to:

- `experiments/retrieval_benchmark_suite/outputs/visualizations/<method>_<dataset_name>_confusion_matrix.png`
- `experiments/retrieval_benchmark_suite/outputs/visualizations/<method>_<dataset_name>_per_class_accuracy.png`
- `experiments/retrieval_benchmark_suite/outputs/visualizations/<dataset_name>_model_accuracy.png`
- `experiments/retrieval_benchmark_suite/outputs/visualizations/<dataset_name>_latency_comparison.png`
