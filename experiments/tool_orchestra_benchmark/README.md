# ToolOrchestra Benchmarking Module

This module benchmarks service routing with a label-only classifier interface.

## Location

`experiments/tool_orchestra_benchmark/`

## Supported Models

- `--model toolorchestra` -> `nvidia/Nemotron-Orchestrator-8B`
- `--model openai` -> `text-embedding-3-large` (embedding baseline router)

Only one model runs per execution.

## Dataset Inputs

- Prompt dataset CSV: defaults to `datasets/dataset_3_services.csv`
  - Required columns: `prompt`, `expected_service`
- Service descriptions sheet: defaults to `datasets/dataset_descriptions.xlsx`
  - Optional reference for label/description context during ToolOrchestra prompting
  - Never used to alter dataset ground-truth labels

## Unified Interface

Both model wrappers expose:

```python
predict(prompt: str) -> str
```

The return value is always normalized to a single tool label.

## Run

From repo root:

```bash
python experiments/tool_orchestra_benchmark/run_benchmark.py \
  --model toolorchestra \
  --dataset datasets/dataset_3_services.csv
```

Or:

```bash
python experiments/tool_orchestra_benchmark/run_benchmark.py \
  --model openai \
  --dataset datasets/dataset_3_services.csv
```

## Outputs

Outputs are written to:

- `experiments/tool_orchestra_benchmark/outputs/<model>-routing-model-outputs/<dataset_name>/`
  - `predictions.csv`
  - `metrics_summary.csv`
  - `latency_log.csv`
  - `visualizations/confusion_matrix.png`
  - `visualizations/per_class_accuracy.png`
  - `visualizations/model_accuracy.png`

`predictions.csv` columns:

- `prompt`
- `expected_service`
- `predicted_service`
- `is_correct`
- `model_name`
- `dataset_name`
