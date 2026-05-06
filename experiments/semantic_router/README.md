# Semantic Router Benchmarking Module

This module benchmarks embedding-based semantic routing for CoSMIC using cosine similarity only.

## Location

`experiments/semantic_router/`

## What It Does

- Builds a service index from dataset descriptions.
- Routes prompts to services using embedding cosine similarity.
- Evaluates top-1 accuracy, top-3 accuracy, and per-class accuracy.
- Produces reproducible benchmark artifacts for each model run.

## Supported Embedding Models

- `--model miniLM` -> `all-MiniLM-L6-v2` (local)
- `--model bge` -> `bge-base-en-v1.5` (local)
- `--model openai` -> `text-embedding-3-large` (OpenAI API)

Only one model is executed per run.

## Required Datasets

- Prompt datasets matching: `datasets/dataset_<N>_services.csv`
  - Example: `dataset_3_services.csv`, `dataset_5_services.csv`, `dataset_8_services.csv`, `dataset_10_services.csv`
  - Required columns: `prompt`, `expected_service`
- Service descriptions: `datasets/dataset_descriptions.xlsx`
  - Required columns: `Dataset`, `Description`

## Install Dependencies

From repo root:

```bash
pip install numpy pandas matplotlib openpyxl transformers torch openai
```

## Run

From repo root:

```bash
python experiments/semantic_router/run.py --model miniLM
python experiments/semantic_router/run.py --model bge
python experiments/semantic_router/run.py --model openai
```

For `--model openai`, if `OPENAI_API_KEY` is not already set, the script prompts in terminal:

```bash
Enter your OpenAI API key:
```

You can also pre-set it:

```bash
export OPENAI_API_KEY="your-key"
python experiments/semantic_router/run.py --model openai
```

## Output Artifacts

Outputs are now organized by model, then dataset:

- `outputs/<model>-embedding-model-outputs/`
  - `all_datasets_metrics_summary.csv`
  - `<dataset_name>/predictions.csv`
  - `<dataset_name>/metrics_summary.csv`
  - `<dataset_name>/per_class_metrics_history.csv`
  - `<dataset_name>/visualizations/model_accuracy.png`
  - `<dataset_name>/visualizations/per_class_accuracy.png`
  - `<dataset_name>/visualizations/confusion_matrix.png`

Example for OpenAI:

- `outputs/openai-embedding-model-outputs/dataset_3_services/...`
- `outputs/openai-embedding-model-outputs/dataset_5_services/...`
- `outputs/openai-embedding-model-outputs/dataset_8_services/...`
- `outputs/openai-embedding-model-outputs/dataset_10_services/...`

## Notes

- No LLM inference is used.
- No prompt engineering or generative reasoning is used.
- Routing is embedding-only and cosine-only.
