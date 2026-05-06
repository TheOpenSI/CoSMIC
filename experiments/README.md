# Unified Benchmark CLI

This document explains how to run all benchmark modules through the root `main.py` entrypoint.

The existing benchmark module logic is unchanged. The root CLI only orchestrates calls into:

- `experiments/semantic_router/run.py`
- `experiments/tool_orchestra_benchmark/run_benchmark.py`
- `experiments/retrieval_benchmark_suite/cli.py`

## Quick Start

Run from repository root:

```bash
python main.py --benchmark-all
```

Run one module only:

```bash
python main.py --benchmark retrieval-suite
python main.py --benchmark semantic-router
python main.py --benchmark tool-orchestra
```

## Arguments

### Core Selection

- `--benchmark {semantic-router,tool-orchestra,retrieval-suite}`
  - Runs exactly one benchmark module.
- `--benchmark-all`
  - Runs all benchmark modules in sequence.

### Dataset and Metadata

- `--dataset <path>`
  - Repeat this flag to provide multiple datasets.
  - If omitted, defaults to:
    - `datasets/dataset_3_services.csv`
    - `datasets/dataset_5_services.csv`
    - `datasets/dataset_8_services.csv`
    - `datasets/dataset_10_services.csv`
- `--descriptions <path>`
  - Default: `datasets/dataset_descriptions.xlsx`

### Module-Specific Controls

- `--semantic-models miniLM bge openai`
  - Default: all three.
- `--tool-models toolorchestra openai`
  - Default: both.
- `--retrieval-methods bi-encoder cross-encoder colbert`
  - Default: all three.
- `--device {auto,cpu,cuda}`
  - Passed to modules that support device selection.
  - Retrieval suite receives `cpu` when `auto` is selected.
- `--checkpoint-every <int>`
  - Passed to ToolOrchestra benchmark.
  - Default: `10`.

## Common Command Patterns

Run all modules on only `dataset_3_services.csv`:

```bash
python main.py \
  --benchmark-all \
  --dataset datasets/dataset_3_services.csv
```

Run retrieval methods on two datasets:

```bash
python main.py \
  --benchmark retrieval-suite \
  --retrieval-methods bi-encoder cross-encoder colbert \
  --dataset datasets/dataset_3_services.csv \
  --dataset datasets/dataset_5_services.csv \
  --device cpu
```

Run semantic router with selected models:

```bash
python main.py \
  --benchmark semantic-router \
  --semantic-models miniLM bge
```

Run ToolOrchestra benchmark with explicit checkpoint interval:

```bash
python main.py \
  --benchmark tool-orchestra \
  --tool-models toolorchestra openai \
  --dataset datasets/dataset_8_services.csv \
  --checkpoint-every 20 \
  --device cpu
```

## Notes

- If no benchmark flags are provided, `main.py` keeps its original CoSMIC execution behavior.
- The unified benchmark CLI is orchestration-only and does not modify benchmark computation logic.
