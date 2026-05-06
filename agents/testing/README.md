# Agent scalability testing

This package evaluates **routing**: how reliably the scalability-test coordinator hands each question to the right domain specialist, as you increase the number of registered **services** (tiers **3**, **5**, **8**, and **10**).

It does **not** replace the production coordinator in [`agents/coordinator/agent.py`](../coordinator/agent.py). Use [`runner.py`](runner.py) and [`benchmark.py`](benchmark.py) only for experiments.

## Preset tiers (services per scenario)

| Scalability case | `presets.yaml` key | Typical scope |
| ---------------- | ------------------ | ------------- |
| 3 services       | `tier_3`           | First three entries in [`ALL_KEYS`](specialists.py) (unless you edit presets). |
| 5 services       | `tier_5`           | First five. |
| 8 services       | `tier_8`           | First eight. |
| 10 services      | `tier_10`          | Full [`ALL_KEYS`](specialists.py) (all specialists). |

Edit [`presets.yaml`](presets.yaml) to change which specialist **services** are registered per tier. Definitions and canonical order live in [`specialists.py`](specialists.py) (`ALL_KEYS`).

## Domain specialists (service id and role)

These ids are the **`service`** labels in CSVs and the sub-agent **`name`** strings used when routing.


| Dataset / service id        | Role |
| ---------------------------- | ---- |
| `abstract_algebra`            | Algebraic structures and abstract reasoning benchmarks. |
| `anatomy`                     | Body structure and organ systems. |
| `astronomy`                   | Celestial and space phenomena. |
| `business_ethics`             | Ethical dilemmas in business settings. |
| `clinical_knowledge`          | Clinical scenarios (diagnosis, symptoms, treatments). |
| `college_biology`             | Conceptual biology without patient-centric framing. |
| `college_chemistry`           | Reaction and theory chemistry tasks. |
| `college_computer_science`    | Algorithms, data structures, CS theory. |
| `mathematics`                 | Broader quantitative problems. |
| `medicine`                    | Broader medical knowledge. |


## Running the routing benchmark

From the **repository root**, with LLM endpoints configured like the rest of CoSMIC ([`agents/config.py`](../config.py)):

```bash
python -m agents.testing.benchmark --tier 5
python -m agents.testing.benchmark --tier 10 --csv path/to/custom.csv
python -m agents.testing.benchmark --all
```

With Docker and the bundled Compose stack (Ollama on **`cosmic_net`**):  
`docker compose -f docker-compose.benchmark.yaml run --rm --build benchmark --tier 5` (override CLI args as needed; see comments in [`docker-compose.benchmark.yaml`](../../docker-compose.benchmark.yaml)).

- **`--tier`**: one of `3`, `5`, `8`, `10`. Loads evaluation rows from default CSV unless `--csv` is set.
- **`--all`**: runs tiers **3, 5, 8,** and **10** in sequence using each tier’s default CSV from [`data/README.md`](data/README.md).
- **`--csv`**: only with **`--tier`**, overrides that tier’s CSV path.
- **`--out-dir`**: directory for outputs (defaults to [`results/`](results/)). Writes:
  - **`summary_tier{N}.csv`**: summary metrics for a specific tier run.
  - **`summary_all_tiers.csv`**: combined summary metrics for all tiers (only with `--all`).
  - **`per_sample_tier{N}.csv`**: per-row experiment log, including full model **`response`**. Updated incrementally: each prompt’s row is written and flushed as soon as that run finishes.
- **`-v` / `--verbose`**: log each ADK event (tools, transfers, final responses) per sample on stderr.

**`per_sample_tier{N}.csv`** has one row per evaluation prompt:
- **`run_id`**: unique UUID for the benchmark session.
- **`timestamp`**: ISO timestamp of the sample run.
- **`row_index`**: 1-based index in the evaluation CSV.
- **`dataset`**: name of the source CSV file.
- **`tier`**: the scalability tier (3, 5, 8, or 10).
- **`question`**: the user prompt.
- **`gold`**: the expected specialist ID.
- **`predicted_service_raw`**: the raw router output (e.g., `transfer_to_agent('math')`).
- **`predicted`**: the normalized prediction label (or `__no_transfer__`).
- **`correct`**: boolean indicating if `predicted == gold`.
- **`response`**: full text response from the coordinator.
- **`prompt_tokens`**: tokens in the request.
- **`completion_tokens`**: tokens in the response.
- **`latency_ms`**: total round-trip time in milliseconds.
- **`error_type`**: one of `routing_failure`, `parsing_error`, `timeout`, or null.

**`summary_tier{N}.csv`** (and `summary_all_tiers.csv`) contains:
- **`run_id`**, **`total_samples`**, **`accuracy`**, **`latency_avg_ms`**, **`latency_p95_ms`**, **`total_prompt_tokens`**, **`total_completion_tokens`**, **`routing_failures`**, **`best_class`**, **`worst_class`**.

**`routing_failures`** counts samples with no predicted transfer.

**`top3_accuracy` is omitted**: the coordinator selects a single `transfer_to_agent` target per question; ranking top-3 would require extra evaluation passes or API changes.


### Environment

- `COSMIC_TEST_TIER`: default tier (`3`, `5`, `8`, or `10`) when [`runner.run_agent`](runner.py) is called with `tier=None`.
- `COSMIC_AGENT_MODEL`, `COSMIC_COORDINATOR_MODEL`: specialist and coordinator model IDs ([`agents/config.py`](../config.py)).
- **`OLLAMA_API_BASE`**: base URL for the **LiteLLM** Ollama integration (e.g. `http://ollama:11434`). Defaults to localhost; **inside Docker**, `localhost:11434` is the container itself, so calls to `ollama_chat/...` models fail with **connection refused** unless you set this (or **`COSMIC_OLLAMA_API_BASE`**).
- **`COSMIC_OLLAMA_API_BASE`**: optional CoSMIC alias; if set, [`agents/config.py`](../config.py) applies it to `OLLAMA_API_BASE` when that is not already set.

**Ollama reachability from a benchmark container** (not the same as Open WebUI’s `OLLAMA_BASE_URL`; that env is not read by LiteLLM):

| Where Ollama runs | Typical value |
| ----------------- | ------------- |
| Compose service `ollama` on the same user-defined network (e.g. `cosmic_net`) | `http://ollama:11434` |
| Ollama on the Docker host (Desktop) | `http://host.docker.internal:11434` |
| Ollama on the Docker host (Linux) | `http://host.docker.internal:11434` with `extra_hosts: ["host.docker.internal:host-gateway"]` on the benchmark service, or the bridge gateway / host LAN IP |

**Sanity check** (inside the container): `curl -sS "$OLLAMA_API_BASE/api/tags"` (or `/api/version`) should succeed before running the benchmark.

**Compose example:** see [`docker-compose.benchmark.yaml`](../../docker-compose.benchmark.yaml) at the repo root (`OLLAMA_API_BASE=http://ollama:11434`, same network as `ollama`).

## Programmatic use

```python
from agents.testing import build_root_agent, load_tier, tier_keys, SPECIALISTS

keys = tier_keys(10)
agent = load_tier(10)  # or build_root_agent(keys)
```


### One routed query (`run_agent`)

```python
import asyncio
from agents.testing.runner import run_agent

async def main():
    text = await run_agent(user_id="bench-1", query="What is a normal subgroup?", tier=3)
    print(text)

asyncio.run(main())
```

### Structured routing metrics (`run_agent_routing`)

Use this when building tools or alternative analysis pipelines:

```python
import asyncio
from agents.testing.runner import run_agent_routing

async def main():
    r = await run_agent_routing(user_id="bench-2", query="...", tier=10)
    print(r.predicted_service, r.latency_ms, r.tokens_total)

asyncio.run(main())
```

## Registering these specialists in a custom coordinator

1. Import the agents you need from `agents.testing.specialists` (or `SPECIALISTS[key]`).
2. Build an `Agent` coordinator with `sub_agents=[...]` and routing instructions listing each sub-agent name (same pattern as [`agents/coordinator/agent.py`](../coordinator/agent.py)).
3. Pass that coordinator to `google.adk.runners.Runner`.

Do not attach these testing specialists to production unless that is intentional.


## Evaluation CSV format

See [`data/README.md`](data/README.md) for filenames and the shared column schema (`question`, `service`).
