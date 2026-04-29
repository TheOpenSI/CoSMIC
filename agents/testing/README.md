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

- **`--tier`**: one of `3`, `5`, `8`, `10`. Loads evaluation rows from default CSV unless `--csv` is set.
- **`--all`**: runs tiers **3, 5, 8,** and **10** in sequence using each tier’s default CSV from [`data/README.md`](data/README.md).
- **`--csv`**: only with **`--tier`**, overrides that tier’s CSV path.
- **`--out-dir`**: directory for outputs (defaults to [`results/`](results/)). Writes **`metrics.md`** (summary table) and **`confusion_tier{N}.png`** per tier evaluated (matplotlib + scikit-learn required for PNGs).

Each run prints a textual **confusion matrix** on stderr (truth vs predicted route, including `__no_transfer__` when the coordinator never calls `transfer_to_agent`). The same aggregate metrics appear as **stdout TSV**, and are **written** to **`results/metrics.md`** (Markdown table). **`token_cost`** is the summed **`total_token_count`** across ADK events—not USD. **`routing_failures`** counts samples with no predicted transfer.

**`top3_accuracy` is omitted**: the coordinator selects a single `transfer_to_agent` target per question; ranking top-3 would require extra evaluation passes or API changes.


### Environment

- `COSMIC_TEST_TIER`: default tier (`3`, `5`, `8`, or `10`) when [`runner.run_agent`](runner.py) is called with `tier=None`.
- `COSMIC_AGENT_MODEL`, `COSMIC_COORDINATOR_MODEL`: specialist and coordinator model IDs ([`agents/config.py`](../config.py)).

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
