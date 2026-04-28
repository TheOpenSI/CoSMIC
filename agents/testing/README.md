# Agent scalability testing

This package measures coordinator behavior as the number of domain **sub-agents** grows (tiers **3**, **5**, and **8**). It does **not** replace the production coordinator in `[agents/coordinator/agent.py](../coordinator/agent.py)`; use `[runner.py](runner.py)` and `[benchmark.py](benchmark.py)` only for experiments.

## 1:1 mapping


| Scalability case | `presets.yaml` key |
| ---------------- | ------------------ |
| 3 services       | `tier_3`           |
| 5 services       | `tier_5`           |
| 8 services       | `tier_8`           |


Edit `[presets.yaml](presets.yaml)` to change which specialist slugs are registered per tier. Specialist definitions and order are in `[specialists.py](specialists.py)` (`ALL_KEYS`).

## Domain specialists (name and description)


| Dataset                    | Description                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `abstract_algebra`         | Contains theoretical mathematics problems focused on algebraic structures, used to evaluate abstract reasoning routing.                      |
| `anatomy`                  | Includes questions about human body structure and systems, helping identify life science and medical queries.                                |
| `astronomy`                | Covers celestial objects and space-related concepts, supporting routing for physics-oriented queries.                                        |
| `business_ethics`          | Consists of ethical decision-making scenarios in business contexts, useful for social science reasoning.                                     |
| `clinical_knowledge`       | Covers patient-centered medical scenarios involving diagnosis, symptoms, or treatment decisions, used to route clinical reasoning tasks.     |
| `college_biology`          | Focuses on theoretical and conceptual biology such as genetics, evolution, and cellular processes, without clinical or patient context.      |
| `college_chemistry`        | Includes chemistry problems involving reactions, equations, and physical or organic principles, supporting chemistry-specific query routing. |
| `college_computer_science` | Covers algorithms and data structures, used for routing technical and computational queries.                                                 |
| `mathematics`              | Contains general math problems across topics, supporting quantitative reasoning routing.                                                     |
| `medicine`                 | Includes broad medical knowledge questions, enabling routing for healthcare-related queries.                                                 |


## Running the benchmark

From the **repository root**, with your LLM endpoints configured (same as main CoSMIC, see `[agents/config.py](../config.py)`):

```bash
python -m agents.testing.benchmark --tier 3
python -m agents.testing.benchmark --tier 5 --runs 2 --warmup 1
python -m agents.testing.benchmark --tier 8 --csv path/to/custom.csv
```

- `**--tier**`: `3`, `5`, or `8` (required).
- `**--csv**`: optional; overrides the default path for that tier (see `[data/README.md](data/README.md)`).
- `**--runs**`: repeat the full task list (default `1`).
- `**--warmup**`: extra iterations before timing (default `0`).

If the default CSV for a tier is missing, the harness uses short **inline** prompts—one per active specialist—so you can still measure latency without data files.

### Environment

- `COSMIC_TEST_TIER`: default tier (`3`, `5`, or `8`) when calling `[runner.run_agent](runner.py)` with `tier=None`.
- `COSMIC_AGENT_MODEL`, `COSMIC_COORDINATOR_MODEL`: model ids for specialists and coordinator (`[agents/config.py](../config.py)`).

## Programmatic use

```python
from agents.testing import build_root_agent, load_tier, tier_keys, SPECIALISTS

keys = tier_keys(5)
agent = load_tier(5)  # or build_root_agent(keys)
```

To run one query asynchronously:

```python
import asyncio
from agents.testing.runner import run_agent

async def main():
    text = await run_agent(user_id="bench-1", query="What is a normal subgroup?", tier=3)
    print(text)

asyncio.run(main())
```

## Registering these specialists in a custom coordinator

1. Import the agents you need from `agents.testing.specialists` (or `SPECIALISTS[key]`).
2. Build an `Agent` coordinator with `sub_agents=[...]` and instructions that list each sub-agent name and when to route to it (same pattern as `[agents/coordinator/agent.py](../coordinator/agent.py)`).
3. Pass that coordinator to `google.adk.runners.Runner`.

Do not add these testing specialists to the production coordinator unless you intend to ship that behavior.

## CSV format

See `[data/README.md](data/README.md)` for the shared column schema and the three-file convention.