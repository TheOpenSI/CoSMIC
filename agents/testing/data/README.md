# Scalability testing data

Four **CSV files** correspond to the four scalability cases (**3**, **5**, **8**, and **10** registered services). Each file is one full evaluation split for routing experiments (`prompts_tier3.csv` … `prompts_tier10.csv`).

## File names and default paths

| Tier | Registered services (default preset) | Default CSV |
| ---- | ------------------------------------ | ----------- |
| 3 | First three keys in [`ALL_KEYS`](../specialists.py) (`tier_3`) | [`prompts_tier3.csv`](prompts_tier3.csv) |
| 5 | First five (`tier_5`) | [`prompts_tier5.csv`](prompts_tier5.csv) |
| 8 | First eight (`tier_8`) | [`prompts_tier8.csv`](prompts_tier8.csv) |
| 10 | All ten (`tier_10`, full [`ALL_KEYS`](../specialists.py)) | [`prompts_tier10.csv`](prompts_tier10.csv) |

Optional overrides live in [`../presets.yaml`](../presets.yaml): `csv_tier_3` … `csv_tier_10` (repo-root-relative paths or absolute).

## Column schema

Reuse the **same headers** wherever possible.

| Column | Required | Notes |
| ------ | -------- | ----- |
| `question` | **Yes** | User text routed through the coordinator. Aliases understood by [`benchmark.py`](../benchmark.py): `query`, `prompt`, `text`. |
| `service` | **Yes** (for evaluation rows) | Gold label: specialist **service id** present in [`SPECIALISTS`](../specialists.py) (`abstract_algebra`, …). Legacy aliases retained for ingestion: `specialist`, `dataset_slug`, `dataset`, `subject_slug`, `expected_service`. Rows whose gold service is missing or **not** in that tier’s preset list are skipped (count printed on stderr). |
| `id` | No | Optional stable identifier. |

Extra columns (`subject`, `correct_answer`, `choices`) are ignored by the harness.

## Alignment rules

- Each `prompts_tierN.csv` row should reference only services declared under `tier_N` inside [`presets.yaml`](../presets.yaml).
- If a CSV is absent, [`benchmark.py`](../benchmark.py) **exits with an error**—prepare data files before running routed evaluation.

**Note:** **`top3_accuracy` is intentionally not tracked** yet; routing returns a single `transfer_to_agent` decision per prompt.
