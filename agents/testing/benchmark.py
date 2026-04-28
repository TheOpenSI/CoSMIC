"""CLI harness for scalability timing (tiers 3 / 5 / 8)."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
import uuid

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agents.testing.coordinator_factory import default_csv_path, tier_keys
from agents.testing.runner import run_agent


# One short prompt per specialist to exercise routing when no CSV exists.
_INLINE_PROMPTS: dict[str, str] = {
    "abstract_algebra": "In group theory, what is the definition of a normal subgroup?",
    "anatomy": "Name the four chambers of the human heart and their primary roles.",
    "astronomy": "What causes the phases of the Moon as seen from Earth?",
    "business_ethics": "When is it ethically justified for a manager to omit a risk from a client brief?",
    "clinical_knowledge": (
        "A patient has sudden unilateral leg swelling and pleuritic chest pain; what diagnosis should "
        "be considered first?"
    ),
    "college_biology": "Contrast mitosis and meiosis in terms of chromosome segregation.",
    "college_chemistry": "Balance the reaction: Fe + O2 -> Fe2O3 (include stoichiometric coefficients).",
    "college_computer_science": "What is the time complexity of lookup in a balanced BST with n nodes?",
    "mathematics": "Solve for x: 2^x = 128.",
    "medicine": "What are common reversible causes of acute confusion in an elderly patient?",
}


def _normalize_row(raw: dict[str, str | None]) -> dict[str, str]:
    return {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}


def _row_question(norm: dict[str, str]) -> str | None:
    for key in ("question", "query", "prompt", "text"):
        if norm.get(key):
            return norm[key]
    return None


def _row_slug(norm: dict[str, str]) -> str | None:
    for key in ("specialist", "dataset_slug", "dataset", "subject_slug"):
        v = norm.get(key)
        if v:
            return v.strip()
    return None


def _load_csv_tasks(path: str, allowed: set[str]) -> list[tuple[str, str | None]]:
    tasks: list[tuple[str, str | None]] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header row in {path}")
        for raw in reader:
            norm = _normalize_row({k: raw.get(k) for k in raw})
            q = _row_question(norm)
            if not q:
                skipped += 1
                continue
            slug = _row_slug(norm)
            if slug and slug not in allowed:
                skipped += 1
                continue
            tasks.append((q, slug))
    if skipped:
        print(f"[benchmark] skipped {skipped} rows (empty question or specialist not in tier)", file=sys.stderr)
    return tasks


def _inline_tasks(tier: int) -> list[tuple[str, str | None]]:
    keys = tier_keys(tier)
    out: list[tuple[str, str | None]] = []
    for k in keys:
        p = _INLINE_PROMPTS.get(k)
        if p:
            out.append((p, k))
    return out


def _stats_ms(samples: list[float]) -> str:
    if not samples:
        return "no samples"
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    mid = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return f"n={n} mean_ms={mean:.1f} min_ms={s[0]:.1f} max_ms={s[-1]:.1f} median_ms={mid:.1f}"


async def _run_benchmark(
    tier: int,
    csv_path: str | None,
    repeats: int,
    warmup: int,
) -> None:
    allowed = set(tier_keys(tier))
    path = csv_path
    if path is None:
        path = default_csv_path(tier)

    if path and os.path.isfile(path):
        print(f"[benchmark] tier={tier} csv={path}", file=sys.stderr)
        tasks = _load_csv_tasks(path, allowed)
    else:
        if csv_path:
            raise FileNotFoundError(f"CSV not found: {path}")
        print(f"[benchmark] tier={tier} no CSV at {path}, using inline prompts", file=sys.stderr)
        tasks = _inline_tasks(tier)

    if not tasks:
        raise SystemExit("No tasks to run (empty CSV or preset).")

    uid_base = str(uuid.uuid4())
    times: list[float] = []

    # Warmup (discarded)
    for w in range(warmup):
        q, _ = tasks[w % len(tasks)]
        await run_agent(user_id=f"{uid_base}-warmup", query=q, tier=tier)

    total_runs = max(1, repeats) * len(tasks)
    i = 0
    for _ in range(max(1, repeats)):
        for q, _ in tasks:
            t0 = time.perf_counter()
            await run_agent(user_id=f"{uid_base}-{i}", query=q, tier=tier)
            dt_ms = (time.perf_counter() - t0) * 1000
            times.append(dt_ms)
            i += 1

    print(_stats_ms(times))


def main() -> None:
    p = argparse.ArgumentParser(description="Scalability benchmark for agents/testing coordinators.")
    p.add_argument("--tier", type=int, required=True, choices=[3, 5, 8], help="Number of services (preset tier).")
    p.add_argument("--csv", type=str, default=None, help="Override CSV path (default: from presets / data/).")
    p.add_argument("--runs", type=int, default=1, help="Repeat full task list this many times (default 1).")
    p.add_argument("--warmup", type=int, default=0, help="Warmup iterations before timing (default 0).")
    args = p.parse_args()
    asyncio.run(
        _run_benchmark(
            tier=args.tier,
            csv_path=args.csv,
            repeats=args.runs,
            warmup=args.warmup,
        )
    )


if __name__ == "__main__":
    main()
