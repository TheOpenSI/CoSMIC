# Routing evaluation: confusion matrices and metrics for scalability tiers 3/5/8/10.

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import os
import sys
import uuid
import re
from datetime import datetime
from typing import Sequence, Any

import pandas as pd
from sklearn.metrics import confusion_matrix

_log = logging.getLogger(__name__)


def _configure_benchmark_logging(verbose: bool) -> None:
    # Benchmark progress goes to stderr; stdout stays metrics TSV.
    pkg_level = logging.DEBUG if verbose else logging.INFO

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setFormatter(fmt)

    pkg = logging.getLogger("agents.testing")
    pkg.handlers.clear()
    pkg.addHandler(stderr_h)
    pkg.setLevel(pkg_level)
    pkg.propagate = False

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    noisy = ("httpx", "httpcore", "google", "google_genai", "google.auth", "urllib3")
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_TESTING_DIR, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agents.deprecation_filters import apply_known_deprecation_filters

apply_known_deprecation_filters()

from agents.testing.coordinator import (
    default_csv_path,
    tier_keys,
    tier_prediction_labels,
)
from agents.testing.runner import RoutingOutcome, clear_runner_cache, run_agent_routing

_DEFAULT_RESULTS_DIR = os.path.normpath(os.path.join(_TESTING_DIR, "results"))


def _normalize_row(raw: dict[str, str | None]) -> dict[str, str]:
    return {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}


def _row_question(norm: dict[str, str]) -> str | None:
    for key in ("question", "query", "prompt", "text"):
        if norm.get(key):
            return norm[key]
    return None


def _row_service(norm: dict[str, str]) -> str | None:
    for key in ("service", "specialist", "dataset_slug", "dataset", "subject_slug", "expected_service"):
        v = norm.get(key)
        if v:
            return v.strip()
    return None


def _load_csv_eval_rows(path: str, allowed: set[str]) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header row in {path}")
        for raw in reader:
            norm = _normalize_row({k: raw.get(k) for k in raw})
            q = _row_question(norm)
            svc = _row_service(norm)
            if not q:
                skipped += 1
                continue
            if not svc:
                skipped += 1
                continue
            if svc not in allowed:
                skipped += 1
                continue
            tasks.append((q, svc))
    if skipped:
        _log.info("skipped %s rows (empty question/service or service not in tier)", skipped)
    return tasks


def _percentile_p95(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * 0.95
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _calculate_metrics(
    y_true: list[str],
    y_pred: list[str],
    latencies: list[float],
    prompt_tokens: list[int],
    completion_tokens: list[int],
    routing_failures: int,
    classes: list[str],
    run_id: str,
) -> dict[str, Any]:
    total_samples = len(y_true)
    accuracy = sum(1 for gt, pd_ in zip(y_true, y_pred) if gt == pd_) / total_samples if total_samples > 0 else 0.0
    lat_avg = sum(latencies) / total_samples if total_samples > 0 else 0.0
    lat_p95 = _percentile_p95(latencies)
    total_prompt_tokens = sum(prompt_tokens)
    total_completion_tokens = sum(completion_tokens)

    # Per-class accuracy for best/worst
    per_class_acc = {}
    for cls in classes:
        indices = [i for i, val in enumerate(y_true) if val == cls]
        if indices:
            correct = sum(1 for i in indices if y_pred[i] == cls)
            per_class_acc[cls] = correct / len(indices)

    best_class_name = ""
    best_class_acc = 0.0
    worst_class_name = ""
    worst_class_acc = 1.0

    if per_class_acc:
        sorted_acc = sorted(per_class_acc.items(), key=lambda x: (x[1], x[0]), reverse=True)
        best_class_name, best_class_acc = sorted_acc[0]
        worst_class_name, worst_class_acc = sorted_acc[-1]

    return {
        "run_id": run_id,
        "total_samples": total_samples,
        "accuracy": accuracy,
        "latency_avg_ms": lat_avg,
        "latency_p95_ms": lat_p95,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "estimated_cost": 0.0,  # Placeholder as per requirement
        "routing_failures": routing_failures,
        "best_class": f"{best_class_name} ({best_class_acc:.4f})",
        "worst_class": f"{worst_class_name} ({worst_class_acc:.4f})",
    }


def _log_iteration(
    run_id: str,
    iteration: int,
    total: int,
    tier: int,
    question: str,
    gold: str,
    predicted: str,
    raw: str,
    success: bool,
    error_type: str | None,
    latency: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    status = "Success" if success else f"Failure - {error_type}"
    q_trunc = (question.replace("\n", " ")[:77] + "...") if len(question) > 80 else question.replace("\n", " ")
    
    print("-" * 80)
    if iteration == 1:
        print(f"Run ID: {run_id}")
    
    print(f"Iteration: {iteration}/{total}")
    print(f"Tier: {tier}")
    print(f"Question: {q_trunc}")
    print(f"Gold Target: {gold}")
    print(f"Predicted Service: {predicted}")
    print(f"Raw Router Output: {raw}")
    print(f"Status: {status}")
    print(f"Latency: {latency:.2f}ms")
    print(f"Tokens: Prompt {prompt_tokens} / Completion {completion_tokens}")


async def _eval_tier(
    tier: int,
    csv_path: str | None,
    out_dir: str,
    run_id: str,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_allowed = set(tier_keys(tier))
    valid_predictions = frozenset(tier_prediction_labels(tier))
    path = csv_path if csv_path else default_csv_path(tier)
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found for tier {tier}: {path}")

    rows = _load_csv_eval_rows(path, gold_allowed)
    if not rows:
        raise SystemExit(f"No evaluation rows for tier {tier} (empty or invalid CSV).")

    clear_runner_cache()

    evaluations = []
    y_true = []
    y_pred = []
    latencies = []
    prompt_tokens_list = []
    completion_tokens_list = []
    routing_failures = 0

    total_rows = len(rows)
    uid_base = str(uuid.uuid4())
    per_sample_path = os.path.join(out_dir, "per_sample_evaluations.csv")

    for i, (q, gold) in enumerate(rows):
        timestamp = datetime.now().isoformat()
        outcome: RoutingOutcome = await run_agent_routing(
            user_id=f"{uid_base}-{i}",
            query=q,
            tier=tier,
            verbose=verbose,
        )
        
        predicted_service = outcome.predicted_service
        raw_output = predicted_service or ""
        error_type = None

        # Fallback: if no structured predicted_service, try to parse from text
        if not predicted_service and "transfer_to_agent" in outcome.final_text:
            raw_output = outcome.final_text.strip()
            match = re.search(r"transfer_to_agent\(['\"]([^'\"]+)['\"]\)", outcome.final_text)
            if match:
                predicted_service = match.group(1)
                _log.info("Salvaged hallucinated text transfer: %s", predicted_service)

        predicted = None
        
        # Validation Logic
        if predicted_service is None:
            error_type = "routing_failure"
            routing_failures += 1
            predicted = "__no_transfer__"
        elif predicted_service not in valid_predictions:
            error_type = "parsing_error"
            predicted = "__no_transfer__"
        else:
            predicted = predicted_service
            
        if outcome.aborted_reason and "timeout" in outcome.aborted_reason.lower():
            error_type = "timeout"

        correct = (predicted == gold)
        
        eval_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "row_index": i + 1,
            "dataset": os.path.basename(path),
            "tier": tier,
            "question": q,
            "gold": gold,
            "predicted_service_raw": raw_output,
            "predicted": predicted,
            "correct": correct,
            "response": outcome.final_text,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "latency_ms": outcome.latency_ms,
            "error_type": error_type,
        }
        evaluations.append(eval_data)
        
        # Append as you go
        df_row = pd.DataFrame([eval_data])
        df_row.to_csv(
            per_sample_path, 
            mode='a', 
            header=not os.path.exists(per_sample_path), 
            index=False
        )

        y_true.append(gold)
        y_pred.append(predicted)
        latencies.append(outcome.latency_ms)
        prompt_tokens_list.append(outcome.prompt_tokens)
        completion_tokens_list.append(outcome.completion_tokens)
        
        _log_iteration(
            run_id=run_id,
            iteration=i + 1,
            total=total_rows,
            tier=tier,
            question=q,
            gold=gold,
            predicted=predicted,
            raw=raw_output,
            success=correct,
            error_type=error_type,
            latency=outcome.latency_ms,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens
        )

    metrics = _calculate_metrics(
        y_true, y_pred, latencies, prompt_tokens_list, completion_tokens_list, 
        routing_failures, list(gold_allowed), run_id
    )
    
    return evaluations, metrics


async def _main_async(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    run_id = str(uuid.uuid4())
    
    os.makedirs(out_dir, exist_ok=True)
    per_sample_path = os.path.join(out_dir, "per_sample_evaluations.csv")
    summary_path = os.path.join(out_dir, "benchmark_summary_metrics.csv")
    
    if os.path.exists(per_sample_path):
        os.remove(per_sample_path)
    if os.path.exists(summary_path):
        os.remove(summary_path)
    
    print("-" * 80)
    print(f"Benchmark starting | Run ID: {run_id}")
    print(f"Destination Output Directory: {out_dir}")
    print("-" * 80)

    if args.all:
        tiers = [3, 5, 8, 10]
    else:
        tiers = [args.tier]

    _log.info("Starting benchmark | Run ID: %s | Tiers: %s", run_id, tiers)

    all_evaluations = []
    all_summaries = []
    for tier in tiers:
        csv_override = None if args.all else args.csv
        evals, metrics = await _eval_tier(tier, csv_override, out_dir, run_id, args.verbose)
        all_evaluations.extend(evals)
        all_summaries.append(metrics)

    # Save CSVs (summaries)
    os.makedirs(out_dir, exist_ok=True)
    
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(out_dir, "benchmark_summary_metrics.csv")
    
    summary_df.to_csv(summary_path, index=False)
    
    _log.info("Saved per-sample evaluations to %s", per_sample_path)
    _log.info("Saved summary metrics to %s", summary_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Refactored Routing Benchmark for OpenSI-CoSMIC.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tier", type=int, choices=[3, 5, 8, 10], help="Number of registered services (preset tier).")
    g.add_argument("--all", action="store_true", help="Run evaluation for tiers 3, 5, 8, and 10.")
    p.add_argument("--csv", type=str, default=None, help="Override CSV path (only with --tier).")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each ADK event during each sample.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=_DEFAULT_RESULTS_DIR,
        help=f"Directory for CSV outputs (default: {_DEFAULT_RESULTS_DIR}).",
    )
    args = p.parse_args()
    
    if args.csv and not args.all and not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
        
    _configure_benchmark_logging(verbose=args.verbose)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
