"""Routing evaluation: confusion matrices and metrics for scalability tiers 3/5/8/10."""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
import uuid
from typing import Sequence

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_TESTING_DIR, "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_DEFAULT_RESULTS_DIR = os.path.normpath(os.path.join(_TESTING_DIR, "results"))

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from agents.testing.coordinator import default_csv_path, tier_keys
from agents.testing.runner import RoutingOutcome, clear_runner_cache, run_agent_routing


_SENTINEL_NO_ROUTE = "__no_transfer__"


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
    """Load (question, gold service) rows for evaluation."""
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
        print(
            f"[benchmark] skipped {skipped} rows (empty question/service or service not in tier)",
            file=sys.stderr,
        )
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


def _per_class_accuracy(y_true: list[str], y_pred: list[str], classes: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in classes:
        gold_idx = [i for i, y in enumerate(y_true) if y == cls]
        if not gold_idx:
            continue
        ok = sum(1 for i in gold_idx if y_pred[i] == cls)
        out[cls] = ok / len(gold_idx)
    return out


def _best_worst_class(per_class: dict[str, float]) -> tuple[str, float, str, float]:
    if not per_class:
        return ("", float("nan"), "", float("nan"))
    items = sorted(per_class.items(), key=lambda x: (x[1], x[0]), reverse=True)
    best_name, best_acc = items[0]
    items_w = sorted(per_class.items(), key=lambda x: (x[1], x[0]))
    worst_name, worst_acc = items_w[0]
    return (best_name, best_acc, worst_name, worst_acc)


def _format_cm_text(labels: list[str], cm: list[list[int]]) -> str:
    label_w = max(len(L) for L in labels) if labels else 8
    lines = []
    header = " " * (label_w + 1) + " pred→"
    lines.append(header)
    lines.append(" " * label_w + " " + "".join(str(i % 10) for i in range(len(labels))))
    for i, row_name in enumerate(labels):
        row = " ".join(str(v) for v in cm[i])
        lines.append(f"{row_name:<{label_w}} {row}")
    return "\n".join(lines)


async def _eval_tier(
    tier: int,
    csv_path: str | None,
    out_dir: str | None,
) -> dict[str, object]:
    allowed = set(tier_keys(tier))
    path = csv_path if csv_path else default_csv_path(tier)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found for tier {tier}: {path}")

    rows = _load_csv_eval_rows(path, allowed)
    if not rows:
        raise SystemExit(f"No evaluation rows for tier {tier} (empty or invalid CSV).")

    clear_runner_cache()
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    tokens_per_row: list[int] = []
    routing_failures = 0

    uid_base = str(uuid.uuid4())
    for i, (q, gold) in enumerate(rows):
        outcome: RoutingOutcome = await run_agent_routing(
            user_id=f"{uid_base}-{i}",
            query=q,
            tier=tier,
        )
        pred = outcome.predicted_service
        if pred is None:
            routing_failures += 1
            pred_label = _SENTINEL_NO_ROUTE
        elif pred not in allowed:
            pred_label = _SENTINEL_NO_ROUTE
        else:
            pred_label = pred
        y_true.append(gold)
        y_pred.append(pred_label)
        latencies.append(outcome.latency_ms)
        tokens_per_row.append(outcome.tokens_total)

    labels_order = list(tier_keys(tier))
    matrix_labels = labels_order + [_SENTINEL_NO_ROUTE]

    cm = confusion_matrix(y_true, y_pred, labels=matrix_labels)

    correct = sum(1 for g, p in zip(y_true, y_pred) if g == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    pca = _per_class_accuracy(y_true, y_pred, labels_order)
    best_c, best_a, worst_c, worst_a = _best_worst_class(pca)

    lat_avg = sum(latencies) / len(latencies) if latencies else float("nan")
    lat_p95 = _percentile_p95(latencies)
    tok_sum = sum(tokens_per_row)

    dataset_id = os.path.basename(path)

    print(f"\n=== Tier {tier} | dataset={dataset_id} | n={len(y_true)} ===", file=sys.stderr)
    print(_format_cm_text(matrix_labels, cm.tolist()), file=sys.stderr)

    if out_dir and plt is not None:
        os.makedirs(out_dir, exist_ok=True)
        safe = f"confusion_tier{tier}.png"
        fig_path = os.path.join(out_dir, safe)
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(matrix_labels)), max(6, 0.4 * len(matrix_labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=matrix_labels)
        disp.plot(ax=ax, xticks_rotation=75, colorbar=False, include_values=True, values_format="d")
        ax.set_title(f"Routing confusion (tier {tier})")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[benchmark] saved {fig_path}", file=sys.stderr)
    elif out_dir and plt is None:
        print("[benchmark] matplotlib not installed; skipping PNG export", file=sys.stderr)

    return {
        "dataset": dataset_id,
        "accuracy": accuracy,
        "total_samples": len(y_true),
        "latency_avg_ms": lat_avg,
        "latency_p95_ms": lat_p95,
        "best_class": best_c,
        "best_class_accuracy": best_a,
        "worst_class": worst_c,
        "worst_class_accuracy": worst_a,
        "token_cost": tok_sum,
        "routing_failures": routing_failures,
    }


_METRIC_COLS = [
    "dataset",
    "accuracy",
    "total_samples",
    "latency_avg_ms",
    "latency_p95_ms",
    "best_class",
    "best_class_accuracy",
    "worst_class",
    "worst_class_accuracy",
    "token_cost",
    "routing_failures",
]


def _cell_str(key: str, val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        if key in ("accuracy", "best_class_accuracy", "worst_class_accuracy"):
            return f"{val:.6f}".rstrip("0").rstrip(".")
        if key in ("latency_avg_ms", "latency_p95_ms"):
            return f"{val:.2f}"
    return str(val)


def _write_metrics_md(path: str, rows: list[dict[str, object]]) -> None:
    """Write one Markdown table with all metric rows."""
    header = "| " + " | ".join(_METRIC_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _METRIC_COLS) + " |"
    lines = ["# Routing benchmark metrics", "", header, sep]
    for r in rows:
        cells = [_cell_str(k, r.get(k)) for k in _METRIC_COLS]
        escaped = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_metrics_table(rows: list[dict[str, object]]) -> None:
    header = "\t".join(_METRIC_COLS)
    print(header)
    for r in rows:
        print("\t".join(_cell_str(c, r.get(c)) for c in _METRIC_COLS))


async def _main_async(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        tiers = [3, 5, 8, 10]
        if args.csv:
            print("[benchmark] ignoring --csv when using --all (per-tier defaults from presets)", file=sys.stderr)
    else:
        tiers = [args.tier]

    results: list[dict[str, object]] = []
    for tier in tiers:
        csv_override = None if args.all else args.csv
        row = await _eval_tier(tier, csv_override, out_dir)
        results.append(row)

    metrics_path = os.path.join(out_dir, "metrics.md")
    _write_metrics_md(metrics_path, results)
    print(f"[benchmark] metrics written to {metrics_path}", file=sys.stderr)

    _print_metrics_table(results)


def main() -> None:
    p = argparse.ArgumentParser(description="Routing benchmark: confusion matrix and metrics per tier.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tier", type=int, choices=[3, 5, 8, 10], help="Number of registered services (preset tier).")
    g.add_argument("--all", action="store_true", help="Run evaluation for tiers 3, 5, 8, and 10.")
    p.add_argument("--csv", type=str, default=None, help="Override CSV path (only with --tier).")
    p.add_argument(
        "--out-dir",
        type=str,
        default=_DEFAULT_RESULTS_DIR,
        help=f"Directory for metrics.md and confusion PNGs (default: {_DEFAULT_RESULTS_DIR}).",
    )
    args = p.parse_args()
    if args.csv and not args.all and not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
