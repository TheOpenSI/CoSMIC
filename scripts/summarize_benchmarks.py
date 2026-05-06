import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate benchmark metrics into one summary CSV.")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Root experiments directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/benchmark_summary.csv",
        help="Output CSV path for merged summary.",
    )
    return parser.parse_args()


def _module_from_path(path: Path) -> str:
    parts = path.parts
    if "semantic_router" in parts:
        return "semantic_router"
    if "retrieval_benchmark_suite" in parts:
        return "retrieval_benchmark_suite"
    if "tool_orchestra_benchmark" in parts:
        return "tool_orchestra_benchmark"
    return "unknown"


def _normalize_row(raw_row: Dict, metrics_path: Path) -> Dict:
    row = {str(k): raw_row[k] for k in raw_row}
    module = _module_from_path(metrics_path)
    method = str(row.get("method", "")).strip()
    model = str(row.get("model", "")).strip()
    dataset = str(row.get("dataset", "")).strip()

    if module == "semantic_router":
        benchmark = "semantic_router"
        runner = "embedding-router"
        if not method:
            method = "embedding-similarity"
    elif module == "retrieval_benchmark_suite":
        benchmark = "retrieval_suite"
        runner = method or "retrieval"
    elif module == "tool_orchestra_benchmark":
        benchmark = "tool_orchestra_benchmark"
        runner = "tool-router"
    else:
        benchmark = "unknown"
        runner = method or "unknown"

    return {
        "benchmark": benchmark,
        "runner": runner,
        "method": method,
        "model": model,
        "dataset": dataset,
        "accuracy": row.get("accuracy"),
        "top3_accuracy": row.get("top3_accuracy"),
        "total_samples": row.get("total_samples"),
        "latency_avg_ms": row.get("latency_avg_ms"),
        "latency_p95_ms": row.get("latency_p95_ms"),
        "best_class": row.get("best_class"),
        "best_class_accuracy": row.get("best_class_accuracy"),
        "worst_class": row.get("worst_class"),
        "worst_class_accuracy": row.get("worst_class_accuracy"),
        "source_file": str(metrics_path),
    }


def build_summary(experiments_dir: Path) -> pd.DataFrame:
    metrics_files: List[Path] = sorted(experiments_dir.glob("**/metrics_summary.csv"))
    rows: List[Dict] = []

    for metrics_file in metrics_files:
        try:
            df = pd.read_csv(metrics_file)
        except Exception:
            continue
        if len(df) == 0:
            continue
        rows.extend(_normalize_row(raw_row.to_dict(), metrics_file) for _, raw_row in df.iterrows())

    if not rows:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "runner",
                "method",
                "model",
                "dataset",
                "accuracy",
                "top3_accuracy",
                "total_samples",
                "latency_avg_ms",
                "latency_p95_ms",
                "best_class",
                "best_class_accuracy",
                "worst_class",
                "worst_class_accuracy",
                "source_file",
            ]
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        by=["benchmark", "runner", "dataset", "accuracy"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return out_df


def main():
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    output_path = Path(args.output)

    summary_df = build_summary(experiments_dir=experiments_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(f"rows={len(summary_df)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
