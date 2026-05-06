"""
Unified CLI for running retrieval-only service routing benchmarks.

This script evaluates a single routing method against a single dataset and writes:
- predictions.csv and metrics_summary.csv under method/dataset output folders
- a small set of reproducible plots under outputs/visualizations
"""

# ruff: noqa: E402
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.retrieval_benchmark_suite.bi_encoder.run import run_bi_encoder
from experiments.retrieval_benchmark_suite.colbert.run import run_colbert
from experiments.retrieval_benchmark_suite.cross_encoder.run import run_cross_encoder
from experiments.retrieval_benchmark_suite.shared.data_loader import (
    load_prompts_dataset,
    load_service_descriptions,
)
from experiments.retrieval_benchmark_suite.shared.metrics import collect_method_metrics, write_outputs
from experiments.retrieval_benchmark_suite.shared.utils import canonical_dataset_paths, dataset_name_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval benchmark suite for service routing.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["bi-encoder", "cross-encoder", "colbert"],
        help="Routing method alias: bi-encoder | cross-encoder | colbert",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset CSV with prompt and expected_service columns.",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default="datasets/dataset_descriptions.xlsx",
        help="Path to service descriptions spreadsheet.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution device.",
    )
    return parser.parse_args()


def _plot_confusion_matrix(predictions_df: pd.DataFrame, output_path: str):
    labels = sorted(
        set(predictions_df["expected_service"].astype(str).tolist())
        | set(predictions_df["predicted_service"].astype(str).tolist())
    )
    matrix = pd.crosstab(
        predictions_df["expected_service"],
        predictions_df["predicted_service"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_per_class_accuracy(predictions_df: pd.DataFrame, output_path: str):
    per_class_df = (
        predictions_df.groupby("expected_service", as_index=False)["is_correct"]
        .mean()
        .rename(columns={"expected_service": "service", "is_correct": "accuracy"})
        .sort_values("service")
    )

    plt.figure(figsize=(12, 5))
    sns.barplot(data=per_class_df, x="service", y="accuracy", color="#4C72B0")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Service Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_accuracy_comparison(metrics_df: pd.DataFrame, output_path: str):
    if len(metrics_df) == 0:
        return
    ordered = metrics_df.sort_values("accuracy", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="method", y="accuracy", palette="deep")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Method")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_latency_comparison(metrics_df: pd.DataFrame, output_path: str):
    if len(metrics_df) == 0:
        return
    ordered = metrics_df.sort_values("latency_avg_ms", ascending=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="method", y="latency_avg_ms", palette="muted")
    plt.xlabel("Method")
    plt.ylabel("Average Latency (ms)")
    plt.title("Latency Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _resolve_model_name(method: str) -> str:
    mapping = {
        "bi-encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "cross-encoder": "cross-encoder/stsb-distilroberta-base",
        "colbert": "colbert-ir/colbertv2.0",
    }
    return mapping[method]


def _run_method(method: str, prompts_df: pd.DataFrame, descriptions: dict, dataset_name: str, device: str):
    if method == "bi-encoder":
        return run_bi_encoder(
            prompts_df=prompts_df,
            service_descriptions=descriptions,
            dataset_name=dataset_name,
            model_name=_resolve_model_name(method),
            device=device,
        )
    if method == "cross-encoder":
        return run_cross_encoder(
            prompts_df=prompts_df,
            service_descriptions=descriptions,
            dataset_name=dataset_name,
            model_name=_resolve_model_name(method),
            device=device,
        )
    if method == "colbert":
        return run_colbert(
            prompts_df=prompts_df,
            service_descriptions=descriptions,
            dataset_name=dataset_name,
            model_name=_resolve_model_name(method),
            device=device,
        )
    raise ValueError(f"Unsupported method: {method}")


def main():
    # Parse CLI arguments and resolve paths relative to the repo root.
    args = parse_args()
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(REPO_ROOT, args.dataset)
    descriptions_path = (
        args.descriptions if os.path.isabs(args.descriptions) else os.path.join(REPO_ROOT, args.descriptions)
    )
    if not Path(dataset_path).exists():
        raise ValueError(f"Dataset file not found: {dataset_path}")
    if dataset_path not in canonical_dataset_paths(REPO_ROOT):
        print("[warn] dataset is outside the canonical benchmark set; proceeding anyway.", flush=True)

    # Load dataset prompts and service descriptions (routing targets).
    dataset_name = dataset_name_from_path(dataset_path)
    prompts_df = load_prompts_dataset(dataset_path)
    service_descriptions = load_service_descriptions(descriptions_path)
    known_services = sorted(prompts_df["expected_service"].astype(str).unique().tolist())
    service_descriptions = {k: v for k, v in service_descriptions.items() if k in known_services}
    if not service_descriptions:
        raise ValueError("No service descriptions matched expected services for this dataset.")

    # Run evaluation for the selected routing method.
    model_name = _resolve_model_name(args.method)
    eval_result = _run_method(args.method, prompts_df, service_descriptions, dataset_name, args.device)

    # Persist predictions and summary metrics for this run.
    outputs_root = os.path.join(MODULE_DIR, "outputs")
    dataset_output_dir = os.path.join(outputs_root, args.method, dataset_name)
    write_outputs(
        output_dir=dataset_output_dir,
        method=args.method,
        model_name=model_name,
        dataset_name=dataset_name,
        eval_result=eval_result,
    )

    # Generate plots for this run and update cross-method comparison plots (if available).
    visualizations_dir = os.path.join(outputs_root, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    predictions_df = eval_result["predictions_df"]
    _plot_confusion_matrix(predictions_df, os.path.join(visualizations_dir, f"{args.method}_{dataset_name}_confusion_matrix.png"))
    _plot_per_class_accuracy(
        predictions_df, os.path.join(visualizations_dir, f"{args.method}_{dataset_name}_per_class_accuracy.png")
    )

    metrics_df = collect_method_metrics(outputs_root=outputs_root, dataset_name=dataset_name)
    _plot_accuracy_comparison(metrics_df, os.path.join(visualizations_dir, f"{dataset_name}_model_accuracy.png"))
    _plot_latency_comparison(metrics_df, os.path.join(visualizations_dir, f"{dataset_name}_latency_comparison.png"))

    row = {
        "accuracy": eval_result["accuracy"],
        "top3_accuracy": eval_result["top3_accuracy"],
        "total_samples": eval_result["total_samples"],
        "best_class": eval_result["best_class"],
        "best_class_accuracy": eval_result["best_class_accuracy"],
        "worst_class": eval_result["worst_class"],
        "worst_class_accuracy": eval_result["worst_class_accuracy"],
        "latency_avg_ms": eval_result["latency_avg_ms"],
        "latency_p95_ms": eval_result["latency_p95_ms"],
    }

    # Print a short, human-readable summary to stdout.
    print("Evaluation complete.")
    print(f"Method: {args.method}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Total Samples: {int(row['total_samples'])}")
    print(f"Accuracy: {row['accuracy']:.2f}")
    print(f"Top-3 Accuracy: {row['top3_accuracy']:.2f}")
    print(f"Average Latency (ms): {row['latency_avg_ms']:.2f}")
    print(f"P95 Latency (ms): {row['latency_p95_ms']:.2f}")
    print(f"Best Performing Class: {row['best_class']} ({row['best_class_accuracy']:.2f})")
    print(f"Worst Performing Class: {row['worst_class']} ({row['worst_class_accuracy']:.2f})")
    print("Outputs saved to:")
    print(f"- {dataset_output_dir}")
    print(f"- {visualizations_dir}")


if __name__ == "__main__":
    main()
