# ruff: noqa: E402
import argparse
import os
import re
import sys
from getpass import getpass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.semantic_router.src.config import build_config
from experiments.semantic_router.src.dataset_utils import (
    load_known_services_from_descriptions,
    load_prompts_dataset,
    redirect_unknown_services_to_general_qa,
)
from experiments.semantic_router.src.embedder import build_embedder
from experiments.semantic_router.src.evaluator import Evaluator
from experiments.semantic_router.src.index import ServiceIndex
from experiments.semantic_router.src.router import SemanticRouter


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding-only semantic router benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["miniLM", "bge", "openai"],
        help="Embedding model alias: miniLM | bge | openai",
    )
    return parser.parse_args()


def _plot_model_accuracy(metrics_path: str, output_path: str):
    metrics_df = pd.read_csv(metrics_path)
    latest_by_model = metrics_df.groupby("model", as_index=False).last()
    latest_by_model = latest_by_model.sort_values("accuracy", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(latest_by_model["model"], latest_by_model["accuracy"])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Embedding Model")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Model Accuracy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_per_class_accuracy(per_class_history_path: str, output_path: str):
    history = pd.read_csv(per_class_history_path)
    pivot = history.pivot_table(index="service", columns="model", values="accuracy", aggfunc="last")
    pivot = pivot.sort_index()

    n_services = len(pivot.index)
    n_models = len(pivot.columns)
    x = np.arange(n_services)
    width = 0.8 / max(1, n_models)

    plt.figure(figsize=(10, 6))
    for idx, model_name in enumerate(pivot.columns):
        offset = (idx - (n_models - 1) / 2.0) * width
        values = pivot[model_name].fillna(0.0).to_numpy()
        plt.bar(x + offset, values, width=width, label=model_name)

    plt.xticks(x, pivot.index, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Service Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_confusion_matrix(predictions_df: pd.DataFrame, output_path: str):
    labels = sorted(
        set(predictions_df["expected_service"].astype(str).tolist())
        | set(predictions_df["predicted_service"].astype(str).tolist())
    )
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for _, row in predictions_df.iterrows():
        i = index[str(row["expected_service"])]
        j = index[str(row["predicted_service"])]
        matrix[i, j] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _request_openai_api_key():
    user_key = getpass("Enter your OpenAI API key: ").strip()
    if not user_key:
        raise ValueError("OPENAI_API_KEY is required for --model openai.")
    os.environ["OPENAI_API_KEY"] = user_key


def _is_openai_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "authenticationerror" in message or "invalid_api_key" in message or "error code: 401" in message


def _print_run_summary(model_name: str, model_run_rows):
    print("Evaluation complete.")
    print(f"Model: {model_name}")
    for row in model_run_rows:
        print(f"Dataset: {row['dataset']}")
        print(f"Total Samples: {row['total_samples']}")
        print(f"Accuracy: {row['accuracy']:.2f}")
        print(f"Top-3 Accuracy: {row['top3_accuracy']:.2f}")
        print(f"Best Performing Class: {row['best_class']} ({row['best_class_accuracy']:.2f})")
        print(f"Worst Performing Class: {row['worst_class']} ({row['worst_class_accuracy']:.2f})")
        print("")
    print("Outputs saved to:")


def _discover_dataset_filenames(datasets_dir: str):
    filenames = []
    for file_name in os.listdir(datasets_dir):
        if re.fullmatch(r"dataset_\d+_services\.csv", file_name):
            filenames.append(file_name)

    def _service_count_key(name: str):
        matched = re.search(r"dataset_(\d+)_services\.csv", name)
        return int(matched.group(1)) if matched else 10**9

    return sorted(filenames, key=_service_count_key)


def main():
    # Parse CLI arguments and construct the benchmark configuration.
    args = parse_args()
    config = build_config(root_dir=REPO_ROOT, model_alias=args.model)

    needs_key = args.model == "openai" and not os.getenv("OPENAI_API_KEY")
    if needs_key:
        _request_openai_api_key()

    os.makedirs(config.model_outputs_dir, exist_ok=True)

    dataset_filenames = _discover_dataset_filenames(os.path.join(REPO_ROOT, "datasets"))
    if not dataset_filenames:
        raise ValueError("No dataset files found. Expected pattern: datasets/dataset_<N>_services.csv")

    model_run_rows = []

    for dataset_filename in dataset_filenames:
        dataset_config = build_config(
            root_dir=REPO_ROOT,
            model_alias=args.model,
            dataset_filename=dataset_filename,
        )

        prompts_df = load_prompts_dataset(dataset_config.dataset_prompts_path)
        known_services = load_known_services_from_descriptions(dataset_config.dataset_descriptions_path)
        prompts_df = redirect_unknown_services_to_general_qa(prompts_df, known_services)
        services = sorted(prompts_df["expected_service"].astype(str).unique().tolist())

        retry_count = 0
        max_retries = 3
        while True:
            try:
                embedder = build_embedder(args.model)
                service_index = ServiceIndex(config=dataset_config, embedder=embedder)
                service_index.build(services=services)
                router = SemanticRouter(service_index=service_index, embedder=embedder)
                evaluator = Evaluator(config=dataset_config, router=router)
                eval_result = evaluator.run()
                break
            except Exception as exc:
                if args.model == "openai" and _is_openai_auth_error(exc) and retry_count < max_retries:
                    retry_count += 1
                    print("OpenAI API key authentication failed. Please re-enter your key.")
                    _request_openai_api_key()
                    continue
                raise

        evaluator.persist_outputs(model_name=dataset_config.resolved_model_name, eval_result=eval_result)

        _plot_model_accuracy(dataset_config.metrics_summary_path, dataset_config.model_accuracy_plot_path)
        _plot_per_class_accuracy(dataset_config.per_class_history_path, dataset_config.per_class_accuracy_plot_path)
        _plot_confusion_matrix(eval_result["predictions_df"], dataset_config.confusion_matrix_plot_path)

        per_class_accuracy = eval_result["per_class_accuracy"]
        best_class, best_score = max(per_class_accuracy.items(), key=lambda item: item[1])
        worst_class, worst_score = min(per_class_accuracy.items(), key=lambda item: item[1])

        model_run_rows.append(
            {
                "model": dataset_config.resolved_model_name,
                "dataset": dataset_config.dataset_name,
                "accuracy": round(eval_result["accuracy"], 6),
                "top3_accuracy": round(eval_result["top3_accuracy"], 6),
                "total_samples": int(eval_result["total_samples"]),
                "best_class": best_class,
                "best_class_accuracy": round(best_score, 6),
                "worst_class": worst_class,
                "worst_class_accuracy": round(worst_score, 6),
                "dataset_output_dir": dataset_config.dataset_outputs_dir,
            }
        )

    model_run_df = pd.DataFrame(model_run_rows)
    model_run_df.to_csv(config.model_run_summary_path, index=False)

    _print_run_summary(config.resolved_model_name, model_run_rows)
    print(f"- {config.model_outputs_dir}")


if __name__ == "__main__":
    main()
