# ruff: noqa: E402
import argparse
import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.tool_orchestra_benchmark.src.config import build_config
from experiments.tool_orchestra_benchmark.src.dataset_utils import (
    load_prompts_dataset,
    load_service_descriptions,
    normalize_service_name,
)
from experiments.tool_orchestra_benchmark.src.evaluator import Evaluator
from experiments.tool_orchestra_benchmark.src.models import build_router
from experiments.tool_orchestra_benchmark.src.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ToolOrchestra benchmark for service routing.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["openai", "toolorchestra"],
        help="Routing model alias: openai | toolorchestra",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/dataset_3_services.csv",
        help="Path to dataset CSV with prompt and expected_service columns.",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default="datasets/dataset_descriptions.xlsx",
        help="Optional path to service descriptions spreadsheet.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device used for toolorchestra model.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write checkpoint outputs every N samples (0 disables checkpointing).",
    )
    return parser.parse_args()


def main():
    # Parse CLI arguments and resolve paths relative to the repo root.
    args = parse_args()
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(REPO_ROOT, args.dataset)
    descriptions_path = (
        args.descriptions if os.path.isabs(args.descriptions) else os.path.join(REPO_ROOT, args.descriptions)
    )
    config = build_config(
        root_dir=REPO_ROOT,
        model_alias=args.model,
        dataset_path=dataset_path,
        descriptions_path=descriptions_path,
    )

    # Load prompts and normalize labels so routing outputs can be compared consistently.
    prompts_df = load_prompts_dataset(config.dataset_path)
    prompts_df["expected_service"] = prompts_df["expected_service"].apply(normalize_service_name)
    services = sorted(prompts_df["expected_service"].astype(str).unique().tolist())
    service_descriptions = load_service_descriptions(config.descriptions_path)

    # Build the router implementation and run the evaluator loop.
    router = build_router(
        model_alias=args.model,
        services=services,
        service_descriptions=service_descriptions,
        device=args.device,
    )

    evaluator = Evaluator(config=config, router=router)
    eval_result = evaluator.run(
        prompts_df=prompts_df,
        model_name=config.resolved_model_name,
        log_progress=args.model == "toolorchestra",
        checkpoint_every=args.checkpoint_every if args.model == "toolorchestra" else 0,
    )
    evaluator.persist_outputs(eval_result=eval_result)

    plot_confusion_matrix(eval_result["predictions_df"], config.confusion_matrix_plot_path)
    plot_per_class_accuracy(eval_result["per_class_accuracy"], config.per_class_accuracy_plot_path)
    plot_model_comparison(config.outputs_dir, config.dataset_name, config.model_accuracy_plot_path)

    row = eval_result["metrics_summary_df"].iloc[0].to_dict()
    # Print a short, human-readable summary to stdout.
    print("Evaluation complete.")
    print(f"Model: {config.resolved_model_name}")
    print(f"Dataset: {row['dataset']}")
    print(f"Total Samples: {int(row['total_samples'])}")
    print(f"Accuracy: {row['accuracy']:.2f}")
    print(f"Top-3 Accuracy: {row['top3_accuracy']:.2f}")
    print(f"Best Performing Class: {row['best_class']} ({row['best_class_accuracy']:.2f})")
    print(f"Worst Performing Class: {row['worst_class']} ({row['worst_class_accuracy']:.2f})")
    print("Outputs saved to:")
    print(f"- {config.dataset_outputs_dir}")


if __name__ == "__main__":
    main()
