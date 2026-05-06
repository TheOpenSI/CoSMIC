# -------------------------------------------------------------------------------------------------------------
# File: main.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
# 
# Copyright (c) 2024 Open Source Institute
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import argparse
import csv
import os
import re
import sys
import pandas as pd

# =============================================================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from experiments.retrieval_benchmark_suite import cli as retrieval_cli
from experiments.semantic_router import run as semantic_router_run
from experiments.tool_orchestra_benchmark import run_benchmark as tool_orchestra_run


def _canonical_dataset_paths(root_dir: str):
    return [
        os.path.join(root_dir, "datasets", "dataset_3_services.csv"),
        os.path.join(root_dir, "datasets", "dataset_5_services.csv"),
        os.path.join(root_dir, "datasets", "dataset_8_services.csv"),
        os.path.join(root_dir, "datasets", "dataset_10_services.csv"),
    ]


def _resolve_dataset_paths(root_dir: str, dataset_args):
    if dataset_args:
        paths = []
        for item in dataset_args:
            path = item if os.path.isabs(item) else os.path.join(root_dir, item)
            if not os.path.exists(path):
                raise ValueError(f"Dataset file not found: {path}")
            paths.append(path)
        return paths
    return _canonical_dataset_paths(root_dir)


def _discover_dataset_filenames(root_dir: str):
    datasets_dir = os.path.join(root_dir, "datasets")
    filenames = []
    for file_name in os.listdir(datasets_dir):
        if re.fullmatch(r"dataset_\d+_services\.csv", file_name):
            filenames.append(file_name)

    def _service_count_key(name: str):
        matched = re.search(r"dataset_(\d+)_services\.csv", name)
        return int(matched.group(1)) if matched else 10**9

    return sorted(filenames, key=_service_count_key)


def _call_with_argv(module_main, argv):
    previous_argv = sys.argv[:]
    try:
        sys.argv = argv
        module_main()
    finally:
        sys.argv = previous_argv


def run_semantic_router(root_dir: str, models):
    _ = root_dir
    for model in models:
        print(f"[benchmark] semantic-router model={model}", flush=True)
        _call_with_argv(
            semantic_router_run.main,
            ["semantic_router/run.py", "--model", model],
        )


def run_tool_orchestra(root_dir: str, models, dataset_paths, descriptions_path: str, device: str, checkpoint_every: int):
    for model in models:
        for dataset_path in dataset_paths:
            print(
                f"[benchmark] tool-orchestra model={model} dataset={os.path.basename(dataset_path)}",
                flush=True,
            )
            _call_with_argv(
                tool_orchestra_run.main,
                [
                    "tool_orchestra_benchmark/run_benchmark.py",
                    "--model",
                    model,
                    "--dataset",
                    dataset_path,
                    "--descriptions",
                    descriptions_path,
                    "--device",
                    device,
                    "--checkpoint-every",
                    str(checkpoint_every),
                ],
            )


def run_retrieval_suite(dataset_paths, descriptions_path: str, methods, device: str):
    for method in methods:
        for dataset_path in dataset_paths:
            print(
                f"[benchmark] retrieval method={method} dataset={os.path.basename(dataset_path)}",
                flush=True,
            )
            _call_with_argv(
                retrieval_cli.main,
                [
                    "retrieval_benchmark_suite/cli.py",
                    "--method",
                    method,
                    "--dataset",
                    dataset_path,
                    "--descriptions",
                    descriptions_path,
                    "--device",
                    device,
                ],
            )


def parse_benchmark_args():
    parser = argparse.ArgumentParser(description="Unified benchmark launcher for CoSMIC experiment modules.")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["semantic-router", "tool-orchestra", "retrieval-suite"],
        help="Run one benchmark module.",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Run all benchmark modules.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset path. Repeat flag for multiple datasets.",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default="datasets/dataset_descriptions.xlsx",
        help="Path to service descriptions spreadsheet.",
    )
    parser.add_argument(
        "--semantic-models",
        nargs="+",
        default=["miniLM", "bge", "openai"],
        choices=["miniLM", "bge", "openai"],
        help="Semantic router model aliases.",
    )
    parser.add_argument(
        "--tool-models",
        nargs="+",
        default=["toolorchestra", "openai"],
        choices=["toolorchestra", "openai"],
        help="Tool orchestra benchmark model aliases.",
    )
    parser.add_argument(
        "--retrieval-methods",
        nargs="+",
        default=["bi-encoder", "cross-encoder", "colbert"],
        choices=["bi-encoder", "cross-encoder", "colbert"],
        help="Retrieval suite method aliases.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Execution device passed to supported benchmark modules.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Checkpoint interval for tool-orchestra benchmark.",
    )
    return parser.parse_known_args()


def run_legacy_cosmic():
    from src.opensi_cosmic import OpenSICoSMIC
    from utils.log_tool import set_color

    # Switch on this to avoid massive warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get the file's absolute path.
    root = f"{CURRENT_DIR}"

    # Set a bunch of questions, can also read from .csv.
    df = pd.read_csv(f"{root}/data/test.csv")
    queries = df["Question"]
    answers = df["Answer"]

    # Build the system for a specific LLM.
    config_path = os.path.join(root, "scripts/configs/config.yaml")
    opensi_cosmic = OpenSICoSMIC(config_path=config_path)

    # Loop over questions to get the answers.
    for idx, (query, gt) in enumerate(zip(queries, answers)):
        # Skip marked questions.
        if query.find("skip") > -1: continue

        # Create a log file.
        if query.find(".csv") > -1:
            # Remove all namespace.
            query = query.replace(" ", "")

            # Return if file is invalid.
            if not os.path.exists(query):
                set_color("error", f"!!! Error, {query} not exist.")
                continue

            # Change the data folder to results for log file.
            log_file = query.replace("/data/", f"/results/{opensi_cosmic.config.llm_name}/")

            # Create a folder to store log file.
            log_file_name = log_file.split("/")[-1]
            log_dir = log_file.replace(log_file_name, "")
            os.makedirs(log_dir, exist_ok=True)
            log_file_pt = open(log_file, "w")
            log_file = csv.writer(log_file_pt)
        else:
            log_file_pt = None
            log_file = None

        # Run for each question/query, return the truncated response if applicable.
        answer, _, _ = opensi_cosmic(query, log_file=log_file)

        # Print the answer.
        if answer is not None and isinstance(gt, str):  # compare with GT string
            # Assign to q variables.
            status = "success" if (answer.find(gt) > -1) else "fail"

            print(set_color(status, f"\nQuestion: '{query}' with GT: {gt}.\nAnswer: '{answer}'."))
        elif answer is not None and answer != '':
            print(set_color("info", f"\nQuestion: '{query}'.\nAnswer: '{answer}'."))

        # Close log file pointer.
        if log_file_pt is not None:
            log_file_pt.close()
        
    # Remove memory cached in the system.
    opensi_cosmic.quit()


if __name__ == "__main__":
    args, unknown = parse_benchmark_args()
    if unknown:
        print(f"[warn] ignored unknown args: {' '.join(unknown)}", flush=True)

    run_module = args.benchmark is not None or args.benchmark_all
    if not run_module:
        run_legacy_cosmic()
        raise SystemExit(0)

    root_dir = CURRENT_DIR
    descriptions_path = args.descriptions if os.path.isabs(args.descriptions) else os.path.join(root_dir, args.descriptions)
    if not os.path.exists(descriptions_path):
        raise ValueError(f"Descriptions file not found: {descriptions_path}")
    dataset_paths = _resolve_dataset_paths(root_dir=root_dir, dataset_args=args.dataset)

    if args.benchmark_all or args.benchmark == "semantic-router":
        run_semantic_router(root_dir=root_dir, models=args.semantic_models)

    if args.benchmark_all or args.benchmark == "tool-orchestra":
        run_tool_orchestra(
            root_dir=root_dir,
            models=args.tool_models,
            dataset_paths=dataset_paths,
            descriptions_path=descriptions_path,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
        )

    if args.benchmark_all or args.benchmark == "retrieval-suite":
        run_retrieval_suite(
            dataset_paths=dataset_paths,
            descriptions_path=descriptions_path,
            methods=args.retrieval_methods,
            device="cpu" if args.device == "auto" else args.device,
        )