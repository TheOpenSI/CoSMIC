# -------------------------------------------------------------------------------------------------------------
# File: demo.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
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

import os
import argparse

from src.opensi_cosmic import OpenSICoSMIC

# =============================================================================================================
def run_cli(cosmic_instance: OpenSICoSMIC) -> None:
    while True:
        # Get a query from the terminal.
        query = input("[Query] ")

        # Exit the program.
        if query.lower() in ["quit", "exit"]: break

        # Run for each question/query, return the truncated response if applicable.
        answer, _, _ = cosmic_instance(query)

        # Print the results.
        print(f"[Answer] {answer}")

def run_benchmark(cosmic_instance: OpenSICoSMIC, questions: list[str]) -> None:
    for question in questions:
        print(f"[Question] {question}")
        answer, _, _ = cosmic_instance(question)
        print(f"[Answer] {answer}")
        print("-" * 50)

if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description = "Academic Governance RAG Benchmark")
    parser.add_argument("--benchmark",
                        action = "store_true",
                        default = False,
                        help = "Run benchmark questions.")

    args = parser.parse_args()
    
    # Switch on this to avoid massive warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get the file's absolute path.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = f"{current_dir}"

    # Benchmark Questions
    question_file_path = "data/BM_questions/questions.txt"
    with open(question_file_path, "r") as f:
        questions = f.read().splitlines()

    # Build the system for a specific LLM.
    config_path = os.path.join(root, "scripts/configs/config.yaml")
    opensi_cosmic = OpenSICoSMIC(config_path=config_path)

    if args.benchmark:
        run_benchmark(opensi_cosmic, questions)
    else:
        run_cli(opensi_cosmic)

    # Remove memory cached in the system.
    opensi_cosmic.quit()