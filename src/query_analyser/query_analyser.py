# # -------------------------------------------------------------------------------------------------------------
# # File: query_analyser.py
# # Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# # Contributors:
# #     Danny Xu <danny.xu@canberra.edu.au>
# #
# # Copyright (c) 2024 Open Source Institute
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# # documentation files (the "Software"), to deal in the Software without restriction, including without
# # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# # the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# # conditions:
# #
# # The above copyright notice and this permission notice shall be included in all copies or substantial
# # portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# # LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# # WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# # -------------------------------------------------------------------------------------------------------------

# import os, sys, re

# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

# from src.services.llms import llm as llm_instances
# from src.services.llms.llm import get_instance
# from src.query_analyser import user_prompt as query_user_prompt_instances
# from src.maps import LLM_INSTANCE_DICT
# from utils.log_tool import set_color
# from utils.module import get_instance

# # =============================================================================================================


# class QueryAnalyser:
#     def __init__(
#         self,
#         llm_name: str = "mistral-7b-instruct-v0.1",
#         seed: int = 0,
#         is_quantized: bool = False,
#         service_index: int = -1,
#         device: str = "cuda",
#     ):
#         """Query analyser to select a service.

#         Args:
#             llm_name (str, optional): LLM name for analyser. Defaults to "mistral-7b-instruct-v0.1".
#             seed (int, optional): response generation seed. Defaults to 0.
#             is_quantized (bool, optional): use quantized LLM. Defaults to False.
#             service_index(int, optional): use selected service, otherwise automatically select.
#             device (str, optional): use cuda or cpu for LLM. Defaults to "cuda".
#         """
#         # Set config.
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         self.root = f"{current_dir}/../.."
#         self.device = device

#         # Set a list of services.
#         self.services = {
#             "0": "if it is a chess game, predict the next chess move by providing a sequence of moves or a FEN",
#             "1": "update the vector database with a declarative sentence (not a question) or a pdf document",
#             "2": "generate or improve a code or answer a question in order to generate or improve a code",
#             "3": "answer a question or provide a reasoning, which cannot be achieved by the other services",
#         }

#         # Set chess services.
#         self.chess_services = {
#             "0.0": "predict next move given a chess FEN",
#             "0.1": "predict next move given a sequence of moves",
#         }

#         # Get full services.
#         self.full_services = {**self.services, **self.chess_services}

#         # Get the number of services.
#         self.num_services = len(self.services)

#         # Set provided service.
#         self.service_index = service_index

#         # Build LLM instance from class defined in .py if llm_name is supported.
#         if llm_name in LLM_INSTANCE_DICT.keys():
#             llm_instance_name = LLM_INSTANCE_DICT[llm_name]
#         elif llm_name.find("gpt") > -1:
#             llm_instance_name = "GPT"
#         elif llm_name.find("ollama") > -1:
#             llm_instance_name = "Ollama"
#         else:
#             print(set_color("error", f"Unsupported LLM: {llm_name}."))
#             sys.exit()

#         # Build LLM instance from class defined in .py
#         self.llm = get_instance(llm_instances, llm_instance_name)(
#             llm_name=llm_name,
#             seed=seed,
#             is_quantized=is_quantized,
#             use_example=False,
#             is_truncate_response=True,
#             device=device,
#         )

#         # Set user prompter for service option.
#         self.user_prompter_service = get_instance(
#             query_user_prompt_instances, "QueryAnalyserService"
#         )(services=self.services)

#         # Set user prompter for system information.
#         self.user_prompter_system_info = get_instance(
#             query_user_prompt_instances, "QueryAnalyserSystemInfo"
#         )(services=self.services)

#     def quit(self):
#         """Quit by releasing model memory and instance."""
#         # Release memory of LLM.
#         if self.llm:
#             self.llm.quit()

#     def mapping(self, response: str):
#         """Parse response to get service option.

#         Args:
#             response (str): response from LLM analysis.

#         Returns:
#             option (int): index of the service option if avaiable; otherwise, None.
#         """
#         # Set to lower case.
#         response = response.lower()

#         # Truncate to get the option index.
#         option = re.search("service (\d{1,3}\.\d{1,3}|\d{1,3})", response)

#         if option:
#             option = option.group(1)

#             if option not in self.full_services.keys():
#                 print(
#                     set_color("error", f"Unknown service '{option}' from '{response}'.")
#                 )

#                 return "-1"
#         else:
#             return "-1"

#         return option

#     def get_service(self, index: int):
#         """Get the description of the service.

#         Args:
#             index (int): index of the service.

#         Returns:
#             service (str): description of the service.
#         """
#         # Option not found.
#         if index not in self.full_services.keys():
#             return None

#         return self.services[index]

#     def chess_parse(self, query: str, service_info_dict: dict):
#         """Parse query to get chess service.

#         Args:
#             query (str): question.
#             service_info_dict (dict): dictionary to contain parsed information.

#         Returns:
#             service_option (str): service option.
#             service_info_dict (dict): updated information dictionary.
#         """
#         # Default service option.
#         service_option = "-1"

#         # Parse move string.
#         move_match = re.search("[\[,\:](.*?[,\s].*?)[\.,\]]?$", query)

#         # Parse FEN string.
#         fen_match = re.search(
#             "(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)"
#             "\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$",
#             query,
#         )

#         if fen_match:
#             # Given FEN.
#             current_fen = fen_match.group()
#             service_option = "0.0"
#             service_info_dict.update({"fen": current_fen})
#         elif move_match:
#             # Given a sequence of moves.
#             current_moves = move_match.group(1)
#             service_option = "0.1"
#             service_info_dict.update({"moves": current_moves})
#         else:
#             # Invalid inputs.
#             if query.find("predict") > -1 or query.find("next move") > -1:
#                 print(
#                     set_color(
#                         "hint",
#                         f'For chess move prediction, index a sequence of moves or FEN with ":".',
#                     )
#                 )

#         return service_option, service_info_dict

#     def update_vector_database_parse(self, query: str, service_info_dict: dict):
#         """Parse query to get text or document path to update vector database.

#         Args:
#             query (str): question.
#             service_info_dict (dict): dictionary to contain parsed information.

#         Returns:
#             service_option (str): service option.
#             service_info_dict (dict): updated information dictionary.
#         """
#         service_option = "1"

#         # Check if context is a .pdf.
#         is_a_document = query.find(".pdf") > -1

#         # Update information dictionary.
#         service_info_dict.update(
#             {"is_a_document": is_a_document, "text": None, "document_path": None}
#         )

#         if is_a_document:
#             # Parse move string
#             document_path = re.search("(?<=\:\s)(.*?)+\.pdf", query)

#             if document_path:
#                 document_path = document_path.group()
#             else:
#                 print(
#                     set_color(
#                         "warning",
#                         f"Invalid document query [tip: pdf file(s) is required.]",
#                     )
#                 )

#                 return service_option, service_info_dict

#             # Check if not an absolute path, convert to an absoluate path.
#             if not os.path.isabs(document_path):
#                 document_path = os.path.join(self.root, document_path)

#             # Update information dictionary.
#             service_info_dict["document_path"] = document_path
#         else:
#             # Extract text.
#             text = re.search("\:((\"|')?(.*?)[\",']?$)", query)

#             if text:
#                 text = text.group(0)
#                 text = text.replace(": ", "").replace(":", "")

#                 # Update information dictionary.
#                 service_info_dict["text"] = text
#             else:
#                 print(
#                     set_color(
#                         "warning", f"Invalid text query [tip: index the text with :]"
#                     )
#                 )

#         return service_option, service_info_dict

#     def get_system_information_relevance(self, response: str):
#         """Get whether the question is related to system information from response.

#         Args:
#             response (str): LLM response.

#         Returns:
#             relevance (bool): whether being related to.
#         """
#         relevance = response.lower().find("yes") > -1

#         return relevance

#     def __call__(self, query: str, verbose: bool = False):
#         """Analyse query to get service option.

#         Args:
#             query (str): question.
#             verbose (bool, optional): debug mode. Default to False.

#         Returns:
#             service_option (str): service option.
#             service_info_dict (dict): updated information dictionary.
#         """
#         # Create an initial information dictionary.
#         service_info_dict = {
#             "query": query,
#             "system_information_relevance": False,
#             "system_information": "",
#         }

#         if self.service_index >= 0:
#             service_option = str(self.service_index)
#         else:
#             # Set the user prompter for service option.
#             self.llm.set_user_prompter(self.user_prompter_service)

#             # Get raw anlysis from LLM to select a service.
#             service_analysis = self.llm(query)[0]

#             # Get the service option.
#             service_option = self.mapping(service_analysis)

#         # Analysis information.
#         if verbose:
#             print(
#                 set_color(
#                     "info",
#                     f"Query: {query}, analysis: {service_analysis}, service: {service_option}.",
#                 )
#             )

#         if service_option == "0":
#             # Remove last symbol.
#             if query[-1] in [",", ".", "!", "?"]:
#                 query = query[:-1]

#             # Predict the next move in chess game.
#             service_option, service_info_dict = self.chess_parse(
#                 query, service_info_dict
#             )
#         elif service_option == "1":
#             # Update the vector database.
#             service_option, service_info_dict = self.update_vector_database_parse(
#                 query, service_info_dict
#             )
#         else:
#             # Set the user prompter for system information relevance.
#             self.llm.set_user_prompter(self.user_prompter_system_info)

#             # Get the response for whether the query is related to system information.
#             relevance_analysis: str = self.llm(query)[0]

#             # Get whether the question is related to system information.
#             relevance: bool = self.get_system_information_relevance(relevance_analysis)

#             # Update system information relevance.
#             service_info_dict["system_information_relevance"] = relevance

#             # Add the system information if it is related to the question.
#             if relevance:
#                 service_info_dict["system_information"] = (
#                     self.user_prompter_system_info.system_information
#                 )

#         return service_option, service_info_dict


# #!/usr/bin/env python3
# # -------------------------------------------------------------------------------------------------------------
# # File: query_analyser.py
# # Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# # Contributors:
# #     Danny Xu <danny.xu@canberra.edu.au>
# #     Muntasir Adnan <adnan.adnan@canberra.edu.au>
# #
# # Usage (inside Docker container):
# #   python src/query_analyser/query_analyser.py --csv /path/to/data.csv [--gt-col 6] [--use-example]
# # -------------------------------------------------------------------------------------------------------------

# import argparse
# import os
# import sys
# import time

# import pandas as pd
# import requests

# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

# from src.query_analyser.user_prompt import QueryAnalyserService, SERVICES
# from src.services.llms.prompts.system_prompt import ServiceSelectorPrompt

# # =============================================================================================================
# # Ollama LLM call — matches how CoSMIC talks to the local Ollama container
# # =============================================================================================================

# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")


# def call_ollama(messages: list, model: str = OLLAMA_MODEL) -> str:
#     """Send a chat-style message list to the local Ollama server.

#     Args:
#         messages (list): list of {role, content} dicts.
#         model (str): Ollama model tag.

#     Returns:
#         str: model reply, stripped of whitespace.
#     """
#     payload = {
#         "model": model,
#         "messages": messages,
#         "stream": False,
#         "options": {"temperature": 0},  # deterministic routing
#     }
#     response = requests.post(
#         f"{OLLAMA_URL}/api/chat",
#         json=payload,
#         timeout=60,
#     )
#     response.raise_for_status()
#     return response.json()["message"]["content"].strip()


# # =============================================================================================================
# # Tag normalisation
# # =============================================================================================================


# def normalise_tag(raw: str, valid_tags: list) -> str:
#     """Map raw LLM output to the closest valid service tag.

#     Args:
#         raw (str): raw text returned by the LLM.
#         valid_tags (list[str]): list of acceptable tag strings.

#     Returns:
#         str: matched tag, or cleaned raw string if no match found.
#     """
#     cleaned = raw.lower().strip().rstrip(".")

#     if cleaned in valid_tags:
#         return cleaned

#     # Substring match — handles cases like "service abstract_algebra" or extra words
#     for tag in valid_tags:
#         if tag in cleaned:
#             return tag

#     return cleaned  # Surface unexpected responses rather than silently masking them


# # =============================================================================================================
# # Orchestrator
# # =============================================================================================================


# class QueryOrchestrator:
#     """Routes a query to the best-matching service using an LLM.

#     Parameters
#     ----------
#     services : dict
#         Mapping of service_tag → description (defaults to SERVICES defined in user_prompt.py).
#     use_example : bool
#         Whether to inject a one-shot example into the system prompt.
#     model : str
#         Ollama model tag.
#     """

#     def __init__(
#         self,
#         services: dict = SERVICES,
#         use_example: bool = False,
#         model: str = OLLAMA_MODEL,
#     ):
#         self.services = services
#         self.model = model

#         self.user_prompt_builder = QueryAnalyserService(services)
#         self.system_prompt_builder = ServiceSelectorPrompt(
#             service_tags=list(services.keys()),
#             use_example=use_example,
#         )

#     def select_service(self, query: str) -> str:
#         """Return the service tag best suited to answer the query.

#         Args:
#             query (str): raw user question from the CSV.

#         Returns:
#             str: selected service tag — this is printed, NOT called.
#         """
#         user_prompt = self.user_prompt_builder(query)
#         messages = self.system_prompt_builder(user_prompt)
#         raw_reply = call_ollama(messages, model=self.model)
#         return normalise_tag(raw_reply, list(self.services.keys()))


# # =============================================================================================================
# # Benchmark runner
# # =============================================================================================================


# def run_benchmark(
#     csv_path: str,
#     gt_col: int | None,
#     use_example: bool,
#     delay: float,
# ) -> None:
#     """Run service-selection over every row of the CSV and print results.

#     Column layout assumed (no header row):
#         col 0  → question / prompt          ← INPUT
#         col 6  → dataset/service label      ← GROUND TRUTH (optional)

#     Args:
#         csv_path (str): path to the CSV file.
#         gt_col (int | None): column index of ground-truth label, or None.
#         use_example (bool): include a few-shot example in system prompt.
#         delay (float): seconds to wait between LLM calls.
#     """
#     # ── Load CSV ─────────────────────────────────────────────────────────────
#     print(f"[INFO] Loading queries from: {csv_path}")
#     df = pd.read_csv(csv_path, header=None)
#     total = len(df)
#     print(f"[INFO] {total} queries found.")
#     print(f"[INFO] Services: {', '.join(SERVICES.keys())}")
#     print(f"[INFO] Model   : {OLLAMA_MODEL}\n")

#     # ── Orchestrator ─────────────────────────────────────────────────────────
#     orchestrator = QueryOrchestrator(use_example=use_example)

#     # ── Iterate ───────────────────────────────────────────────────────────────
#     correct = 0
#     separator = "-" * 74

#     print(separator)
#     print(f"{'#':<6} {'SELECTED SERVICE':<32} {'GROUND TRUTH':<30}")
#     print(separator)

#     for idx, row in df.iterrows():
#         query = str(row[0])
#         gt = str(row[gt_col]).strip().lower() if gt_col is not None else None

#         predicted = orchestrator.select_service(query)

#         match_marker = ""
#         if gt is not None:
#             if predicted == gt:
#                 correct += 1
#                 match_marker = "  ✓"
#             else:
#                 match_marker = "  ✗"

#         gt_display = gt if gt is not None else "N/A"
#         print(f"{idx + 1:<6} {predicted:<32} {gt_display:<30}{match_marker}")

#         if delay > 0:
#             time.sleep(delay)

#     # ── Summary ───────────────────────────────────────────────────────────────
#     print(separator)
#     print(f"\n[SUMMARY] Total queries  : {total}")

#     if gt_col is not None:
#         accuracy = correct / total * 100
#         print(f"[SUMMARY] Correct        : {correct}/{total}")
#         print(f"[SUMMARY] Accuracy       : {accuracy:.1f}%")
#     else:
#         print("[SUMMARY] No ground-truth column — accuracy not computed.")
#     print()


# # =============================================================================================================
# # CLI
# # =============================================================================================================


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="CoSMIC orchestrator benchmark — service selection from CSV input.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument(
#         "--csv",
#         required=True,
#         metavar="PATH",
#         help="Path to the input CSV file (no header; queries in column 0).",
#     )
#     parser.add_argument(
#         "--gt-col",
#         type=int,
#         default=6,
#         metavar="N",
#         help="Column index (0-based) of the ground-truth service label. "
#         "Pass -1 to disable accuracy reporting.",
#     )
#     parser.add_argument(
#         "--use-example",
#         action="store_true",
#         help="Include a one-shot example in the system prompt.",
#     )
#     parser.add_argument(
#         "--delay",
#         type=float,
#         default=0.1,
#         metavar="SEC",
#         help="Seconds to wait between LLM calls.",
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     gt_col = args.gt_col if args.gt_col >= 0 else None

#     run_benchmark(
#         csv_path=args.csv,
#         gt_col=gt_col,
#         use_example=args.use_example,
#         delay=args.delay,
#     )


# -------------------------------------------------------------------------------------------------------------
# File: query_analyser.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#
# Copyright (c) 2024 Open Source Institute
# -------------------------------------------------------------------------------------------------------------

import os, sys, re

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.services.llms import llm as llm_instances
from src.services.llms.llm import get_instance
from src.query_analyser import user_prompt as query_user_prompt_instances
from src.maps import LLM_INSTANCE_DICT
from utils.log_tool import set_color
from utils.module import get_instance

# =============================================================================================================


class QueryAnalyser:
    def __init__(
        self,
        llm_name: str = "mistral-7b-instruct-v0.1",
        seed: int = 0,
        is_quantized: bool = False,
        service_index: int = -1,
        device: str = "cuda",
    ):
        # Set config.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."
        self.device = device

        # Set a list of services.
        self.services = {
            "0": "if it is a chess game, predict the next chess move by providing a sequence of moves or a FEN",
            "1": "update the vector database with a declarative sentence (not a question) or a pdf document",
            "2": "generate or improve a code or answer a question in order to generate or improve a code",
            "3": "answer a question or provide a reasoning, which cannot be achieved by the other services",
        }

        self.chess_services = {
            "0.0": "predict next move given a chess FEN",
            "0.1": "predict next move given a sequence of moves",
        }

        self.full_services = {**self.services, **self.chess_services}
        self.num_services = len(self.services)
        self.service_index = service_index

        if llm_name in LLM_INSTANCE_DICT.keys():
            llm_instance_name = LLM_INSTANCE_DICT[llm_name]
        elif llm_name.find("gpt") > -1:
            llm_instance_name = "GPT"
        elif llm_name.find("ollama") > -1:
            llm_instance_name = "Ollama"
        else:
            print(set_color("error", f"Unsupported LLM: {llm_name}."))
            sys.exit()

        self.llm = get_instance(llm_instances, llm_instance_name)(
            llm_name=llm_name,
            seed=seed,
            is_quantized=is_quantized,
            use_example=False,
            is_truncate_response=True,
            device=device,
        )

        self.user_prompter_service = get_instance(
            query_user_prompt_instances, "QueryAnalyserService"
        )(services=self.services)

        self.user_prompter_system_info = get_instance(
            query_user_prompt_instances, "QueryAnalyserSystemInfo"
        )(services=self.services)

    def quit(self):
        if self.llm:
            self.llm.quit()

    def mapping(self, response: str):
        response = response.lower()
        option = re.search("service (\d{1,3}\.\d{1,3}|\d{1,3})", response)
        if option:
            option = option.group(1)
            if option not in self.full_services.keys():
                print(
                    set_color("error", f"Unknown service '{option}' from '{response}'.")
                )
                return "-1"
        else:
            return "-1"
        return option

    def get_service(self, index: int):
        if index not in self.full_services.keys():
            return None
        return self.services[index]

    def chess_parse(self, query: str, service_info_dict: dict):
        service_option = "-1"
        move_match = re.search("[\[,\:](.*?[,\s].*?)[\.,\]]?$", query)
        fen_match = re.search(
            "(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)"
            "\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$",
            query,
        )
        if fen_match:
            current_fen = fen_match.group()
            service_option = "0.0"
            service_info_dict.update({"fen": current_fen})
        elif move_match:
            current_moves = move_match.group(1)
            service_option = "0.1"
            service_info_dict.update({"moves": current_moves})
        else:
            if query.find("predict") > -1 or query.find("next move") > -1:
                print(
                    set_color(
                        "hint",
                        f'For chess move prediction, index a sequence of moves or FEN with ":".',
                    )
                )
        return service_option, service_info_dict

    def update_vector_database_parse(self, query: str, service_info_dict: dict):
        service_option = "1"
        is_a_document = query.find(".pdf") > -1
        service_info_dict.update(
            {"is_a_document": is_a_document, "text": None, "document_path": None}
        )
        if is_a_document:
            document_path = re.search("(?<=\:\s)(.*?)+\.pdf", query)
            if document_path:
                document_path = document_path.group()
            else:
                print(
                    set_color(
                        "warning",
                        f"Invalid document query [tip: pdf file(s) is required.]",
                    )
                )
                return service_option, service_info_dict
            if not os.path.isabs(document_path):
                document_path = os.path.join(self.root, document_path)
            service_info_dict["document_path"] = document_path
        else:
            text = re.search("\:((\"|')?(.*?)[\",']?$)", query)
            if text:
                text = text.group(0)
                text = text.replace(": ", "").replace(":", "")
                service_info_dict["text"] = text
            else:
                print(
                    set_color(
                        "warning", f"Invalid text query [tip: index the text with :]"
                    )
                )
        return service_option, service_info_dict

    def get_system_information_relevance(self, response: str):
        return response.lower().find("yes") > -1

    def __call__(self, query: str, verbose: bool = False):
        service_info_dict = {
            "query": query,
            "system_information_relevance": False,
            "system_information": "",
        }
        if self.service_index >= 0:
            service_option = str(self.service_index)
        else:
            self.llm.set_user_prompter(self.user_prompter_service)
            service_analysis = self.llm(query)[0]
            service_option = self.mapping(service_analysis)

        if verbose:
            print(
                set_color(
                    "info",
                    f"Query: {query}, analysis: {service_analysis}, service: {service_option}.",
                )
            )

        if service_option == "0":
            if query[-1] in [",", ".", "!", "?"]:
                query = query[:-1]
            service_option, service_info_dict = self.chess_parse(
                query, service_info_dict
            )
        elif service_option == "1":
            service_option, service_info_dict = self.update_vector_database_parse(
                query, service_info_dict
            )
        else:
            self.llm.set_user_prompter(self.user_prompter_system_info)
            relevance_analysis: str = self.llm(query)[0]
            relevance: bool = self.get_system_information_relevance(relevance_analysis)
            service_info_dict["system_information_relevance"] = relevance
            if relevance:
                service_info_dict["system_information"] = (
                    self.user_prompter_system_info.system_information
                )

        return service_option, service_info_dict


import argparse
import time
import pandas as pd
import requests

from src.query_analyser.user_prompt import QueryAnalyserService, SERVICES
from src.services.llms.prompts.system_prompt import ServiceSelectorPrompt

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")


def call_ollama(messages: list, model: str = OLLAMA_MODEL, retries: int = 3) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            print(f"  [WARN] Timeout attempt {attempt}/{retries}, retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"  [WARN] Error attempt {attempt}/{retries}: {e}")
            time.sleep(5)
    return "error_timeout"


def normalise_tag(raw: str, valid_tags: list) -> str:
    cleaned = raw.lower().strip().rstrip(".")
    if cleaned in valid_tags:
        return cleaned
    for tag in valid_tags:
        if tag in cleaned:
            return tag
    return cleaned


class QueryOrchestrator:
    def __init__(
        self,
        services: dict = SERVICES,
        use_example: bool = False,
        model: str = OLLAMA_MODEL,
    ):
        self.services = services
        self.model = model
        self.user_prompt_builder = QueryAnalyserService(services)
        self.system_prompt_builder = ServiceSelectorPrompt(
            service_tags=list(services.keys()),
            use_example=use_example,
        )

    def select_service(self, query: str) -> str:
        user_prompt = self.user_prompt_builder(query)
        messages = self.system_prompt_builder(user_prompt)
        raw_reply = call_ollama(messages, model=self.model)
        return normalise_tag(raw_reply, list(self.services.keys()))


def run_benchmark(csv_path: str, gt_col, use_example: bool, delay: float) -> None:
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, header=None)
    total = len(df)
    print(f"[INFO] Queries  : {total}")
    print(f"[INFO] Services : {', '.join(SERVICES.keys())}")
    print(f"[INFO] Model    : {OLLAMA_MODEL}\n")

    orchestrator = QueryOrchestrator(use_example=use_example)
    correct = 0
    sep = "-" * 80

    print(sep)
    if gt_col is not None:
        print(f"{'#':<6} {'SELECTED':<32} {'EXPECTED':<32} MATCH")
    else:
        print(f"{'#':<6} {'SELECTED SERVICE':<32} PROMPT (first 60 chars)")
    print(sep)

    for idx, row in df.iterrows():
        query = str(row[0])
        gt = str(row[gt_col]).strip().lower() if gt_col is not None else None
        predicted = orchestrator.select_service(query)

        if gt is not None:
            match = "✓" if predicted == gt else "✗"
            if predicted == gt:
                correct += 1
            print(f"{idx + 1:<6} {predicted:<32} {gt:<32} {match}")
        else:
            print(f"{idx + 1:<6} {predicted:<32} {query[:60]}")

        if delay > 0:
            time.sleep(delay)

    print(sep)
    print(f"\n[DONE] {total} queries processed.")
    if gt_col is not None:
        print(f"[DONE] Accuracy : {correct}/{total} = {correct/total*100:.1f}%")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--gt-col", type=int, default=-1)
    parser.add_argument("--use-example", action="store_true")
    parser.add_argument("--delay", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gt_col = args.gt_col if args.gt_col >= 0 else None
    run_benchmark(
        csv_path=args.csv,
        gt_col=gt_col,
        use_example=args.use_example,
        delay=args.delay,
    )
