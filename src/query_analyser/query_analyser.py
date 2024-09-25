# -------------------------------------------------------------------------------------------------------------
# File: query_analyser.py
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
        llm_name: str="mistral-7b-instruct-v0.1",
        user_prompt_instance_name: str="QueryAnalyser",
        seed: int=0,
    ):
        """Query analyser to select a service.

        Args:
            llm_name (str, optional): LLM name for analyser. Defaults to "mistral-7b-instruct-v0.1".
            user_prompt_instance_name (str, optional): user prompter name. Defaults to "QueryAnalyser".
            seed (int, optional): response generation seed. Defaults to 0.
        """
        # Set config.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."

        # Set a list of services.
        self.services = {
            "0": "predict the next or the best move in a chess game",
            "1": "add a statement or a document to vector database",
            "2": "answer a question or provide a reasoning",
            "3": "generate code in any programming language"
        }

        # Set chess services.
        self.chess_services = {
            "0.0": "predict next move given a chess FEN",
            "0.1": "predict next move given a sequence of moves"
        }

        # Get the number of services.
        self.num_services = len(self.services)

        # Check if llm_name is supported.
        if llm_name not in LLM_INSTANCE_DICT.keys():
            print(set_color("error", f"Unsupported LLM: {llm_name}."))
            sys.exit()

        # Build LLM instance from class defined in .py
        self.llm = get_instance(llm_instances, LLM_INSTANCE_DICT[llm_name])(
            seed=seed,
            is_quantized=True,
            use_example=False,
            is_truncate_response=True,
        )

        # Set user prompt with services to LLM.
        user_prompter = get_instance(
            query_user_prompt_instances,
            user_prompt_instance_name
        )(services=self.services)

        self.llm.set_user_prompter(user_prompter)

    def quit(self):
        """Quit by releasing model memory and instance.
        """
        # Release memory of LLM.
        self.llm.quit()

    def mapping(
        self,
        response: str
    ):
        """Parse response to get service option.

        Args:
            response (str): response from LLM analysis.

        Returns:
            option (int): index of the service option if avaiable; otherwise, None.
        """
        # Set to lower case.
        response = response.lower()

        # Truncate to get the option index.
        option = re.search('option (\d{1,3}\.\d{1,3}|\d{1,3})', response)

        if option:
            option = option.group(1)
            full_services = {**self.services, **self.chess_services}

            if option not in full_services.keys():
                print(set_color("error", f"Unknown option '{option}' from '{response}'."))

                return "-1"
        else:
            return "-1"

        return option

    def get_service(
        self,
        index: int
    ):
        """Get the description of the service.

        Args:
            index (int): index of the service.

        Returns:
            service (str): description of the service.
        """
        full_services = {**self.services, **self.chess_services}

        # Option not found.
        if index not in full_services.keys():
            return None

        return self.services[index]

    def chess_parse(
        self,
        query: str,
        service_info_dict: dict
    ):
        """Parse query to get chess service.

        Args:
            query (str): question.
            service_info_dict (dict): dictionary to contain parsed information.

        Returns:
            service_option (str): service option.
            service_info_dict (dict): updated information dictionary.
        """
        # Default service option.
        service_option = "-1"

        # Parse move string.
        move_match = re.search('[\[,\:](.*?[,\s].*?)[\.,\]]?$', query)

        # Parse FEN string.
        fen_match = re.search(
            '(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)' \
            '\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$',
            query
        )

        if fen_match:
            # Given FEN.
            current_fen = fen_match.group()
            service_option = "0.0"
            service_info_dict.update({"fen": current_fen})
        elif move_match:
            # Given a sequence of moves.
            current_moves = move_match.group(1)
            service_option = "0.1"
            service_info_dict.update({"moves": current_moves})
        else:
            # Invalid inputs.
            print(set_color(
                "warning",
                f"Chess query: '{query}'. [tip: index a sequence of moves or FEN with :]")
            )

        return service_option, service_info_dict

    def update_vector_database_parse(
        self,
        query: str,
        service_info_dict: dict
    ):
        """Parse query to get text or document path to update vector database.

        Args:
            query (str): question.
            service_info_dict (dict): dictionary to contain parsed information.

        Returns:
            service_option (str): service option.
            service_info_dict (dict): updated information dictionary.
        """
        service_option = "1"

        # Check if context is a .pdf.
        is_a_document = query.find(".pdf") > -1

        # Update information dictionary.
        service_info_dict.update({
            "is_a_document": is_a_document,
            "text": None,
            "document_path": None
        })

        if is_a_document:
            # Parse move string
            document_path = re.search('(\S+\.pdf)', query)

            if document_path:
                document_path = document_path.group()
            else:
                print(set_color("error", f"Invalid query: {query}"))

                return service_option, service_info_dict

            # Check if not an absolute path, convert to an absoluate path.
            if not os.path.isabs(document_path):
                document_path = os.path.join(self.root, document_path)

            # Update information dictionary.
            service_info_dict["document_path"] = document_path
        else:
            # Extract text.
            text = re.search('\:((\"|\')?(.*?)[\",\']?$)', query)

            if text:
                text = text.group(0)
                text = text.replace(": ", "").replace(":", "")

                # Update information dictionary.
                service_info_dict["text"] = text
            else:
                print(set_color(
                    "error",
                    f"Invalid query: {query} [tip: index the text with :]"
                ))

        return service_option, service_info_dict

    def __call__(
        self,
        query: str,
        verbose: bool=False
    ):
        """Analyse query to get service option.

        Args:
            query (str): question.
            verbose (bool, optional): debug mode. Default to False.

        Returns:
            service_option (str): service option.
            service_info_dict (dict): updated information dictionary.
        """
        # Create an initial information dictionary.
        service_info_dict = {"query": query}

        # Get raw anlysis from LLM to select a service.
        service_analysis = self.llm(query)[0]

        # Get the service option.
        service_option = self.mapping(service_analysis)

        # Analysis information.
        if verbose:
            print(set_color(
                "info",
                f"Query: {query}, analysis: {service_analysis}, service: {service_option}."
            ))

        if service_option == "0":
            # Predict the next move in chess game.
            service_option, service_info_dict = self.chess_parse(query, service_info_dict)
        elif service_option == "1":
            # Update the vector database.
            service_option, service_info_dict = self.update_vector_database_parse(
                query,
                service_info_dict
            )

        return service_option, service_info_dict