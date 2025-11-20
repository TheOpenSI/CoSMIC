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
        seed: int=0,
        is_quantized: bool=False,
        service_index: int=-1,
        device: str="cuda"
    ):
        """Query analyser to select a service.

        Args:
            llm_name (str, optional): LLM name for analyser. Defaults to "mistral-7b-instruct-v0.1".
            seed (int, optional): response generation seed. Defaults to 0.
            is_quantized (bool, optional): use quantized LLM. Defaults to False.
            service_index(int, optional): use selected service, otherwise automatically select.
            device (str, optional): use cuda or cpu for LLM. Defaults to "cuda".
        """
        # Set config.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."
        self.device = device

        # Set a list of services.
        self.services = {
            "0": "if it is a chess game, predict the next chess move by providing a sequence of moves or a FEN",
            "1": "update the vector database with a declarative sentence (not a question) or a pdf document",
            "2": "generate or improve a code or answer a question in order to generate or improve a code",
            "3": "answer a question or provide a reasoning, which cannot be achieved by the other services"
        }

        # Set chess services.
        self.chess_services = {
            "0.0": "predict next move given a chess FEN",
            "0.1": "predict next move given a sequence of moves"
        }

        # Get full services.
        self.full_services = {**self.services, **self.chess_services}

        # Get the number of services.
        self.num_services = len(self.services)

        # Set provided service.
        self.service_index = service_index

        # Build LLM instance from class defined in .py if llm_name is supported.
        if llm_name in LLM_INSTANCE_DICT.keys():
            llm_instance_name = LLM_INSTANCE_DICT[llm_name]
        elif llm_name.find("gpt") > -1:
            llm_instance_name = "GPT"
        elif llm_name.find("ollama") > -1:
            llm_instance_name = "Ollama"
        else:
            print(set_color("error", f"Unsupported LLM: {llm_name}."))
            sys.exit()

        # Build LLM instance from class defined in .py
        self.llm = get_instance(llm_instances, llm_instance_name)(
            llm_name=llm_name,
            seed=seed,
            is_quantized=is_quantized,
            use_example=False,
            is_truncate_response=True,
            device=device
        )

        # Set user prompter for service option.
        self.user_prompter_service = get_instance(
            query_user_prompt_instances,
            "QueryAnalyserService"
        )(services=self.services)

        # Set user prompter for system information.
        self.user_prompter_system_info = get_instance(
            query_user_prompt_instances,
            "QueryAnalyserSystemInfo"
        )(services=self.services)

    def quit(self):
        """Quit by releasing model memory and instance.
        """
        # Release memory of LLM.
        if self.llm: self.llm.quit()

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
        option = re.search('service (\d{1,3}\.\d{1,3}|\d{1,3})', response)

        if option:
            option = option.group(1)

            if option not in self.full_services.keys():
                print(set_color("error", f"Unknown service '{option}' from '{response}'."))

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
        # Option not found.
        if index not in self.full_services.keys():
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

        # Parse FEN string first (more specific pattern).
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
        else:
            # Try multiple move patterns:
            # 1. Bracketed or colon-prefixed
            bracketed_match = re.search('[\[,\:](.*?[,\s].*?)[\.,\]]?$', query)
            
            # 2. Natural language with chess notation"
            # Look for sequences of chess moves (algebraic notation)
            # Chess moves: piece letters (N,B,R,Q,K) or pawn moves (a-h), followed by coordinates or captures
            natural_move_match = re.search(
                r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8][+#]?(?:\s*,?\s*[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8][+#]?)+)\b',
                query
            )
            
            # 3. Simple comma or space-separated notation"
            simple_move_match = re.search(
                r'(?:^|\s)([a-h][1-8](?:\s*,?\s*[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8][+#]?)+)\b',
                query
            )
            
            if bracketed_match:
                # Given a sequence of moves (old format).
                current_moves = bracketed_match.group(1)
                service_option = ["0.1"]
                service_info_dict.update({"moves": current_moves})
            elif natural_move_match:
                # Natural language chess moves detected
                current_moves = natural_move_match.group(1).strip()
                service_option = ["0.1"]
                service_info_dict.update({"moves": current_moves})
            elif simple_move_match:
                # Simple move sequence detected
                current_moves = simple_move_match.group(1).strip()
                service_option = ["0.1"]
                service_info_dict.update({"moves": current_moves})
            else:
                # Invalid inputs - no moves or FEN detected.
                if "predict" in query.lower() or "next move" in query.lower() or "follow-up" in query.lower():
                    print(
                        set_color(
                            "hint",
                            f"For chess move prediction, provide a sequence of moves (e.g., 'e4, e5, Nf3') or FEN notation."
                        )
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
            document_path = re.search('(?<=\:\s)(.*?)+\.pdf', query)

            if document_path:
                document_path = document_path.group()
            else:
                print(set_color(
                    "warning",
                    f"Invalid document query [tip: pdf file(s) is required.]")
                )

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
                    "warning",
                    f"Invalid text query [tip: index the text with :]")
                )

        return service_option, service_info_dict

    def get_system_information_relevance(
        self,
        response: str
    ):
        """Get whether the question is related to system information from response.

        Args:
            response (str): LLM response.

        Returns:
            relevance (bool): whether being related to.
        """
        relevance = response.lower().find("yes") > -1

        return relevance

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
        service_info_dict = {
            "query": query,
            "system_information_relevance": False,
            "system_information": ""
        }
        
        # Early capability detection using the sophisticated logic in user_prompter_system_info
        # This checks if the query is asking "what can you do?" / "what are your services?" etc.
        capability_check_result = self.user_prompter_system_info(query)
        
        # If the prompter returns "YES", it's a direct capability question (detected via regex/patterns)
        if capability_check_result == "YES":
            # This is a capability explanation request - no LLM call needed
            service_info_dict["system_information_relevance"] = True
            service_info_dict["system_information"] = self.user_prompter_system_info.system_information
            # Return a special marker that qa.py can handle
            return {"capability_query": "system_information"}, service_info_dict
        
        # If service_index is configured (not auto-select), use the configured service directly
        # No need for LLM to select - user already chose the service!
        if self.service_index != ["-1"]:
            # If only ONE service is pre-configured, use it directly
            if len(self.service_index) == 1:
                selected_services = dict(self.selected_services)
                
                # Parse the query for the selected service
                option = list(selected_services.keys())[0]
                if option == "0":
                    # Chess service - parse to determine sub-service
                    parsed_option, service_info_dict = self.chess_parse(query, service_info_dict)
                    if parsed_option != "-1" and parsed_option in ["0.0", "0.1"]:
                        del selected_services[option]
                        selected_services[parsed_option] = self.full_services[parsed_option]
            else:
                # Multiple services pre-configured - need to intelligently pick the right one
                # Use LLM to select from the pre-configured services only
                self.llm.set_user_prompter(self.user_prompter_service)
                service_analysis = self.llm(query)[0]
                service_option = self.mapping(service_analysis)
                
                # Normalize to list
                if isinstance(service_option, str):
                    service_option = [service_option]
                
                # Filter: only use services that are BOTH selected by LLM AND in pre-configured list
                selected_services = {}
                for opt in service_option:
                    if opt in self.selected_services:
                        selected_services[opt] = self.selected_services[opt]
                
                # If LLM selected a service not in pre-configured list, use first pre-configured
                if not selected_services:
                    selected_services = dict(self.selected_services)
                
                # Parse the selected service(s)
                for option in list(selected_services.keys()):
                    if option == "0":
                        # Chess service - parse to determine sub-service
                        parsed_option, service_info_dict = self.chess_parse(query, service_info_dict)
                        if parsed_option != "-1" and parsed_option in ["0.0", "0.1"]:
                            del selected_services[option]
                            selected_services[parsed_option] = self.full_services[parsed_option]
                            
                    elif option.startswith("0."):
                        _, service_info_dict = self.chess_parse(query, service_info_dict)

                    elif option == "1":
                        _, service_info_dict = self.update_vector_database_parse(query, service_info_dict)

                    elif option in ["2", "3"]:
                        self.llm.set_user_prompter(self.user_prompter_system_info)
                        relevance_analysis = self.llm(query)[0]
                        relevance = self.get_system_information_relevance(relevance_analysis)
                        service_info_dict["system_information_relevance"] = relevance
                        if relevance:
                            service_info_dict["system_information"] = self.user_prompter_system_info.system_information
            
            if verbose:
                print(set_color(
                    "info",
                    f"Query: {query}, using pre-configured services: {selected_services}."
                ))
            
            return selected_services, service_info_dict
        
        # Auto-select mode: Use LLM to analyze and select appropriate service(s)
        
    
        self.llm.set_user_prompter(self.user_prompter_service)

        # Get raw analysis from LLM to select service(s).
        service_analysis = self.llm(query)[0]

        # Get the service option(s)
        service_option = self.mapping(service_analysis)

        # Always normalize to a list
        if isinstance(service_option, str):
            service_option = [service_option]

        # Filter valid services - only keep those that are in full_services
        selected_services = {
            opt: self.full_services[opt]
            for opt in service_option
            if opt in self.full_services
        }
        
        # If no valid services found, default to Service 3 (General Q&A)
        if not selected_services:
            selected_services = {"3": self.full_services["3"]}

        # Loop through all selected services and parse if needed
        for option in list(selected_services.keys()):
            if option == "0":
                # Main chess service - need to parse to determine sub-service
                parsed_option, service_info_dict = self.chess_parse(query, service_info_dict)
                # Update the selected_services with the actual sub-service
                if parsed_option != "-1" and parsed_option in ["0.0", "0.1"]:
                    # Remove generic "0" and add specific sub-service
                    del selected_services[option]
                    selected_services[parsed_option] = self.full_services[parsed_option]
                    
            elif option.startswith("0."):
                # Already a specific chess sub-service, just parse for info
                _, service_info_dict = self.chess_parse(query, service_info_dict)

            elif option == "1":
                # Update vector database
                _, service_info_dict = self.update_vector_database_parse(query, service_info_dict)

            elif option in ["2", "3"]:
                # Services 2 (code gen) and 3 (general Q&A) - check system info relevance
                self.llm.set_user_prompter(self.user_prompter_system_info)
                relevance_analysis = self.llm(query)[0]
                relevance = self.get_system_information_relevance(relevance_analysis)

                service_info_dict["system_information_relevance"] = relevance

                if relevance:
                    service_info_dict["system_information"] = self.user_prompter_system_info.system_information

        
        if verbose:
            print(set_color(
                "info",
                f"Query: {query}, analysis: {service_analysis}, service: {service_option}."
            ))

        if service_option == "0":
            # Remove last symbol.
            if query[-1] in [",", ".", "!", "?"]: query = query[:-1]

            # Predict the next move in chess game.
            service_option, service_info_dict = self.chess_parse(query, service_info_dict)
        elif service_option == "1":
            # Update the vector database.
            service_option, service_info_dict = self.update_vector_database_parse(
                query,
                service_info_dict
            )
        else:
            # Set the user prompter for system information relevance.
            self.llm.set_user_prompter(self.user_prompter_system_info)

            # Get the response for whether the query is related to system information.
            relevance_analysis: str = self.llm(query)[0]

            # Get whether the question is related to system information.
            relevance: bool = self.get_system_information_relevance(relevance_analysis)

            # Update system information relevance.
            service_info_dict["system_information_relevance"] = relevance

            # Add the system information if it is related to the question.
            if relevance:
                service_info_dict["system_information"] = \
                    self.user_prompter_system_info.system_information

        return service_option, service_info_dict