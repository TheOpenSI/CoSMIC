# -------------------------------------------------------------------------------------------------------------
# File: qa.py
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

import os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.services import chess as chess_instances
from src.services.base import ServiceBase
from src.services.llms.llm import LLMBase
from src.services.rag import RAGBase

# =============================================================================================================

class QABase(ServiceBase):
    def __init__(
        self,
        query_analyser: LLMBase,
        llm: LLMBase,
        rag: RAGBase,
        **kwargs
    ):
        """Base class for QA.

        Args:
            query_analyser (LLMBase): query analyser.
            llm (LLMBase): LLM instance.
            rag (RAGBase): RAG instance containing vector database service.
        """
        super().__init__( **kwargs)

        # Set config globally.
        self.query_analyser = query_analyser
        self.llm = llm
        self.rag = rag

    def __call__(
        self,
        query: str,
        context: str="",
        is_rag: bool=False,
        verbose: bool=False
    ):
        """Process each QA.

        Args:
            query (str): a question.
            context (str|dict, optional): contex associated with the question. Defaults to "".
            is_rag (bool, optional): if retrieve context for the question. Defaults to False.
            verbose (bool, optional): debug mode. Default to False.

        Returns:
            response (str): truncated answer if applicable.
            raw_response (str): original answer from LLM.
            retrieve_score: score of context retrieving if applicable.
        """
        # Set initial return answers.
        response = None
        raw_response = None
        retrieve_score = -1

        # Get service option through query analyser.
        service_option, service_info_dict = self.query_analyser(query)

        # Whether this query is related to system information.
        system_information_relevance = service_info_dict["system_information_relevance"]

        # Skip query as required or unknown service option.
        if query.find("skip") > -1:
            return response, raw_response, retrieve_score

        # Process query with service parsing.
        if service_option == "0.0":
            # Set game move mode.
            if context == "": move_mode = "algebric"
            else: move_mode = context

            # Get chess FEN.
            current_fen = service_info_dict["fen"]

            # Set up next move predictor as Stockfish.
            next_move_predictor = chess_instances.StockfishFENNextMove()

            # Predict the next move.
            next_move = next_move_predictor(fen=current_fen, move_mode=move_mode, topk=5)

            # Set the response with question and next move.
            response = f"The next move of {[current_fen]} is one of {next_move}."
        elif service_option == "0.1":
            # Set game move mode.
            if context == "": move_mode = "algebric"
            else: move_mode = context

            # Get moves.
            current_moves = service_info_dict["moves"]

            # Set up next move predictor as Stockfish.
            next_move_predictor = chess_instances.StockfishSequenceNextMove()

            # Predict the next move.
            next_move = next_move_predictor(current_moves, move_mode=move_mode, topk=5)

            # Set the response with question and next move.
            response = f"The next move of {[current_moves]} is one of {next_move}."
        elif service_option == "1":
            # Check if context is a .pdf.
            is_a_document = service_info_dict["is_a_document"]

            if is_a_document:
                # Get absolute document path.
                document_path = service_info_dict["document_path"]

                if document_path is not None:
                    # Update the knowledge database and return the status.
                    self.rag.vector_database.update_database_from_document(document_path=document_path)
            else:
                # Get text.
                text = service_info_dict["text"]

                if text is not None:
                    # Add text to database.
                    self.rag.vector_database.update_database_from_text(text=text)
        elif service_option == "2":
            response = "Code generation to be implemented."
        else:
            if is_rag:
                # If retrieving context, first generate the user prompt given the
                # user prompter format.
                user_prompt = self.llm.user_prompter(query, context=context)

                # Get the retrieved context.
                context_retrieved, retrieve_score = self.rag(query)

                # Remain the other variables in context if it is a dictionary,
                # otherwise overwrite it.
                if isinstance(context, dict):
                    context["context"] = context_retrieved
                else:
                    context = context_retrieved
            else:
                user_prompt = query
                retrieve_score = -1

            # If the question is related to system information,
            # add system information to context.
            if system_information_relevance:
                # Get system information.
                system_information = service_info_dict["system_information"]

                # Add the information to existing context.
                if isinstance(context, dict):
                    context["context"] += system_information
                else:
                    context += system_information

            # Get the response with retrieved context if applicable.
            response, raw_response = self.llm(user_prompt, context=context)

        # Print service name.
        if verbose \
            and (response is not None) \
            and (service_option in self.query_analyser.full_services.keys()):
            response += f" [service: {self.query_analyser.full_services[service_option]}" \
                f"; system info relevance: {system_information_relevance}]"

        return response, raw_response, retrieve_score