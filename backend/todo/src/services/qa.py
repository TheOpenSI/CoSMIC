# -------------------------------------------------------------------------------------------------------------
# File: qa.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
#     Bing Tran <binhsan1307@gmail.com> (2026)
#
# Copyright (c) 2024 - 2026 Open Source Institute
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

from . import chess as chess_instances
from .base import ServiceBase
from .llms.llm import LLMBase
from .rag import RAGBase
from modules.code_generation.code_generation import CodeGenerator
from box import Box

# =============================================================================================================

class QABase(ServiceBase):
    def __init__(
        self,
        query_analyser: LLMBase,
        llm: LLMBase,
        rag: RAGBase,
        code_generator: CodeGenerator,
        config: Box | None = None,
        **kwargs
    ):
        """Base class for QA.

        Args:
            query_analyser (LLMBase): query analyser.
            llm (LLMBase): LLM instance.
            rag (RAGBase): RAG instance containing vector database service.
            code_generator (CodeGenerator): code generation service.
            config (Box): config file to extract settings. Default to None.
        """
        super().__init__( **kwargs)

        # Set config globally.
        self.query_analyser = query_analyser
        self.llm = llm
        self.rag = rag
        self.code_generator = code_generator
        self.config = config

    def __call__(
        self,
        query: str,
        context: str = "",
        is_rag: bool = False,
        verbose: bool = False
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
        if service_option.find("0.") > -1:
            if service_option == "0.0":
                # Set game move mode.
                if context == "": move_mode = "algebric"
                else: move_mode = context

                # Get chess FEN.
                current_fen = service_info_dict["fen"]

                # Set up next move predictor as Stockfish.
                binary_path = self.config.chess.stockfish_path if self.config else ""
                next_move_predictor = chess_instances.StockfishFENNextMove(binary_path=binary_path)

                # Predict the next move.
                next_move = next_move_predictor(fen=current_fen, move_mode=move_mode, topk=5)

                # Set the response with question and next move.
                move_prediction_context = f"The current chess FEN is {[current_fen]}."
            else:  # this is for prediction given moves, service_option == "0.1":
                # Set game move mode.
                if context == "": move_mode = "algebric"
                else: move_mode = context

                # Get moves.
                current_moves = service_info_dict["moves"]

                # Set up next move predictor as Stockfish.
                binary_path = self.config.chess.stockfish_path if self.config else ""
                next_move_predictor = chess_instances.StockfishSequenceNextMove(binary_path=binary_path)

                # Predict the next move.
                next_move = next_move_predictor(current_moves, move_mode=move_mode, topk=5)

                # Set the response with question and next move.
                move_prediction_context = f"The previous chess moves are {[current_moves]}."

            # Explain why these moves are feasible.
            user_prompt = f"Select the best next move from {next_move} and explain why it is the best."
            response, raw_response = self.llm(user_prompt, context=move_prediction_context)

            # Attach all the moves in case LLM cannot select the best one.
            response = f"The next moves are from {next_move}.\n{response}"
            raw_response = f"The next moves are from {next_move}.\n{raw_response}"
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

            response = raw_response = "Vector database updated."

        elif service_option == "2":
            raw_response, response = self.code_generator(query)

        else:
            # General question has RAG activated
            if is_rag:
                # Check if context is chat hostory
                chat_history_context = context if "Conversation History:" in context else ""
                rag_context = "" if "Conversation History:" in context else context

                # If retrieving context, first generate the user prompt given the
                # user prompter format.
                user_prompt = self.llm.user_prompter(query, context=rag_context)

                # Get the retrieved context.
                context_retrieved, retrieve_score = self.rag(query)

                # Remain the other variables in context if it is a dictionary,
                # otherwise overwrite it.
                suffix = "" \
                    if context_retrieved == "" \
                    else "\nContext:\n" + context_retrieved 

                if isinstance(context, dict):
                    context = str(
                        context.update(
                            {"context": chat_history_context + suffix}
                        )
                    )
                else:
                    context = chat_history_context + suffix

            else:
                user_prompt = query
                retrieve_score = -1

            # If the question is related to system information,
            # add system information to context.
            if system_information_relevance:
                # Get system information.
                system_information = service_info_dict["system_information"]

                # Add the information to existing context.
                # This context is likely to be chat history.
                if isinstance(context, dict):
                    context["context"] = "{0:s}{1:s}{2:s}".format(
                        "OpenSI System Information:\n",
                        f"{system_information}\n\n",
                        context["context"]
                    )
                else:
                    context = "{0:s}{1:s}{2:s}".format(
                        "OpenSI System Information:\n",
                        f"{system_information}\n\n",
                        context
                    )

            # Get the response with retrieved context if applicable.
            (response, raw_response) = self.llm(user_prompt, context={"response": context})

            # Print service name.
            if (
                (verbose)
                and
                (response is not None)
                and
                (service_option in self.query_analyser.full_services.keys())
            ):
                response += "[{0:s}{1:s}]".format(
                    f"service: {self.query_analyser.full_services[service_option]}",
                    f"; system info relevance: {system_information_relevance}"
                )

        return (response, raw_response, retrieve_score)
