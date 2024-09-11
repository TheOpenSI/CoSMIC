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
        llm: LLMBase=None,
        rag: RAGBase=None,
        **kwargs
    ):
        """Base class for QA.

        Args:
            llm (LLMBase): LLM instance.
            rag (RAGBase): RAG instance containing vector database service.
        """
        super().__init__( **kwargs)

        # Set config globally.
        self.llm = llm
        self.rag = rag

    def __call__(
        self,
        query: str,
        context: str="",
        is_rag: bool=False
    ):
        """Process each QA.

        Args:
            query (str): a question.
            context (str|dict, optional): contex associated with the question. Defaults to "".
            is_rag (bool, optional): if retrieve context for the question. Defaults to False.

        Returns:
            response (str): truncated answer if applicable.
            raw_response (str): original answer from LLM.
            retrieve_score: score of context retrieving if applicable.
        """
        # Set initial return answers.
        response = None
        raw_response = None
        retrieve_score = -1

        # Process query with keyword recogintion.
        if query.find("exit") > -1:
            response = "exit"
        elif query.find("skip") > -1:
            response = None
        elif query.find("__next__move__") > -1:
            # Parse move string
            current_move = query.split("__next__move__")[-1]

            if context == "":
                move_mode = "algebric"
            else:
                move_mode = context

            # Set up next move predictor as Stockfish.
            next_move_predictor = chess_instances.StockfishSequenceNextMove()

            # Predict the next move.
            next_move = next_move_predictor(current_move, move_mode=move_mode, topk=5)

            # Set the response with question and next move.
            response = f"The next move of {[current_move]} is one of {next_move}."
        elif query.find("__update__store__") > -1:
            context = query.split("__update__store__")[-1]

            # Check if context is a .pdf.
            is_context_a_document = context.find(".pdf") > -1

            if is_context_a_document:
                # Remove empty space to generate the absolute document path.
                context = context.replace(" ", "")

                # Check if not an absolute path, convert to an absoluate path.
                if not os.path.isabs(context):
                    context = os.path.join(self.root, context)

                # Update the knowledge database and return the status.
                self.rag.vector_database.update_database_from_document(document_path=context)
            else:
                # Add text to database.
                self.rag.vector_database.update_database_from_text(text=context)
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

            # Get the response with retrieved context if applicable.
            response, raw_response = self.llm(user_prompt, context=context)

        return response, raw_response, retrieve_score