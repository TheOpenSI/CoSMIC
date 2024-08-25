# -------------------------------------------------------------------------------------------------------------
# File: rag.py
# Project: OpenSI AI System
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

from src.services.base import ServiceBase
from src.services.vector_database import VectorDatabase

# =============================================================================================================

class RAGBase(ServiceBase):
    def __init__(
        self,
        vector_database: VectorDatabase,
        retrieve_score_threshold: float=0.7,
        topk: int=5,
        **kwargs
    ):
        """Context retriever service.

        Args:
            vector_database (VectorDatabase): vector database.
            retrieve_score_threshold (float, optional): retrieve score threshold to filter out retrieved
            context with similarity under this threshold. Defaults to 0.7.
            topk (int, optional): up to topk retrieved context returned. Defaults to 5.
        """
        super().__init__(**kwargs)

        # Set config.
        self.retrieve_score_threshold = retrieve_score_threshold
        self.topk = topk
        self.vector_database = vector_database

    def set_vector_database(
        self,
        vector_database: VectorDatabase
    ):
        """Set the vector database externally on demand.

        Args:
            vector_database (VectorDatabase): an external vector database.
        """
        self.vector_database = vector_database

    def set_retrieve_score_threshold(
        self,
        retrieve_score_threshold: float
    ):
        """Change the retrieve score externally on demand.

        Args:
            retrieve_score_threshold (float): an external threshold.
        """
        self.retrieve_score_threshold = retrieve_score_threshold

    def set_topk(
        self,
        topk: int
    ):
        """Change the topk externally on demand.

        Args:
            topk (int): topk documents to be retrieved.
        """
        self.topk = topk

    def __call__(
        self,
        user_prompt: str
    ):
        """Retrieve context for a given user prompt.

        Args:
            user_prompt (str): a question from the user.

        Returns:
            context (str): retrieved context from the vector database.
            retrieved_doc_score: score of the retrieved context.
        """
        # Find the topk relevant tokens.
        retrieved_contents = self.vector_database.similarity_search_with_relevance_scores(
            query=user_prompt,
            k=self.topk
        )

        # Store the retrieved page contents and page scores.
        retrieved_context = []
        retrieved_context_score = []

        for doc, score in retrieved_contents:
            # Filter out low confidence context.
            if score >= self.retrieve_score_threshold:
                retrieved_context.append(doc.page_content)
            else:
                retrieved_context.append(None)

            # Store all the scores.
            retrieved_context_score.append(score)

        if len(retrieved_context) == 0:
            context = ""
        else:
            # Change the linechange to avoid messing up the print and log file.
            context = "".join([
                f"Document {str(i)}: " + doc.replace("\n", " ") + ". " \
                for i, doc in enumerate(retrieved_context) if (doc is not None)
            ])

        return context, retrieved_context_score