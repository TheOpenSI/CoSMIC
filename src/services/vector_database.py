# -------------------------------------------------------------------------------------------------------------
# File: vector_database.py
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

import os, glob, pytz, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from utils.log_tool import set_color
from src.services.base import ServiceBase

# =============================================================================================================

class VectorDatabase(ServiceBase):
    def __init__(
        self,
        document_analyser_model="gte-small",
        **kwargs
    ):
        """Vector database service.

        Args:
            document_analyser_model (str, optional): document analyser/process model.
            Defaults to "gte-small".
        """
        super().__init__(**kwargs)

        # For document analysis and knowledge database generation/update.
        EMBEDDING_MODEL_DICT = {'gte-small': "thenlper/gte-small"}

        # Set page separators.
        MARKDOWN_SEPARATORS = ["\n\n", "\n", ""]

        # Set splitter to split a document into pages.
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        # Build a document analyser.
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_DICT[document_analyser_model]

        self.database_update_embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,  # TODO
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Build processor to handle a new document for database updates.
        # Find the API at https://api.python.langchain.com/en/latest/vectorstores
        # /langchain_community.vectorstores.faiss.FAISS.html
        # Build a processor to handle a sentence for database updates.
        self.database = FAISS.from_texts(
            ["Use FAISS as database updater"],
            self.database_update_embedding,
            distance_strategy=DistanceStrategy.COSINE
        )

        # Set a time stamp to highlight the most recently updated information.
        self.time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%B, %Y")

    def similarity_search_with_relevance_scores(self, *args, **kwargs):
        """Retriever from the vector database.

        Returns:
            context (str): retrieved information.
        """
        return self.database.similarity_search_with_relevance_scores(*args, **kwargs)

    def quit(self):
        """Release document analyser model.
        """
        del self.database_update_embedding

    def add_documents(self, document_paths):
        """Add context from a document or multiple documents to the vector database.

        Args:
            document_paths (string or list): a document path or multiple such paths.
        """
        # Set as a list for loop.
        if not isinstance(document_paths, list):
            document_paths = [document_paths]

        # Update per document.
        for document_path in document_paths:
            if not os.path.exists(document_path): continue
            self.update_database_from_document(document_path)
            print(set_color('info', f"Add {document_path} to database."))

    def add_document_directory(self, document_dir):
        """Add all .pdf in a folder to the vector database.

        Args:
            document_dir (_type_): a directory of .pdf to be added to the vector database.
        """
        if os.path.exists(document_dir):
            # Find all pdf in a folder.
            document_paths = glob.glob(f"{document_dir}/*.pdf")

            # Add these documents.
            self.add_documents(document_paths)

            # Print the progress.
            print(set_color('info', f"Add documents in {document_dir} to database."))

    def update_database_from_document(self, document_path: str):
        """Add a document to the vector database.

        Args:
            document_path (str): a document path.
        """
        # Check if the document exists.
        if os.path.exists(document_path):
            # Read pages of a document.
            loader = PyPDFLoader(document_path)
            pages = loader.load_and_split() # split by page number

            for i in range(len(pages)):
                pages[i].page_content = pages[i].page_content.replace("\t", " ")

            # Split each page into tokens.
            document_processed = []

            for doc in pages:
                document_processed += self.document_splitter.split_documents([doc])

            # Obtain new knowledge from the splitted tokens.
            if len(document_processed) > 0:  # for invalid pdf such as a scanned .pdf
                self.database.add_documents(document_processed)
        else:
            print(set_color('warning', f"Document {document_path} not exists."))

    def update_database_from_text(self, text: str):
        """Add a sentence to the vector database.

        Args:
            text (str): a text sentence.
        """
        if text != '':
            # Update the text with timestamp.
            text = f"{text} by the date {self.time_stamper(datetime.now())}"

            # Add text to database.
            self.database.add_texts([text])

            # Print the progress.
            print(set_color('info', f"Update database from text."))