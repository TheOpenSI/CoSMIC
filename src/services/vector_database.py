# -------------------------------------------------------------------------------------------------------------
# File: vector_database.py
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

from ast import List
import os, glob, pytz, sys, csv

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
        # document_analyser_model: str="gte-small",
        document_analyser_model: str="Qwen/Qwen3-Embedding-8B",
        local_database_path: str="database/vector_database",
        vector_database_update_threshold: float=0.98,
        device: str="cuda",
        
        # paragraph grouping controls
        paragraph_separator: str = "\n\n",
        min_paragraph_chars: int = 30,
        target_group_chars: int = 1200,
        paragraph_group_overlap: int = 1,

        **kwargs
    ):
        """Vector database service.

        Args:
            document_analyser_model (str, optional): document analyser/process model.
            local_database_path (str, optional): path of local vector database on disk.
                Default to "database/vector_database".
            vector_database_update_threshold (float, optional): contents with similarity >= this threshold
                will be skipped. Default to 0.98.
            device (str, optional): use cuda or cpu for LLM. Defaults to "cuda".
            Defaults to "Qwen/Qwen3-Embedding-8B".
            
            paragraph_separator (str, optional): how to split paragraphs; default double newline.
            min_paragraph_chars (int, optional): paragraphs shorter than this are filtered out.
            target_group_chars (int, optional): approx char budget per grouped chunk.
            paragraph_group_overlap (int, optional): number of previous paragraphs to overlap between chunks.

        """
        super().__init__(**kwargs)

        # Set config.
        # Set to absolute path.
        if local_database_path != "" and not os.path.isabs(local_database_path):
            local_database_path = os.path.join(self.root, local_database_path)

        # Use default one.
        if not os.path.exists(local_database_path):
            if local_database_path != "":
                print(
                    set_color(
                        "warning",
                        f"Vector database \"{local_database_path}\" not exist" \
                        f", use default \"database/vector_database\"."
                    )
                )

            local_database_path = os.path.join(self.root, "database/vector_database")

        # Get the catalogue path and threshold.
        self.local_database_path = local_database_path
        self.local_database_catalogue_path = os.path.join(local_database_path, "file_list.csv")
        self.vector_database_update_threshold = vector_database_update_threshold

        # Create local database directory.
        if local_database_path != "":
            local_database_name = local_database_path.split("/")[-1]
            local_database_directory = local_database_path.replace(
                "/" + local_database_name,
                ""
            )
            os.makedirs(local_database_directory, exist_ok=True)

        # Write head in catalogue file.
        if not os.path.exists(self.local_database_catalogue_path):
            catalogue_pt = open(self.local_database_catalogue_path, "w")
            catalogue = csv.writer(catalogue_pt)
            catalogue.writerow(["Source", "Time", "Comment"])
            catalogue_pt.close()

        # Support two embedding models
        EMBEDDING_MODEL_DICT = {
            "gte-small": "thenlper/gte-small",
            "Qwen/Qwen3-Embedding-8": "Qwen/Qwen3-Embedding-8",
        }

        # Paragraph-grouping settings bound to self
        self.paragraph_separator = paragraph_separator
        self.min_paragraph_chars = min_paragraph_chars
        self.target_group_chars = target_group_chars
        self.paragraph_group_overlap = paragraph_group_overlap


        # Set splitter to split a document into pages.
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ""],
        )

        # Build a document analyser.
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_DICT.get(
            document_analyser_model, document_analyser_model
            )

        self.database_update_embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,  # TODO
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Build processor to handle a new document for database updates.
        # Find the API at https://api.python.langchain.com/en/latest/vectorstores
        # /langchain_community.vectorstores.faiss.FAISS.html
        # Build a processor to handle a sentence for database updates.

        # Load a local database from a file
        if os.path.exists(f"{local_database_path}/index.faiss"):
            self.database = FAISS.load_local(
                local_database_path,
                self.database_update_embedding,
                allow_dangerous_deserialization=True
            )

            print(
                set_color(
                    "success",
                    f"Load \"{os.path.abspath(local_database_path)}\" to vector database."
                )
            )
        else:
            self.database = FAISS.from_texts(
                ["Use FAISS as database updater"],
                self.database_update_embedding,
            )

        # Set search strategy.
        self.database.distance_strategy = DistanceStrategy.COSINE

        # Set a time stamp to highlight the most recently updated information.
        self.time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%B, %Y")

        # Set a time stamp to update vector database catalogue.
        self.catalogue_time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%m/%d/%Y, %H:%M:%S")

    ## new functions for better paragraph grouping
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into logical paragraphs, filter short ones, normalize whitespace."""
        # Normalize line endings
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = [p.strip() for p in t.split(self.paragraph_separator)]
        # Filter very short or empty paragraphs
        paras = [p for p in parts if len(p) >= self.min_paragraph_chars]
        return paras

    def _group_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Group consecutive paragraphs up to target_group_chars with paragraph_group_overlap."""
        groups = []
        i = 0
        while i < len(paragraphs):
            # Start group with optional overlap from previous
            start = max(0, i - self.paragraph_group_overlap) if groups else i
            group = []
            length = 0
            j = start
            while j < len(paragraphs) and length < self.target_group_chars:
                group.append(paragraphs[j])
                length += len(paragraphs[j]) + len(self.paragraph_separator)
                j += 1
            # Deduplicate if overlap pulled in already-consumed paragraphs
            group_text = self.paragraph_separator.join(group).strip()
            if not groups or groups[-1] != group_text:
                groups.append(group_text)
            # Advance by the size of the new segment without counting overlap
            i = max(i + max(1, (j - i)), j)
        return groups

    def paragraph_group_chunks(self, raw_text: str) -> List[str]:
        """Public API to produce paragraph-grouped chunks from raw text."""
        paras = self._split_into_paragraphs(raw_text)
        if not paras:
            # fallback to recursive splitter if the input has no paragraphs
            return self.document_splitter.split_text(raw_text)
        return self._group_paragraphs(paras)

    
    
    
    def similarity_search_with_relevance_scores(
        self,
        *args,
        **kwargs
    ):
        """Retriever from the vector database.

        Returns:
            context (str): retrieved information.
        """
        return self.database.similarity_search_with_relevance_scores(*args, **kwargs)

    def quit(self):
        """Release document analyser model.
        """
        if self.database_update_embedding:
            del self.database_update_embedding

    def add_documents(
        self,
        document_paths
    ):
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

    def add_document_directory(
        self,
        document_dir: str
    ):
        """Add all .pdf in a folder to the vector database.

        Args:
            document_dir (str): a directory of .pdf to be added to the vector database.
        """
        if os.path.exists(document_dir):
            # Find all pdf in a folder.
            document_paths = glob.glob(f"{document_dir}/*.pdf")

            # Add these documents.
            self.add_documents(document_paths)

    def update_database_catalogue(
        self,
        metadata: str
    ):
        """Update catalogue of vector database.

        Args:
            metadata (str|list): contents to be added.
        """
        # Set as a list for loop.
        if not isinstance(metadata, list):
            metadata = [metadata]

        # Open the catalogue file.
        catalogue_pt = open(self.local_database_catalogue_path, "a")
        catalogue = csv.writer(catalogue_pt)

        # Write metadata.
        for data in metadata:
            if isinstance(data, str):
                catalogue.writerow([
                    data,
                    self.catalogue_time_stamper(datetime.now()),
                    ""
                ])

        # Close the file.
        catalogue_pt.close()

    def update_database_from_document(
        self,
        document_path: str
    ):
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
                # If not highly similar to existing contents, add the content.
                content_retrieved, similarity_score = self.similarity_search_with_relevance_scores(
                    doc.page_content,
                    k=1
                )[0]

                # Skip if already in the database or has a high similiarity.
                if similarity_score >= self.vector_database_update_threshold \
                    or content_retrieved.page_content.find(doc.page_content) > -1 \
                    or doc.page_content.find(content_retrieved.page_content) > -1:
                    continue

                # Ready to add to the vector database.
                document_processed += self.document_splitter.split_documents([doc])

            # Obtain new knowledge from the splitted tokens.
            if len(document_processed) > 0:  # for invalid pdf such as a scanned .pdf
                self.database.add_documents(document_processed)

                # Update database catalogue.
                self.update_database_catalogue(document_path)

                # Save to local database.
                self.database.save_local(self.local_database_path)

                print(set_color("info", f"Add '{document_path}'."))
            else:
                print(set_color("warning", f"Contents of '{document_path}' exist."))
        else:
            print(set_color("warning", f"Document {document_path} not exists."))

    def update_database_from_text(
        self,
        text: str
    ):
        """Add a sentence to the vector database.

        Args:
            text (str): a text sentence.

        Returns:
            status (int): skip (-1) or not (0).
        """
        if text != '':
            # Skip for high-similar text.
            content_retrieved, _ = self.similarity_search_with_relevance_scores(text, k=1)[0]
            content_retrieved = content_retrieved.page_content

            # If the same as existing contents, skip the text.
            if content_retrieved.find(text) > -1:
                print(set_color(
                    "warning",
                    f"Similar contents found: '{content_retrieved}' for '{text}'."
                ))

                return -1

            # Update the text with timestamp.
            text = f"{text} by the date {self.time_stamper(datetime.now())}"

            # Add text to database.
            self.database.add_texts([text])

            # Update database catalogue.
            self.update_database_catalogue(text)

            # Save to local file.
            self.database.save_local(self.local_database_path)

            # Print the progress.
            print(set_color('info', f"Update database with '{text}'."))

            return 0