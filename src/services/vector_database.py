### Core modules ###
from pathlib import Path
from csv import writer
from glob import glob
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader


### Type hints ###


### Internal modules ###
from ...utils.log_tool import set_color
from .base import ServiceBase


class VectorDatabase(ServiceBase):
    def __init__(
        self,
        document_analyser_model: str = "gte-small",
        # document_analyser_model: str = "Qwen/Qwen3-Embedding-8B",
        local_database_path: str = "database/vector_database",
        vector_database_update_threshold: float = 0.98,
        device: str = "cuda",
        **kwargs
    ):
        """
        Vector database service.

        Args:
            document_analyser_model             (str, optional):    document analyser/process model.
            local_database_path                 (str, optional):    path of local vector database on disk.
                                                                    Default to "database/vector_database".
            vector_database_update_threshold    (float, optional):  contents with similarity >= this threshold
                                                                    will be skipped. Default to 0.98.
            device                              (str, optional):    use cuda or cpu for LLM. Defaults to "cuda".
                                                                    Defaults to "gte-small".
        """
        super().__init__(**kwargs)

        # Set config.
        # Set to absolute path.
        self.local_database_path: Path = Path(local_database_path).resolve(strict=True)

        if local_database_path != "" and not self.local_database_path.is_absolute():
            self.local_database_path: Path = self.root.joinpath(self.local_database_path)

        # Use default one.
        if not self.local_database_path.exists(follow_symlinks=True):
            if self.local_database_path != "":
                print(
                    set_color(
                        status="warning",
                        information="{0:s}{1:s}".format(
                            f"Vector database \"{local_database_path}\" not exist",
                            f", use default \"database/vector_database\"."
                        )
                    )
                )

            self.local_database_path: Path = self.root.joinpath("database/vector_database")

        # Get the catalogue path and threshold.
        self.current_local_database_path: Path = self.local_database_path
        self.local_database_catalogue_path: Path = self.current_local_database_path.joinpath("file_list.csv")
        self.vector_database_update_threshold = vector_database_update_threshold

        # Create local database directory.
        if self.current_local_database_path != "":
            local_database_name = str(object=self.current_local_database_path).split("/")[-1]
            local_database_directory = str(object=self.current_local_database_path).replace(
                f"/{local_database_name}",
                ""
            )
            Path(local_database_directory).resolve(strict=True).mkdir(
                mode=0o777,
                parents=False,
                exist_ok=True
            )

        # Write head in catalogue file.
        if not self.local_database_catalogue_path.exists(follow_symlinks=True):
            with self.local_database_catalogue_path.open(
                mode="w",
                buffering=-1,
                encoding="utf-8",
                errors=None,
                newline=None
            ) as catalogue_pt:
                catalogue = writer(catalogue_pt)
                catalogue.writerow(
                    [
                        "Source",
                        "Time",
                        "Comment"
                    ]
                )

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
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Build processor to handle a new document for database updates.
        # Find the API at https://api.python.langchain.com/en/latest/vectorstores
        # /langchain_community.vectorstores.faiss.FAISS.html
        # Build a processor to handle a sentence for database updates.

        # Load a local database from a file
        if Path(self.current_local_database_path / "index.faiss").exists(follow_symlinks=True):
            self.database = FAISS.load_local(
                folder_path=local_database_path,
                embeddings=self.database_update_embedding,
                index_name="index",
                allow_dangerous_deserialization=True
            )

            print(
                set_color(
                    status="success",
                    information=f"Load \"{str(object=self.current_local_database_path)}\" to vector database."
                )
            )
        else:
            self.database = FAISS.from_texts(
                texts=["Use FAISS as database updater"],
                embedding=self.database_update_embedding,
                metadatas=None,
                ids=None
            )

        # Set search strategy.
        self.database.distance_strategy = DistanceStrategy.COSINE

        # Set a time stamp to highlight the most recently updated information.
        self.time_stamper:              str = datetime.now(tz=ZoneInfo(key="Australia/Sydney")).strftime(format="%B, %Y")

        # Set a time stamp to update vector database catalogue.
        self.catalogue_time_stamper:    str = datetime.now(tz=ZoneInfo(key="Australia/Sydney")).strftime(format="%m/%d/%Y, %H:%M:%S")


    def similarity_search_with_relevance_scores(
        self,
        *args,
        **kwargs
    ):
        """
        Retriever from the vector database.

        Returns:
            context (str): retrieved information.
        """
        return self.database.similarity_search_with_relevance_scores(*args, **kwargs)


    def quit(self):
        """
        Release document analyser model.
        """
        if self.database_update_embedding:
            del self.database_update_embedding


    def add_documents(
        self,
        document_paths
    ):
        """
        Add context from a document or multiple documents to the vector database.

        Args:
            document_paths (string or list): a document path or multiple such paths.
        """
        print("Adding documents to vector database...")

        # Set as a list for loop.
        if not isinstance(document_paths, list):
            document_paths = [document_paths]

        # Update per document.
        for document_path in document_paths:
            if not Path(document_path).resolve(strict=True).exists(follow_symlinks=True):
                continue

            print(f"{document_path=}")
            self.update_database_from_document(document_path)


    def add_document_directory(
        self,
        document_dir: str
    ):
        """
        Add all .pdf in a folder to the vector database.

        Args:
            document_dir (str): a directory of .pdf to be added to the vector database.
        """
        self.document_dir: Path = Path(document_dir).resolve(strict=True)

        if self.document_dir.exists(follow_symlinks=True):
            # Find all pdf in a folder.
            document_paths = glob(f"{str(object=self.document_dir)}/*.pdf")

            # Add these documents.
            self.add_documents(document_paths)


    def update_database_catalogue(
        self,
        metadata: str
    ):
        """
        Update catalogue of vector database.

        Args:
            metadata (str|list): contents to be added.
        """
        # Set as a list for loop.
        if not isinstance(metadata, list):
            metadatas: list[str] = [metadata]
        else:
            metadatas: list[str] = metadata

        # Open the catalogue file.
        with self.local_database_catalogue_path.open(
            mode="a",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as catalogue_pt:
            catalogue = writer(catalogue_pt)

            # Write metadata.
            for data in metadatas:
                if isinstance(data, str):
                    catalogue.writerow(
                        [
                            data,
                            self.catalogue_time_stamper,
                            ""
                        ]
                    )


    def update_database_from_document(
        self,
        document_path: str
    ):
        """
        Add a document to the vector database.

        Args:
            document_path (str): a document path.
        """
        # Check if the document exists.
        self.document_path: Path = Path(document_path).resolve(strict=True)

        if self.document_path.exists(follow_symlinks=True):
            document_title = self.document_path.stem

            # Read pages of a document.
            loader = PyPDFLoader(self.document_path)
            pages = loader.load_and_split() # split by page number

            for i in range(len(pages)):
                pages[i].page_content = pages[i].page_content.replace("\t", " ")

            # Split each page into tokens.
            document_processed = []

            for doc in pages:
                # Attach per-document metadata (propagates to all chunks).
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["title"] = document_title

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
                chunks = self.document_splitter.split_documents([doc])
                for chunk in chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata.setdefault("title", document_title)
                document_processed += chunks

            # Obtain new knowledge from the splitted tokens.
            if len(document_processed) > 0:  # for invalid pdf such as a scanned .pdf
                self.database.add_documents(document_processed)

                # Update database catalogue.
                self.update_database_catalogue(document_path)

                # Save to local database.
                self.database.save_local(str(object=self.current_local_database_path))

                print(
                    set_color(
                        status="info",
                        information=f"Add '{str(object=self.document_path)}'."
                    )
                )
            else:
                print(
                    set_color(
                        status="warning",
                        information=f"Contents of '{str(object=self.document_path)}' exist."
                    )
                )
        else:
            print(
                set_color(
                    status="warning",
                    information=f"Document {str(object=document_path)} not exists."
                )
            )


    def update_database_from_text(
        self,
        text: str
    ):
        """
        Add a sentence to the vector database.

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
            text = f"{text} by the date {self.time_stamper}"

            # Add text to database.
            self.database.add_texts([text])

            # Update database catalogue.
            self.update_database_catalogue(text)

            # Save to local file.
            self.database.save_local(str(object=self.current_local_database_path))

            # Print the progress.
            print(
                set_color(
                    status='info',
                    information=f"Update database with '{text}'."
                )
            )

            return 0
