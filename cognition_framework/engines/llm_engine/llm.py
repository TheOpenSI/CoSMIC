import torch, os, pytz, glob

from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader


# =============================================================================================================

class LLMEngine:
    def __init__(
        self,
        llm_model_name='mistral',
        document_analyser_model_name='gte-small',
        prompt_variables='',
        prompt_template='',
        retrieve_score_threshold=0.
    ):
        # Set config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.retrieve_score_threshold = retrieve_score_threshold

        # Login
        self.login()

        # Set a time stamp
        self.time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%d-%m, %Y")

        # -----------------------------------------------------------------------------------------------------
        # For LLM engine
        # Set model options
        LLM_MODEL_DICT = {
            'mistral': "mistralai/Mistral-7B-v0.1"
        }

        # Build LLM model and tokenizer
        READER_MODEL_NAME = LLM_MODEL_DICT[llm_model_name]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Set model
        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME,
            quantization_config=bnb_config,
        )

        # Set tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            READER_MODEL_NAME,
        )

        # Build LLM reader
        self.llm_reader = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            # temperature=0.2,  # block do_sample thus remove this
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )

        # Set prompt constructor
        self.prompt = PromptTemplate(
            input_variables=prompt_variables,
            template=prompt_template,
        )

        # -----------------------------------------------------------------------------------------------------
        # For document analysis and knowledge database generation/update
        EMBEDDING_MODEL_DICT = {
            'gte-small': "thenlper/gte-small"
        }

        # Set page separators
        MARKDOWN_SEPARATORS = ["\n\n", "\n", ""]

        # Set splitter to split a document into pages
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        # Build a document analyser
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_DICT[document_analyser_model_name]

        self.database_update_embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
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

    def set_retrieve_score_threshold(self, retrieve_score_threshold):
        self.retrieve_score_threshold = retrieve_score_threshold

    def get_retrieve_score_threshold(self):
        return self.retrieve_score_threshold

    def login(self):
        # Set the token stored file
        load_dotenv(f"{self.root}/finetune/.env")

        # Variable hf_token2 stores huggingface token
        login(os.getenv('hf_token2'), add_to_git_credential=True)

    def add_documents(self, document_paths):
        # Set as a list for loop
        if not isinstance(document_paths, list):
            document_paths = [document_paths]

        # Update per document
        for document_path in document_paths:
            if os.path.exists(document_path):
                self.update_database_from_document(document_path)
                print(f'!!! Success, add {document_path} to database.')

    def add_document_directory(self, document_dir):
        if os.path.exists(document_dir):
            # Find all pdf in a folder
            document_paths = glob.glob(f"{document_dir}/*.pdf")

            # Add these documents
            self.add_documents(document_paths)

            print(f'!!! Success, add documents in {document_dir} to database.')

    def update_database_from_document(self, document_path):
        # Check if the document exists
        if os.path.exists(document_path):
            # Read pages of a document
            loader = PyPDFLoader(document_path)
            pages = loader.load_and_split() # it splits by page number

            for i in range(len(pages)):
                pages[i].page_content = pages[i].page_content.replace("\t", " ")

            # Split each page into tokens
            document_processed = []

            for doc in pages:
                document_processed += self.document_splitter.split_documents([doc])

            # Obtain new knowledge from the splitted tokens
            if len(document_processed) > 0:  # for invalid pdf that is scanned
                self.database.add_documents(document_processed)
        else:
            print(f'!!! Warning document {document_path} not exists.')

    def update_database_from_text(self, text):
        if text != '':
            # Update the text with timestamp
            text = f"{text} by date {self.time_stamper(datetime.now())}"

            # Add text to database
            self.database.add_texts([text])

        print('!!! Success, update database from text.')

    def generate_text(self, query, context):
        # Set values to prompt
        query = self.prompt.format(question=query, context=context)

        # Get the generated text
        answer = self.llm_reader(query)[0]["generated_text"]

        return answer

    def retrieve_context(self, query, topk=1):
        # Find the topk relevant tokens
        retrieved_contents = self.database.similarity_search_with_relevance_scores(
            query=query,
            k=topk
        )

        # Store the retrieved page contents and page scores
        retrieved_docs_text = []

        for doc, score in retrieved_contents:
            # Filter out low confidence context
            if score >= self.retrieve_score_threshold:
                retrieved_docs_text.append(doc.page_content)

        if len(retrieved_docs_text) == 0:
            context = ''
        else:
            context = "\nExtracted documents:\n"
            context += "".join(
                [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
            )

        return context

    def __call__(
            self,
            user_query,
            context,
            topk=1,
            is_cotext_a_document=False,
            update_database_only=False
        ):
        # Get context from database
        if context != '':
            if is_cotext_a_document:
                # Update the knowledge database and return the status
                self.update_database_from_document(document_path=context)
            else:
                # Update text to database
                self.update_database_from_text(text=context)

        if update_database_only:
            # __update__store__ only updates the database without QA
            answer = None
        else:
            # Retrieve context from stored database
            context = self.retrieve_context(user_query, topk=topk)

            # Question answering from distilled knowledge of model or document context
            answer = self.generate_text(user_query, context)

        return answer