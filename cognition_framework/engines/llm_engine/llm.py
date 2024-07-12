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
from utils.log_tool import set_color


# =============================================================================================================

class LLMEngine:
    def __init__(
        self,
        llm_model_name='mistral',
        document_analyser_model_name='gte-small',
        retrieve_score_threshold=0.,
        back_end='instance'
    ):
        # Set config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.retrieve_score_threshold = retrieve_score_threshold
        self.back_end = back_end

        # Check valid engine mode
        if back_end not in ['instance', 'chat']:
            set_color('error', f"Back-end {back_end} is not supported.")

        # Login
        self.login()

        # Set a time stamp, day is not accurate, so remove
        self.time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%B, %Y")

        # -----------------------------------------------------------------------------------------------------
        # For LLM engine
        # Set model options
        LLM_MODEL_DICT = {
            'mistral': "mistralai/Mistral-7B-Instruct-v0.1"
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
            low_cpu_mem_usage=True
        )

        # Set tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            READER_MODEL_NAME,
        )

        # Suppress the warning from
        # https://stackoverflow.com/questions/74682597/
        # fine-tuning-gpt2-attention-mask-and-pad-token-id-errors
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set LLM model for different uses
        if self.back_end == 'chat':
            # This user-assisant chat instance could be more accessable
            # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
            # User-assistant chat template
            chat_template = lambda query: [{"role": "user", "content": query}]

            query_encoded = lambda query: \
                tokenizer.apply_chat_template(
                    chat_template(query),
                    return_tensors="pt"
                )

            self.llm_reader = lambda query: \
                tokenizer.batch_decode(
                    model.generate(
                        query_encoded(query).to('cuda'),
                        max_new_tokens=1000,
                        do_sample=False,
                        # temperature=0.2,
                        pad_token_id=tokenizer.pad_token_id
                    )
                )[0]

            # Set prompt templates for those without/with context
            prompt_template = "{question}"

            prompt_template_context = \
                "Given the context: '{context}', '{question}'. "
        else:
            # Build LLM reader
            self.llm_reader = lambda query: pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                # temperature=0.2,  # block do_sample thus remove this
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=500,
            )(query)[0]["generated_text"]

            # Set prompt templates for those without/with context
            # 1. Set up LLM prompt template
            prompt_template = \
                "<s> [INST] Answer the question: 'What is the capital of China?' [/INST]\n" \
                "The capital of China is Beijing.</s> \n" \
                "[INST] Answer the question: '{question}'\n [/INST]"

            # 2. Set up LLM prompt template with context
            prompt_template_context = \
                "<s> [INST] Given context: 'Beijing is the capital of China', answer the question: 'What is the capital of China?'. [/INST]\n" \
                "The capital of China is Beijing.</s> \n" \
                "[INST] Given context: '{context}', answer the question: '{question}'.\n [/INST]"

        # Set prompt instances for those without/with context
        # May check out this https://huggingface.co/jondurbin/bagel-34b-v0.2#prompt-formatting
        # https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146
        self.prompt = PromptTemplate(
            input_variables=['question'],
            template=prompt_template,
        )

        self.prompt_context = PromptTemplate(
            input_variables=['question', 'context'],
            template=prompt_template_context,
        )

        # Set prompt template and instance for chess move analysis
        prompt_template_chess_analysis = \
            "Explain with one reason as short as possible why {player} takes {move} given the chess board FEN '{fen}'?"

        # TODO
        # prompt_template_chess_analysis = \
        #     "Given chess board FEN '{fen}', tell me the number and name of pieces on the board?"

        self.prompt_chess_analysis = PromptTemplate(
            input_variables=['player', 'move', 'fen'],
            template=prompt_template_chess_analysis,
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

    def get_back_end(self):
        return self.back_end

    def set_back_end(self, back_end):
        self.back_end = back_end

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
                print(set_color('info', f"Add {document_path} to database."))

    def add_document_directory(self, document_dir):
        if os.path.exists(document_dir):
            # Find all pdf in a folder
            document_paths = glob.glob(f"{document_dir}/*.pdf")

            # Add these documents
            self.add_documents(document_paths)

            print(set_color('info', f"Add documents in {document_dir} to database."))

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
            print(set_color('warning', f"Document {document_path} not exists."))

    def update_database_from_text(self, text):
        if text != '':
            # Update the text with timestamp
            text = f"{text} by the date {self.time_stamper(datetime.now())}"

            # Add text to database
            self.database.add_texts([text])

        print(set_color('info', f"Update database from text."))

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
            context = "".join(
                [f"Document {str(i)}:" + doc + '\n' for i, doc in enumerate(retrieved_docs_text)]
            )

        return context

    def generate_chess_analysis_prompt(self, player, move, fen):
        # Chess analysis has specific prompt template and instance
        prompt = self.prompt_chess_analysis.format(
            player=player,
            move=move,
            fen=fen
        )

        return prompt

    def generate_prompt(self, question, context=''):
        if context == '':
            prompt = self.prompt.format(question=question)
        else:
            prompt = self.prompt_context.format(
                question=question,
                context=context
            )

        return prompt

    def chess_analysis(self, player, move, fen):
        # Specific to chess move analysis due to predefined prompt template
        prompt = self.generate_chess_analysis_prompt(player, move, fen)

        # Generate the answer
        analysis = self.llm_reader(prompt)

        # Remove the question
        analysis = analysis.split('[/INST]')[-1].replace('\n', '').replace('</s>', '')

        return analysis

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

            # Generate prompt
            prompt = self.generate_prompt(question=user_query, context=context)

            # Generate the answer
            answer = self.llm_reader(prompt)

            # Parse the answer
            if self.back_end == 'chat':
                answer = answer.split('[/INST]')[-1].split('</s>')[0].strip()
            else:
                answer = answer.split('[INST]')[0].strip()

        return answer