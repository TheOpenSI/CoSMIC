import torch, os, pytz, glob

from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from peft import PeftModel
from openai import OpenAI
from utils.log_tool import set_color
from .model_prompt import get_llm_reader, extract_answer_from_response, extract_chess_answer_from_response
# from .load_model_test import load_model_external


# =============================================================================================================

# Set model specific dictionary
LLM_MODEL_DICT = {
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-7b-it",  # bad
    "mistral-7b-finetuned": "adnaan525/opensi_mistral_3tasks",
    "mistral-7b-finetuned-new": "OpenSI/cognitive_AI",
    "gpt-4o": "gpt-4o"
}

# =============================================================================================================

class LLMEngine:
    def __init__(
        self,
        llm_model='mistral-7b-v0.1',
        document_analyser_model='gte-small',
        retrieve_score_threshold=0.,
        seed=0
    ):
        # Set config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.retrieve_score_threshold = retrieve_score_threshold
        self.llm_model = llm_model.lower()
        self.seed = seed

        # Automatically set back_end and whether use prompt example
        if self.llm_model in ["mistral-7b-instruct-v0.1", "gemma-7b-it", "gpt-4o"]:
            self.back_end = "chat"
            self.prompt_example = False  # this will not affect chat mode which has no prompt
        else:
            self.back_end = "instance"

            if self.llm_model.find('finetune') > -1:
                # Finetuned model always input question and context even if context is empty
                self.prompt_example = False
            else:
                self.prompt_example = True  # changable, better to switch on to truncate the the response with keywords

        # Check if LLM model is supported
        assert self.llm_model in LLM_MODEL_DICT.keys(), \
            print(set_color("error", f"LLM model {self.llm_model} is not supported."))

        # Check valid engine mode
        if self.back_end not in ['instance', 'chat']:
            set_color('error', f"Back-end {self.back_end} is not supported.")

        # Login only when the model is not downloaded to .cache
        if self.llm_model.find('gpt') <= -1:
            cache_model_name = "models--" + LLM_MODEL_DICT[self.llm_model].replace('/', '--')
            cache_model_directory = os.path.join(os.path.expanduser("~"), '.cache/huggingface/hub')
            cache_model_path = os.path.join(cache_model_directory, cache_model_name)
            if not os.path.exists(cache_model_path): self.login()

        # Set a time stamp, day is not accurate, so remove
        self.time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%B, %Y")

        # -----------------------------------------------------------------------------------------------------
        # For LoRA finetuned model
        if self.llm_model.find('mistral') > -1 and self.llm_model.find('finetune') > -1:
            # Model and tokenizer need to be based model config
            base_llm_model = "mistral-7b-v0.1"

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_DICT[base_llm_model],
                quantization_config=bnb_config,
                # low_cpu_mem_usage=True
                use_cache=True,
                device_map="auto"
            )

            self.model = PeftModel.from_pretrained(
                base_model,
                LLM_MODEL_DICT[self.llm_model]
            )

            # Set tokenizer
            tokenizer_cls = LLM_MODEL_DICT[base_llm_model]
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_cls,
                add_bos_token=True
            )
        elif self.llm_model.find('gpt') > -1:
            # Cannot hard-code this key, as forbidden by GitHub
            api_key = ""
            self.model = OpenAI(api_key=api_key)
            tokenizer = None
        else:
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            # )

            # self.model = AutoModelForCausalLM.from_pretrained(
            #     LLM_MODEL_DICT[self.llm_model],
            #     quantization_config=bnb_config,
            #     # low_cpu_mem_usage=True
            #     use_cache=True,
            #     device_map="cuda"
            # )

            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_DICT[self.llm_model],
                # low_cpu_mem_usage=True
                use_cache=True,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )

            # Set tokenizer
            tokenizer_cls = LLM_MODEL_DICT[self.llm_model]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_cls)

        # Suppress the warning from
        # https://stackoverflow.com/questions/74682597/
        # fine-tuning-gpt2-attention-mask-and-pad-token-id-errors
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set LLM model for different uses
        if self.back_end == 'chat':
            # Set prompt templates for those without/with context
            prompt_template = "{question}"

            prompt_template_context = \
                "Given that '{context}', {question}"
        else:
            if self.llm_model.find('mistral') > -1:
                if self.llm_model.find('finetune') > -1:
                    prompt_template = prompt_template_context = \
                        "<s>### Instruction:\n{question}\n### Context: \n{context}\n### Response:"
                else:
                    # Set prompt templates for those without/with context
                    if self.prompt_example:
                        # For Mistral, https://www.promptingguide.ai/models/mistral-7b
                        prompt_template = \
                            "<s> [INST] What is the capital of China? [/INST]\n" \
                            "Beijing</s>\n" \
                            "[INST] {question} [/INST]"

                        prompt_template_context = \
                            "<s> [INST] Given that 'Beijing is the capital of China'," \
                            " what is the capital of China? [/INST]\n" \
                            "Beijing</s>\n" \
                            "[INST] Given that '{context}', {question} [/INST]"
                    else:
                        # prompt_template = prompt_template_context = \
                        #     "<s>[INST] \n" \
                        #     "Instruction: Always answer the question even if the context isn't useful. \n" \
                        #     "Write a response that appropriately completes the request. Do not say anything unnecessary.\n" \
                        #     "Here is context to help -\n" \
                        #     "{context}\n\n" \
                        #     "### QUESTION:\n" \
                        #     "{question} \n\n" \
                        #     "[/INST]\n"

                        prompt_template = prompt_template_context = \
                            "<s> Always answer the question briefly even if the context isn't useful.\n" \
                            "Given the context: '{context}', the question is '{question}'"
            elif self.llm_model.find('gemma') > -1:
                if self.prompt_example:
                    # https://medium.com/@coldstart_coder/
                    # getting-started-with-googles-gemma-llm-using-huggingface-libraries-a0d826c552ae
                    # https://www.promptingguide.ai/models/gemma
                    prompt_template = \
                        "<bos><start_of_turn>user\n" \
                        "What is the capital of China?<end_of_turn>\n" \
                        "<start_of_turn>model\n" \
                        "Beijing<end_of_turn><eos>\n" \
                        "<start_of_turn>user\n" \
                        "{question}<end_of_turn>\n" \
                        "<start_of_turn>model"

                    prompt_template_context = \
                        "<bos><start_of_turn>user\n" \
                        "Given that 'Beijing is the capital of China'," \
                        " what is the capital of China?<end_of_turn>\n" \
                        "<start_of_turn>model\n" \
                        "Beijing<end_of_turn><eos>\n" \
                        "<start_of_turn>user\n" \
                        "Given that '{context}', {question}<end_of_turn>\n" \
                        "<start_of_turn>model"
                else:
                    # prompt_template = prompt_template_context = \
                    #     "<bos><start_of_turn>user\n" \
                    #     "Always answer the question even if the context isn't useful. \n" \
                    #     "Write a response that appropriately completes the request. Do not say anything unnecessary.\n" \
                    #     "Here is context to help -\n" \
                    #     "{context}\n\n" \
                    #     "### QUESTION:\n" \
                    #     "{question} \n\n<end_of_turn>" \
                    #     "<start_of_turn>model"

                    prompt_template = prompt_template_context = \
                        "<bos> Always answer the question briefly even if the context isn't useful.\n" \
                        "Given the context: '{context}', the question is '{question}'"

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
        if (self.llm_model.find('mistral') > -1) and (self.llm_model.find('finetune') > -1):
            chess_prompt_template = chess_prompt_context_template = \
                "<s>### Instruction:\nExplain the rationale behind {player}'s {move}? " \
                "\n### Context: \n### Response:"
        else:
            chess_prompt_template = \
                "Given chess board FEN '{fen}', explain briefly why {player} takes {move}?"

            chess_prompt_context_template = \
                "Given chess board FEN '{fen}' and context that '{context}', explain briefly why {player} takes {move}?" \
                " If the context is useless, ignore it."

        self.chess_prompt = PromptTemplate(
            input_variables=['player', 'move', 'fen'],
            template=chess_prompt_template,
        )

        self.chess_prompt_context = PromptTemplate(
            input_variables=['player', 'move', 'fen', 'context'],
            template=chess_prompt_context_template,
        )

        # -----------------------------------------------------------------------------------------------------
        # Get LLM reader with inbuilt query encoder
        self.llm_reader = get_llm_reader(self.llm_model, tokenizer, self.model)

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
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_DICT[document_analyser_model]

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

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_back_end(self):
        return self.back_end

    def set_back_end(self, back_end):
        self.back_end = back_end

    def set_retrieve_score_threshold(self, retrieve_score_threshold):
        self.retrieve_score_threshold = retrieve_score_threshold

    def get_retrieve_score_threshold(self):
        return self.retrieve_score_threshold

    def quit(self):
        if self.llm_model.find('gpt') > -1:
            # No model is loaded for OpenAI but simply call the API
            self.model.close()
        else:
            self.model = self.model.to('cpu')

        del self.model
        del self.database_update_embedding
        torch.cuda.empty_cache()

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
        retrieved_docs_score = []

        for doc, score in retrieved_contents:
            # Filter out low confidence context
            if score >= self.retrieve_score_threshold:
                retrieved_docs_text.append(doc.page_content)
            else:
                retrieved_docs_text.append(None)

            # Store all the scores
            retrieved_docs_score.append(score)

        if len(retrieved_docs_text) == 0:
            context = ''
        else:
            # Not use change line for each document
            if False:
                context = "\nExtracted documents:\n"
                context += "".join(
                    [f"\nDocument {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
                )
            else:
                context = "".join([
                    f"Document {str(i)}: " + doc.replace('\n', ' ') + '. ' \
                    for i, doc in enumerate(retrieved_docs_text) if (doc is not None)
                ])

        return context, retrieved_docs_score

    def generate_chess_analysis_prompt(self, player, move, fen, context=''):
        # Chess analysis has specific prompt template and instance
        if self.prompt_example:
            if context == '':
                prompt = self.chess_prompt.format(
                    player=player,
                    move=move,
                    fen=fen
                )
            else:
                prompt = self.chess_prompt_context.format(
                    player=player,
                    move=move,
                    fen=fen,
                    context=context
                )
        else:
            prompt = self.chess_prompt_context.format(
                player=player,
                move=move,
                fen=fen,
                context=context
            )

        return prompt

    def generate_prompt(self, question, context=''):
        # Assume that if context is empty, the prompt will ignore it
        if self.prompt_example:
            if context == '':
                prompt = self.prompt.format(question=question)
            else:
                prompt = self.prompt_context.format(
                    question=question,
                    context=context
                )
        else:  # without example, always use prompt with context, although context is useless
            prompt = self.prompt_context.format(
                question=question,
                context=context
            )

        return prompt

    def chess_analysis(self, player, move, fen, is_rag, topk=1):
        # Get the question
        prompt = self.generate_chess_analysis_prompt(player, move, fen)

        # Retrieve context from stored database
        if is_rag:
            context, _ = self.retrieve_context(prompt, topk=topk)
            prompt = self.generate_chess_analysis_prompt(player, move, fen, context=context)

        # Set seed to make the response deterministric
        self.set_seed(self.seed)

        # Generate the answer
        raw_analysis = self.llm_reader(prompt)

        # Extract the key answer
        analysis = extract_chess_answer_from_response(self.llm_model, raw_analysis, self.prompt_example)

        return analysis

    def __call__(
            self,
            user_query,
            context,
            topk=1,
            is_cotext_a_document=False,
            update_database_only=False,
            is_rag=False
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
            context_score = None
            raw_answer = None
        else:
            # Retrieve context from stored database
            if is_rag:
                context, context_score = self.retrieve_context(user_query, topk=topk)
            else:
                context = ''
                context_score = [-1]

            # Generate prompt
            prompt = self.generate_prompt(question=user_query, context=context)

            # Set seed
            self.set_seed(self.seed)

            # Generate the answer
            raw_answer = self.llm_reader(prompt)

            # Parse answer
            answer = extract_answer_from_response(self.llm_model, raw_answer, self.prompt_example)

        return answer, context_score, raw_answer