import torch, os

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from huggingface_hub import login


# =============================================================================================================

class LLMEngine():
    def __init__(
        self,
        model_name='mistral',
        prompt_variables='',
        prompt_template=''
    ):
        # Set config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."

        # Set model options
        model_dict = {
            'mistral': "mistralai/Mistral-7B-v0.1"
        }

        # Login
        self.login()

        # Build LLM model and tokenizer
        READER_MODEL_NAME = model_dict[model_name]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Set model
        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME,
            quantization_config=bnb_config
        )

        # Set tokenizer
        tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

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

    def login(self):
        # Set the token stored file
        load_dotenv(f"{self.root}/finetune/.env")

        # Variable hf_token2 stores huggingface token
        login(os.getenv('hf_token2'), add_to_git_credential=True)

    def run(self, user_query, context):
        # Set values to prompt
        query = self.prompt.format(question=user_query, context=context)

        # Get the generated text
        answer = self.llm_reader(query)[0]["generated_text"]

        return answer