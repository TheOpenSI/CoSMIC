# -------------------------------------------------------------------------------------------------------------
# File: llm.py
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

import torch, os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
from src.maps import LLM_INSTANCE_DICT, LLM_MODEL_DICT
from src.llms.prompts import system_prompt as system_prompt_instances
from src.llms.prompts import user_prompt as user_prompt_instances
from src.llms import tokenizer as tokenizer_instances
from utils.module import get_instance

# =============================================================================================================

class LLMBase:
    def __init__(
        self,
        llm_name: str,
        user_prompt_instance_name: str="",
        system_prompt_instance_name: str="",
        use_example: bool=True,
        seed: int=0,
        is_truncate_response: bool=True,
    ):
        """LLM Base Class. Check the names from src/maps.py

        Args:
            llm_name (str): LLM base model name.
            user_prompt_instance_name (str, optional): user prompt instance name. Defaults to "".
            system_prompt_instance_name (str, optional): system prompt instance name. Defaults to "".
            use_example (bool, optional): use an example in system prompt. Defaults to True.
            seed (int, optional): seed for response generation. Defaults to 0.
            is_truncate_response (bool, optional): truncate the raw response. Defaults to True.
        """
        # Set config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."
        self.llm_name = llm_name
        self.use_example = use_example
        self.is_truncate_response = is_truncate_response
        self.seed = seed

        # Use user prompt for general questions if not specified.
        if user_prompt_instance_name == "":
            user_prompt_instance_name = "GeneralUserPrompt"

        # Build user prompter.
        self.set_user_prompter(user_prompt_instance_name)

        # Use system prompt by LLM type.
        if system_prompt_instance_name == "":
            system_prompt_instance_name = LLM_INSTANCE_DICT[llm_name]

        # Build system prompter.
        self.set_system_prompter_by_instance_name(
            system_prompt_instance_name,
            use_example=use_example
        )

        # Build tokenizer by LLM type.
        self.tokenizer = get_instance(
            tokenizer_instances,
            LLM_INSTANCE_DICT[llm_name]
        )(llm_name=llm_name)

        # Model and LLM are set from the children class by LLM type.
        self.model = None
        self.llm = None

    def set_user_prompter(
        self,
        user_prompt_instance_name: str,
        *args,
        **kwargs
    ):
        """Change user prompter externally.

        Args:
            user_prompt_instance_name (str): set an user prompt instance name.
        """
        self.user_prompter = get_instance(
            user_prompt_instances,
            user_prompt_instance_name
        )(*args, **kwargs)

    def set_system_prompter_by_instance_name(
        self,
        system_prompt_instance_name: str,
        *args,
        **kwargs
    ):
        """Change system prompter externally.

        Args:
            system_prompt_instance_name (str): set a system prompt instance name.
        """
        self.system_prompter = get_instance(
            system_prompt_instances,
            system_prompt_instance_name
        )(*args, **kwargs)

    def set_system_prompter(
        self,
        system_prompt_instance: system_prompt_instances.SystemPromptBase,
    ):
        self.system_prompter = system_prompt_instance

    def set_seed(
        self,
        seed: int
    ):
        """Set generation seed externally.

        Args:
            seed (int): generation seed before calling LLM model.
        """
        self.seed = seed

    def set_torch_seed(
        self,
        seed: int
    ):
        """Set PyTorch seed externally.

        Args:
            seed (int): seed for PyTorch program, CPU and GPU.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def set_truncate_response(
        self,
        is_truncate_response: bool
    ):
        """Set the flag of truncating response externally.

        Args:
            is_truncate_response (bool): truncate the response using key words in system prompt.
        """
        self.is_truncate_response = is_truncate_response

    def truncate_response(
        self,
        response: str
    ):
        """Truncate response.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        response = response.replace("\n", "").strip()

        return response

    def __call__(
        self,
        question: str,
        context: dict = {}
    ):
        """Process the question answering.

        Args:
            question (str): user question in string.
            context (str, optional): context retrieved externally if applicable. Defaults to "".

        Returns:
            response: truncated response.
            raw_response: original response without truncation.
        """
        # Set a seed for reproduction.
        self.set_torch_seed(self.seed)

        # Generate user prompt with question and context.
        user_prompt = self.user_prompter(question, context=context)
        # print("="*50)
        # print(f"[User Prompt] {user_prompt}")
        # print("="*50)

        # Merge user prompt to system prompt by LLM type.
        system_prompt = self.system_prompter(user_prompt, context=context)
        # print("="*50)
        # print(f"[System Prompt/Full prompt] {system_prompt}")
        # print("="*50)

        # Encode system prompt for LLM.
        system_prompt_encoded = self.tokenizer.encode(system_prompt)

        # Get response from LLM.
        response_encoded = self.llm(system_prompt_encoded)

        # Decode response since some are torch.tensor.
        raw_response = self.tokenizer.decode(response_encoded)
        # print("="*50)
        # print(f"[Raw Response] {raw_response}")
        # print("="*50)


        # Truncate response, is_truncate_response can be set externally by LLM type.
        response = self.truncate_response(raw_response)
        # print(response)

        # Return response with and without truncation.
        return response, raw_response

# =============================================================================================================

class Mistral7bv01(LLMBase):
    def __init__(
        self,
        llm_name: str="mistral-7b-v0.1",
        **kwargs
    ):
        """For Mistral 7B.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "mistral-7b-v0.1".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Login if model is not downloaded locally.
        self.login()

        # Load model to GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DICT[llm_name],
            # low_cpu_mem_usage=True
            use_cache=True,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        # Build QA pipeline.
        self.llm = lambda system_prompt: pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer.tokenizer,
            do_sample=False,
            # temperature=0.2,  # block do_sample thus remove this
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )(system_prompt)[0]["generated_text"]

    def quit(self):
        """Release model memory and instance.
        """
        self.model = self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()

    def truncate_response(
        self,
        response: str
    ):
        """Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        if self.use_example:  # with an example in the prompt, can always parse by [INST]
            response = response.split("[/INST]")[0].split("[INST]")[0]

        # response = response.replace("\n", "").strip()

        return response

    def login(self):
        """Login huggingface if no local model found.
        """
        cache_model_name = "models--" + LLM_MODEL_DICT[self.llm_name].replace("/", "--")
        cache_model_directory = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")
        cache_model_path = os.path.join(cache_model_directory, cache_model_name)

        if not os.path.exists(cache_model_path):
            # Set the token stored file.
            load_dotenv(f"{self.root}/.env")

            # Required token for huggingface login.
            if self.llm_name.find("finetune") > -1:
                login(os.getenv("hf_token_finetune"), add_to_git_credential=True)
            else:
                login(os.getenv("hf_token"), add_to_git_credential=True)

    def __call__(
        self,
        question: str,
        context: str=""
    ):
        """Process the question answering.
        Set LLM model to evaluation model, which is only applicable to local model but rather OpenAI API.

        Args:
            question (str): user question in string.
            context (str, optional): context retrieved externally if applicable. Defaults to "".

        Returns:
            response: truncated response.
            raw_response: original response without truncation.
        """
        # Set model to evaluation mode.
        self.model.eval()

        with torch.no_grad():  # without modelg gradients
            # Use the parent process interface.
            return super().__call__(question, context=context)

# =============================================================================================================

class Mistral7bInstructv01(Mistral7bv01):
    def __init__(
        self,
        llm_name="mistral-7b-instruct-v0.1",
        **kwargs
    ):
        """For Mistral 7B Instruction.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "mistral-7b-instruct-v0.1".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            max_new_tokens=1000,
            do_sample=False,
            # temperature=0.2,
            pad_token_id=self.tokenizer.tokenizer.pad_token_id
        )[0]

    def truncate_response(
        self,
        response: str
    ):
        """Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        response = response.split("[/INST]")[-1].split("</s>")[0]
        response = response.replace("\n", "").strip()

        return response

# =============================================================================================================

class Gemma7b(Mistral7bv01):
    def __init__(
        self,
        llm_name: str="gemma-7b",
        **kwargs
    ):
        """For Gemma 7B.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gemma-7b".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            max_new_tokens=500,
            do_sample=False,
            # temperature=0.2,
            # pad_token_id=tokenizer.pad_token_id
        )[0]

    def truncate_response(
        self,
        response: str
    ):
        """Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        if self.use_example:
            response = response.split("model\n")[-1].split("\n")[0]
        else:
            response = response.split("### ANSWER:\n")[-1]

        response = response.replace("\n", "").strip()

        return response

# =============================================================================================================

class Gemma7bIt(Mistral7bv01):
    def __init__(
        self,
        llm_name: str="gemma-7b-it",
        **kwargs
    ):
        """For Gemma 7B Instruction.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gemma-7b-it".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            max_new_tokens=1000,
            do_sample=False,
            # temperature=0.2,
            pad_token_id=self.tokenizer.tokenizer.pad_token_id
        )[0]

    def truncate_response(
        self,
        response: str
    ):
        """Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        if not self.is_truncate_response:
            return response

        response = response.split("model\n")[-1]
        response = response.replace("\n", "").strip()

        return response

# =============================================================================================================

class GPT35Turbo(LLMBase):
    def __init__(
        self,
        llm_name: str="gpt-3.5-turbo",
        **kwargs
    ):
        """For GPT 3.5-turbo.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gpt-3.5-turbo".
        """
        super().__init__(llm_name=llm_name, **kwargs)

        # Get API key stored in .env.
        api_key = self.get_openai_key()

        # OpenAI model entry with key.
        self.model = OpenAI(api_key=api_key)

        # OpenAI API call.
        self.llm = lambda system_prompt: \
            self.model.chat.completions.create(
                model=LLM_MODEL_DICT[llm_name],
                max_tokens=2048,
                temperature=0.0,
                messages=system_prompt
            ).choices[0].message.content

    def quit(self):
        """Close OpenAI API model entry.
        """
        self.model.close()

    def get_openai_key(self):
        """Get API key stored in .env.

        Returns:
            openai_key (str): API key.
        """
        # Set the key stored file.
        load_dotenv(f"{self.root}/.env")

        # Variable openai_key stores OpenAI key.
        openai_key = os.getenv("openai_key")

        return openai_key

# =============================================================================================================

class GPT4o(GPT35Turbo):
    def __init__(
        self,
        llm_name: str="gpt-4o",
        **kwargs
    ):
        """For GPT 4-o.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "gpt-4-o".
        """
        super().__init__(llm_name=llm_name, **kwargs)

# =============================================================================================================

class MistralFinetuned(Mistral7bv01):
    def __init__(
        self,
        llm_name: str="mistral-7b-finetuned",
        **kwargs
    ):
        """For Mistral 7B finetuned model.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "mistral-7b-finetuned".
        """
        super().__init__(llm_name=llm_name, use_example=False, **kwargs)

        # Use the base model to build model.
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
            LLM_MODEL_DICT[llm_name]
        )

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            attention_mask=torch.where(system_prompt == 2, 0, 1),
            max_new_tokens=2048,
            do_sample=False,
            # top_p=0.9,
            # temperature=0.5
        )[0]

    def truncate_response(
        self,
        response: str
    ):
        """Truncate response by specific system prompt keywords.

        Args:
            response (str): raw response from LLM.

        Returns:
            response (str): truncated response.
        """
        # Return the raw response if truncation is not required.
        if not self.is_truncate_response:
            return response

        response = response.split('###')[-1]

        if response.find("<answer>:") > -1 or response.find("<ANSWER>:") > -1:
            response = response.split("<answer>:")[-1].split("<ANSWER>:")[-1]

        response = response.replace("\n", "").strip()

        return response