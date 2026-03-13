# -------------------------------------------------------------------------------------------------------------
# File: llm.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
#     Bing Tran <binhsan1307@gmail.com> (2026)
#
# Copyright (c) 2024 - 2026 Open Source Institute
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
import torch, os, sys, ollama, dotenv

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI
from transformers import pipeline
from ...maps import LLM_INSTANCE_DICT, LLM_MODEL_DICT
from .LLMBase import LLMBase
from .Ollama import Ollama
from .prompts import system_prompt as system_prompt_instances
from .prompts import user_prompt as user_prompt_instances
from . import tokenizer as tokenizer_instances
from ..base import ServiceBase
from .login import LLMLogin
from utils.module import get_instance
from utils.log_tool import set_color


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

        # Login if model has been downloaded locally.
        LLMLogin(llm_name).login()

        # Load model to GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DICT[llm_name],
            use_cache=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config,
        )  # low_cpu_mem_usage=True

        # Build QA pipeline.
        self.llm = lambda system_prompt: pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer.tokenizer,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )(system_prompt)[0]["generated_text"]

    def quit(self):
        """Release model memory and instance.
        """
        if not self.is_quantized:
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

        return response

    def __call__(
        self,
        question: str,
        context: str=""
    ):
        """Process the question answering.
        Set LLM model to evaluation model, which is only applicable to local model but rather OpenAI API.

        Args:
            question (str): user question in string.
            context (str|dict, optional): context retrieved externally if applicable. Defaults to "".

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
            attention_mask=self.attention_mask(system_prompt),
            pad_token_id=self.tokenizer.tokenizer.pad_token_id,
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

        if self.use_example:
            response = response.split("model\n")[2].split("\n")[0]
        else:
            response = response.split("### ANSWER:\n")[-1]

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

        return response

# =============================================================================================================

class GPT(LLMBase):
    def __init__(
        self,
        llm_name: str="gpt-3.5-turbo",
        **kwargs
    ):
        """For OpenAI API.

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
                model=llm_name,
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
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if openai_key == "":
            envs = dotenv.dotenv_values(f"{self.root}/.env")

            if "OPENAI_API_KEY" in envs.keys():
                openai_key = envs["OPENAI_API_KEY"]
            else:
                print(set_color("warning", "OPENAI_API_KEY is required in .env."))
                openai_key = ""

        # Get warning for invalid API key.
        if openai_key == "":
            print(set_color("warning", "The OPENAI_API_KEY in .env is invalid."))

        return openai_key

# =============================================================================================================

# class Ollama(LLMBase):
#     def __init__(
#         self,
#         llm_name: str="mistral",
#         **kwargs
#     ):
#         """For Ollama supported LLMs.

#         Args:
#             llm_name (str, optional): Ollama supported LLMs available at https://ollama.com/library.
#         """
#         super().__init__(llm_name=llm_name, **kwargs)

#         # Ollama model name will be in "ollama:[llm_name]", so truncate it to get the exact one.
#         llm_name = llm_name.replace("ollama:", "")

#         # Pull model.
#         ollama.pull(llm_name)

#         # Ollama API call.
#         self.llm = lambda system_prompt: \
#             ollama.chat(
#                 model=llm_name,
#                 messages=system_prompt
#             )['message']['content']

#     def quit(self):
#         """Close OpenAI API model entry.
#         """
#         if self.model: self.model.close()

# =============================================================================================================

class MistralFinetuned(Mistral7bv01):
    def __init__(
        self,
        llm_name: str="mistral-7b-finetuned",
        use_example=False,
        is_quantized=True,
        **kwargs
    ):
        """For Mistral 7B finetuned model.

        Args:
            llm_name (str, optional): LLM name in src/maps.py. Defaults to "mistral-7b-finetuned".
            use_example (bool, optional): use example instance. Defaults to False.
            is_quantized (bool, optional): use quantized model, always true. Default to True.
        """
        super().__init__(llm_name=llm_name, use_example=use_example, is_quantized=True, **kwargs)

        # Use the base model to build model.
        base_llm_model = "mistral-7b-v0.1"

        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DICT[base_llm_model],
            quantization_config=self.quantization_config,
            use_cache=True,
            device_map="auto"
        )  # low_cpu_mem_usage=True

        self.model = PeftModel.from_pretrained(
            base_model,
            LLM_MODEL_DICT[llm_name]
        )

        # Set up LLM.
        self.llm = lambda system_prompt: self.model.generate(
            system_prompt,
            attention_mask=self.attention_mask(system_prompt),
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=self.tokenizer.tokenizer.eos_token_id
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

        return response
