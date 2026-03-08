# -------------------------------------------------------------------------------------------------------------
# File: LLMBase.py
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

import torch, os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.maps import LLM_INSTANCE_DICT, LLM_MODEL_DICT
from src.services.llms.prompts import system_prompt as system_prompt_instances
from src.services.llms.prompts import user_prompt as user_prompt_instances
from src.services.llms import tokenizer as tokenizer_instances
from src.services.base import ServiceBase
from utils.module import get_instance

class LLMBase(ServiceBase):
    def __init__(
        self,
        llm_name: str,
        user_prompt_instance_name: str="",
        system_prompt_instance_name: str="",
        use_example: bool=True,
        seed: int=0,
        is_truncate_response: bool=True,
        is_quantized: bool=False,
        device: str="cuda",
        **kwargs
    ):
        """LLM Base Class as a Service. Check the names from src/maps.py

        Args:
            llm_name (str): LLM base model name.
            user_prompt_instance_name (str, optional): user prompt instance name. Defaults to "".
            system_prompt_instance_name (str, optional): system prompt instance name. Defaults to "".
            use_example (bool, optional): use an example in system prompt. Defaults to True.
            seed (int, optional): seed for response generation. Defaults to 0.
            is_truncate_response (bool, optional): truncate the raw response. Defaults to True.
            is_quantized (bool, optional): whether use quantized model. Defaults to False.
            device (str, optional): use cuda or cpu for LLM. Defaults to "cuda".
        """
        super().__init__(**kwargs)

        # Set config.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.llm_name = llm_name
        self.use_example = use_example
        self.is_truncate_response = is_truncate_response
        self.seed = seed
        self.is_quantized = is_quantized
        self.device = device

        # Use user prompt for general questions if not specified.
        if user_prompt_instance_name == "":
            user_prompt_instance_name = "GeneralUserPrompt"

        # Build user prompter.
        self.set_user_prompter_by_instance_name(user_prompt_instance_name)

        # Get LLM instance name.
        if llm_name in LLM_INSTANCE_DICT.keys():
            llm_instance_name = LLM_INSTANCE_DICT[llm_name]
        elif llm_name.find("ollama") > -1:
            llm_instance_name = "Ollama"
        else:
            llm_instance_name = "GPT"

        # Use system prompt by LLM type.
        if system_prompt_instance_name == "":
            system_prompt_instance_name = llm_instance_name

        # Build system prompter.
        self.set_system_prompter_by_instance_name(
            system_prompt_instance_name,
            use_example=use_example
        )

        # Build tokenizer by LLM type.
        self.tokenizer = get_instance(
            tokenizer_instances,
            llm_instance_name
        )(llm_name=llm_name, device=self.device)

        # CPU model cannot support quantization.
        if self.device.find("cpu") > -1:
            is_quantized = False

        # Set quantization configs.
        if is_quantized:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            self.quantization_config = None

        # Set attention_mask.
        self.attention_mask = lambda system_prompt: \
            torch.any(torch.stack([system_prompt==v for v in [0,1,2]], dim=-1), dim=-1).logical_not()

        # Model and LLM are set from the children class by LLM type.
        self.model = None
        self.llm = None

    def set_user_prompter_by_instance_name(
        self,
        user_prompt_instance_name: str,
        **kwargs
    ):
        """Change user prompter by instance name externally.

        Args:
            user_prompt_instance_name (str): set an user prompter instance name.
        """
        self.user_prompter = get_instance(
            user_prompt_instances,
            user_prompt_instance_name
        )(**kwargs)

    def set_user_prompter(
        self,
        user_prompt_instance: user_prompt_instances.UserPromptBase,
    ):
        """Change user prompter externally.

        Args:
            user_prompt_instance (UserPromptBase): set an user prompter instance.
        """
        self.user_prompter = user_prompt_instance

    def set_system_prompter_by_instance_name(
        self,
        system_prompt_instance_name: str,
        **kwargs
    ):
        """Change system prompter by instance name externally.

        Args:
            system_prompt_instance_name (str): set a system prompter instance name.
        """
        self.system_prompter = get_instance(
            system_prompt_instances,
            system_prompt_instance_name
        )(**kwargs)

    def set_system_prompter(
        self,
        system_prompt_instance: system_prompt_instances.SystemPromptBase,
    ):
        """Change system prompter externally.

        Args:
            system_prompt_instance (SystemPromptBase): set a system prompter instance.
        """
        self.system_prompter = system_prompt_instance

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
        # Remove DeepSeek model reasoning part in the response.
        if self.llm_name.find("deepseek") > -1:
            response = response.split("</think>")[-1]

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

        # Merge user prompt to system prompt by LLM type.
        system_prompt = self.system_prompter(user_prompt, context=context)

        # Encode system prompt for LLM.
        system_prompt_encoded = self.tokenizer.encode(system_prompt)

        # Get response from LLM.
        response_encoded = self.llm(system_prompt_encoded)

        # Decode response since some are torch.tensor.
        raw_response = self.tokenizer.decode(response_encoded)

        # Truncate response, is_truncate_response can be set externally by LLM type.
        response = self.truncate_response(raw_response)

        # Return response with and without truncation.
        return response, raw_response