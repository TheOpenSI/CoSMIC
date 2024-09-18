# -------------------------------------------------------------------------------------------------------------
# File: tokenizer.py
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

import os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

from transformers import AutoTokenizer
from src.maps import LLM_MODEL_DICT

# =============================================================================================================

class TokenizerBase:
    def __init__(
        self,
        llm_name: str=""
    ):
        """Base class for tokenizer.

        Args:
            llm_name (str, optional): LLM name, see src/maps.py, adapting tokenizer to different models.
                Defaults to "".
        """
        if (llm_name == "") or (llm_name not in LLM_MODEL_DICT.keys()):
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_DICT[llm_name],
                add_eos_token=False,
            )

            # Set tokenizer pad_token.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        system_prompt: str,
        **kwargs
    ):
        """Encode prompt for LLM.

        Args:
            system_prompt (str): system prompt containing user prompt and context.

        Returns:
            system_prompt (str): encoded system prompt, can be string or torch.tensor.
        """
        return system_prompt

    def decode(
        self,
        response: str,
        **kwargs
    ):
        """Decode response according to the encoder.

        Args:
            response (str): raw response, string or torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return response

# =============================================================================================================

class Mistral7bv01(TokenizerBase):
    def __init__(
        self,
        llm_name: str="mistral-7b-v0.1"
    ):
        """For Mistral 7B.

        Args:
            llm_name (str, optional): LLM name. Defaults to "mistral-7b-v0.1".
        """
        super().__init__(llm_name)

# =============================================================================================================

class Mistral7bInstructv01(TokenizerBase):
    def __init__(
        self,
        llm_name: str="mistral-7b-instruct-v0.1"
    ):
        """For Mistral 7B Instruction.

        Args:
            llm_name (str, optional): LLM name. Defaults to "mistral-7b-instruct-v0.1".
        """
        super().__init__(llm_name)

    def encode(
        self,
        system_prompt: str,
        **kwargs
    ):
        """Encode prompt for LLM.

        Args:
            system_prompt (str): system prompt containing user prompt and context.

        Returns:
            system_prompt (str): encoded system prompt, which is torch.tensor.
        """
        return self.tokenizer.apply_chat_template(
            system_prompt,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).to("cuda")

    def decode(
        self,
        response: str,
        **kwargs
    ):
        """Decode response according to the encoder.

        Args:
            response (str): raw response, torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return self.tokenizer.decode(response, **kwargs)

# =============================================================================================================

class Gemma7b(TokenizerBase):
    def __init__(
        self,
        llm_name: str="gemma-7b"
    ):
        """For Gemma 7B.

        Args:
            llm_name (str, optional): LLM name. Defaults to "mistral-gemma-7b".
        """
        super().__init__(llm_name)

    def encode(
        self,
        system_prompt: str,
        **kwargs
    ):
        """Encode prompt for LLM.

        Args:
            system_prompt (str): system prompt containing user prompt and context.

        Returns:
            system_prompt (str): encoded system prompt, which is torch.tensor.
        """
        return self.tokenizer(
            system_prompt,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).input_ids.to("cuda")

    def decode(
        self,
        response: str,
        **kwargs
    ):
        """Decode response according to the encoder.

        Args:
            response (str): raw response, torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return self.tokenizer.decode(
            response,
            skip_special_tokens=True,
            **kwargs
        )

# =============================================================================================================

class Gemma7bIt(TokenizerBase):
    def __init__(
        self,
        llm_name: str="gemma-7b-it"
    ):
        """For Gemma 7B Instruction.

        Args:
            llm_name (str, optional): LLM name. Defaults to "gemma-7b-instruct".
        """
        super().__init__(llm_name)

    def encode(
        self,
        system_prompt: str,
        **kwargs
    ):
        """Encode prompt for LLM.

        Args:
            system_prompt (str): system prompt containing user prompt and context.

        Returns:
            system_prompt (str): encoded system prompt, which is torch.tensor.
        """
        return self.tokenizer.apply_chat_template(
            system_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).to("cuda")

    def decode(
        self,
        response: str,
        **kwargs
    ):
        """Decode response according to the encoder.

        Args:
            response (str): raw response, torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return self.tokenizer.decode(
            response,
            skip_special_tokens=True,
            **kwargs
        )

# =============================================================================================================

class GPT35Turbo(TokenizerBase):
    def __init__(
        self,
        llm_name: str=""
    ):
        """For GPT 3.5-turbo.
        GPT does not require tokenizer, just keep the interface.

        Args:
            llm_name (str, optional): LLM name. Defaults to "".
        """
        super().__init__(llm_name="")

# =============================================================================================================

class GPT4o(TokenizerBase):
    def __init__(
        self,
        llm_name: str=""
    ):
        """For GPT 4-o.
        GPT does not require tokenizer, just keep the interface.

        Args:
            llm_name (str, optional): LLM name. Defaults to "".
        """
        super().__init__(llm_name="")

# =============================================================================================================

class MistralFinetuned(TokenizerBase):
    def __init__(
        self,
        llm_name: str=""
    ):
        """For Mistral 7B finetuned.
        Since the tokenizer depends on base model, not finetuned model, remaining the definition internally.

        Args:
            llm_name (str, optional): LLM name. Defaults to "".
        """
        super().__init__(llm_name="")
        base_llm_name = "mistral-7b-v0.1"

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_DICT[base_llm_name],
            add_bos_token=True
        )

    def encode(
        self,
        system_prompt: str,
        **kwargs
    ):
        """Encode prompt for LLM.

        Args:
            system_prompt (str): system prompt containing user prompt and context.

        Returns:
            system_prompt (str): encoded system prompt, which is torch.tensor.
        """
        return self.tokenizer(
            system_prompt,
            return_tensors="pt",
            **kwargs
        ).input_ids.to("cuda")

    def decode(
        self,
        response: str,
        **kwargs
    ):
        """Decode response according to the encoder.

        Args:
            response (str): raw response, torch.tensor, from LLM.

        Returns:
            response: decoded response.
        """
        return self.tokenizer.decode(
            response,
            skip_special_tokens=True,
            **kwargs
        )