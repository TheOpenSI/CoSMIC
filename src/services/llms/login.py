# -------------------------------------------------------------------------------------------------------------
# File: login.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
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

from huggingface_hub import login
from src.maps import LLM_MODEL_DICT
from dotenv import load_dotenv

# =============================================================================================================

class LLMLogin:
    def __init__(
        self,
        llm_name
    ):
        """Login LLM.

        Args:
            model (str, optional): model name. Defaults to "base".
        """
        # Set config.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.llm_name = llm_name

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