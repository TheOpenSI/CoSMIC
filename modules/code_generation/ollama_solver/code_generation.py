# -------------------------------------------------------------------------------------------------------------
# File: code_generation.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
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

import os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

from src.services.qa import QABase
from src.services.code_generation.ollama import Ollama
from utils.log_tool import set_color

# =============================================================================================================

class CodeGenerator(QABase):
    def __init__(
        self,
        model="mistral",
        enable_chat_history=False,
        **kwargs
    ):
        """Construct code generation instance.

        Args:
            model (str, optional): model name. Defaults to "mistral".
            enable_chat_history (bool, optional): keep chat history. Defaults to False.
        """
        super().__init__(**kwargs)
        self.ollama = Ollama(model=model, enable_chat_history=enable_chat_history)

    def __call__(
        self,
        user_prompt: str=None
    ):
        """Generate code.

        Args:
            user_prompt (str, optional): question from entry. Defaults to None.
        """
        if user_prompt is not None:
            # Answer question from entry inputs.
            response = self.ollama(user_prompt)

            print(set_color("info", f"Question: {user_prompt}\nResponse: {response}"))
        else:
            # Answer question from user inputs.
            while True:
                user_prompt = input("[INPUT] Enter a question: ")
                if user_prompt == "exit": break
                response = self.ollama(user_prompt)
                print(set_color("info", f"Question: {user_prompt}\nResponse: {response}"))

        # Note: set warning to resource_tracker for leaked semaphore objects.
        self.ollama.cleanup()

# =============================================================================================================

if __name__ == "__main__":
    code_generator = CodeGenerator(enable_chat_history=True)
    code_generator()