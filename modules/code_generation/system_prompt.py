# -------------------------------------------------------------------------------------------------------------
# File: system_prompt.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
# 
# Copyright (c) 2024 Open Source Institute
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including but not
# limited to the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
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

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from transformers import AutoTokenizer
from src.llms.prompts.system_prompt import SystemPromptBase

# =============================================================================================================

class CodeGeneratorSystemPrompt(SystemPromptBase):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        use_example: bool = False,
        **kwargs
    ):
        """System prompter for code generation.

        Args:
            tokenizer (AutoTokenizer): tokenizer instance set externally.
            use_example (bool, optional): use example in system prompt for key words.
                Defaults to False.
        """
        super().__init__(use_example=use_example, **kwargs)
        self.prefix = tokenizer.bos_token

    def __call__(
        self,
        user_prompt: str,
        context: dict
    ):
        """Generate system prompt.

        Args:
            user_prompt (str): user prompt.
            context (dict|str): context with information indicating code error.

        Returns:
            system_prompt: system prompt for code generation or correction.
        """
        system_prompt = (
            f"{self.prefix} You are a python code generation assistant.\n"
            "In your response, do not add any text that will be unfamiliar to a python compiler.\n\n"
            "- In ### Code section, respond with only the necessary code to fulfill the question, "
            "including any imports required.\n- In ### Requirements, list all the libraries required to "
            "run the code. Add 'none' if no libraries are required.\n"
            "- In ### Example, always provide an example to run the code.\n\n"
            # "Format your code like the following - \n\n"
            # "### Requirements\n"
            # "$libraries\n\n"
            # "### Code\n"
            # "$python_code\n\n"
            # "### Example\n"
            # "$example_to_run_code\n\n"
            "A sample response - \n"
            "### Question - Write a python function to load a csv file.\n\n"
            "Response -\n"
            "### Requirements\n"
            "pandas\n\n"
            "### Code\n"
            "import pandas as pd\n"
            "def load_csv(file_path):\n"
            "    return pd.read_csv(file_path)\n\n"
            "### Example\n"
            'df = load_csv("data.csv")\n'
            "print(df.head())\n\n"
            "Now answer the following question -  \n"
            f"### Question - {user_prompt}.\n"
            "Response -\n"
        )

        # "fix_mode"=True indicates code error, i.e., exit code!=0.,
        # use the new system prompt to enquire the code correctness.
        if context["fix_mode"]:
            system_prompt = self.prefix + user_prompt
        
        return system_prompt