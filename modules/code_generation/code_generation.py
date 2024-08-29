# -------------------------------------------------------------------------------------------------------------
# File: code_generation.py
# Project: OpenSI AI System
# Contributors:
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
import pandas as pd

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.services.pycapsule import PyCapsule
from modules.code_generation.system_prompt import CodeGeneratorSystemPrompt
from src.services.qa import QABase
from utils.code_parser import parse_input
from utils.pycapsule_util import create_requirements_file, create_py_file, get_context, clean
from typing import List

# =============================================================================================================
class CodeGenerator(QABase):
    def __init__(
        self,
        **kwargs
    ):
        """Generate Python code.
        """
        super().__init__(**kwargs)
        self.container = PyCapsule()
        self.KEEP_HISTORY = 1
        self.llm.system_prompter.set_use_example(False)
        self.tokenizer = self.llm.tokenizer.tokenizer

        # Put code container here, self.container(*args,option **kwargs).
        self.llm.set_system_prompter(CodeGeneratorSystemPrompt(self.tokenizer)) # access the bos token here

    def batch_process(
        self,
        query_csv: str
    ):
        """Process multiple queries in a .csv file.

        Args:
            query_csv (str): .csv file path.
        """
        # Put a loop after getting all questions from query_csv, and mimic modules/chess/chess_qa_quality.py
        # For each question, call the container to generate code; then, pass the code to llm by
        # super().__call__(query_per) if applicable.

        # extracting questions from query_csv
        df = pd.read_csv(query_csv)
        queries = df["Question"]

        # Save to log file.
        if self.log_file is not None:
            self.log_file.writerow([
                "Question", 
                "Answer"
            ])

        for query in queries:
            context = {"fix_mode": False}

            response = super().__call__(query=query, context= context)
            requirements, code, example = parse_input(response[1]) # response[1] is the code section with raw response
            
            # create requirements file
            create_requirements_file(self.container.container_mount_path + "/requirements", requirements)
            # create main.py file
            create_py_file(self.container.container_mount_path + "/main", code + "\n" + example)
            
            # pycapsule
            pycapsule_return_code = -1
            pycapsule_response = ""
            pycapsule_error = ""

            # first run
            if not self.container.check_if_container_exists():
                pycapsule_return_code, pycapsule_response, pycapsule_error = self.container.create_container()
            else:
                pycapsule_return_code, pycapsule_response, pycapsule_error = self.container.start_container()

            # the following is only relevant if retrun code is not 0
            # history
            question_history:List[str] = []
            response_history:List[str] = []

            # attempt count, will break after 5
            attempt_count = 0
            
            while pycapsule_return_code != 0 and attempt_count < 2:
                print("[OPENSI PYCAPSULE] Generated code had an error, starting PYCAPSULE service")

                response_history.append(response[1])
                question_history.append(query)

                conv_history = get_context(query, question_history, response_history)
                new_question = f"### Question - Your code had the following error: {pycapsule_error}.\n\nPlease correct your code and response with:\n- the corrected code in the same format, and \n- an example."
                fix_context = {"fix_mode": True} # gets activated when the code has an error
                mistral_response_update = super().__call__(query=new_question + "\n\n" + conv_history, context=fix_context)
                line_gap = "\n\n"
                print(f"[SENDING QUESTION] {new_question + line_gap + conv_history}")
                print(f"[MISTRAL-RESPONSE]\n{mistral_response_update}")

                response_history.append(mistral_response_update)
                question_history.append(new_question)

                requirements, code, example = parse_input(mistral_response_update[1])
                print(f"[MISTRAL-RESPONSE] Updated code -{code, example}")

                if len(response_history) > self.KEEP_HISTORY:
                    # keeping a limited number of conversation history
                    response_history.pop(0)
                    question_history.pop(0)

                create_py_file(self.container.container_mount_path + "/main", code + "\n" + example)
                create_requirements_file(self.container.container_mount_path + "/requirements", requirements)
                pycapsule_return_code, pycapsule_error, pycapsule_error = self.container.start_container()

                attempt_count += 1
            
            # clean(self.container.container_mount_path, ["main.py", "requirements.txt"])
            
            # saving to csv file
            if self.log_file is not None:
                self.log_file.writerow([
                    query,
                    code
                ])