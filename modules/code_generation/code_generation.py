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

from typing import List
from src.services.pycapsule import PyCapsule
from src.services.qa import QABase
from modules.code_generation.system_prompt import CodeGeneratorSystemPrompt
from modules.code_generation.utils.code_parser import parse_input
from modules.code_generation.utils.pycapsule_util import create_requirements_file, create_py_file, get_context, clean
from utils.log_tool import set_color

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
        self.KEEP_HISTORY = 1 # conversation pair.
        self.llm.system_prompter.set_use_example(False)
        # Tokenizer instance to dynamically collect the 'bos_token'.
        self.tokenizer = self.llm.tokenizer.tokenizer

        # Put code container here, self.container(*args,option **kwargs).
        self.llm.set_system_prompter(CodeGeneratorSystemPrompt(self.tokenizer)) # access the bos token here.

    def batch_process(
        self,
        query_csv: str
    ):
        """Process multiple queries in a .csv file.

        Args:
            query_csv (str): .csv file path.
        """
        # Extracting questions from query_csv.
        df = pd.read_csv(query_csv)
        queries = df["Question"]

        # Save to log file.
        if self.log_file is not None:
            self.log_file.writerow([
                "Question", 
                "Answer"
            ])

        for query in queries:
            # This is to change the fix mode, it activates when the code has an error.
            context = {"fix_mode": False}

            response = super().__call__(query=query, context= context)
            # Using parser utility.
            # From return, response[1] is the code section with raw response.
            requirements, code, example = parse_input(response[1])
            
            # Create requirements file.
            # This creates requirements.txt file in the container.
            create_requirements_file(self.container.container_mount_path + "/requirements", requirements)

            # This creates main.py file in the container and gets executed at entrypoint.
            create_py_file(self.container.container_mount_path + "/main", code + "\n" + example)
            
            pycapsule_return_code = -1
            pycapsule_response = ""
            pycapsule_error = ""

            # First run, changes exit code to 1 if code has an error.
            if not self.container.check_if_container_exists():
                pycapsule_return_code, pycapsule_response, pycapsule_error = self.container.create_container()
            else:
                pycapsule_return_code, pycapsule_response, pycapsule_error = self.container.start_container()

            # The following is only relevant if retrun code is not 0.
            # Conversation history
            question_history:List[str] = []
            response_history:List[str] = []

            # Atempt count, breaks after 2.
            attempt_count = 0
            
            while pycapsule_return_code != 0 and attempt_count < 2: # set attempt count here.
                print(set_color("error", "Generated code had an error, starting PYCAPSULE service"))

                response_history.append(response[1])
                question_history.append(query)

                # Genertating contetxt using conversation history.
                conv_history = get_context(query, question_history, response_history)
                # New question in fix mode.
                new_question = (
                    f"### Question - Your code had the following error: {pycapsule_error}.\n\n"
                    "Please correct your code and respond with:\n"
                    "- the corrected code in the same format, and \n"
                    "- an example."
                )
                fix_context = {"fix_mode": True} # gets activated when the code has an error.
                
                # Fix mode attempt.
                mistral_response_update = super().__call__(query=new_question + 
                                                           "\n\n" + 
                                                           conv_history, 
                                                           context=fix_context)

                # Appeding conversation history.
                response_history.append(mistral_response_update)
                question_history.append(new_question)

                # Parsing response.
                requirements, code, example = parse_input(mistral_response_update[1])

                if len(response_history) > self.KEEP_HISTORY:
                    # Keeping a limited number of conversation history.
                    response_history.pop(0)
                    question_history.pop(0)

                # Running in fix mode.
                # This updates the main.py and requirements.txt file with the new code.
                create_py_file(self.container.container_mount_path + "/main", code + "\n" + example)
                create_requirements_file(self.container.container_mount_path + "/requirements", requirements)
                # Starting the container to run the new code.
                pycapsule_return_code, pycapsule_error, pycapsule_error = self.container.start_container()

                attempt_count += 1
            
            # Cleaning up the volume.
            # This removes the main.py and requirements.txt file from the container.
            # Ouput gets saved in results.
            clean(self.container.container_mount_path, ["main.py", "requirements.txt"])
            
            # Saving to csv file.
            if self.log_file is not None:
                self.log_file.writerow([
                    query,
                    code
                ])
