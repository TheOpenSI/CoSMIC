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

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.services.qa import QABase

# =============================================================================================================

class CodeGenerator(QABase):
    def __init__(
        self,
        **kwargs
    ):
        """Generate Python code.
        """
        super().__init__(**kwargs)

        # Put code container here, self.container(*args, **kwargs).
        pass

    def batch_process(self, query_csv: str):
        """Process multiple queries in a .csv file.

        Args:
            query_csv (str): .csv file path.
        """
        # Put a loop after getting all questions from query_csv, and mimic modules/chess/chess_qa_quality.py
        # For each question, call the container to generate code; then, pass the code to llm by
        # super().__call__(query_per) if applicable.
        pass