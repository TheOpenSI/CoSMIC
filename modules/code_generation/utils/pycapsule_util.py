# -------------------------------------------------------------------------------------------------------------
# File: pycapsule_util.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
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

import subprocess, os
import regex as re

from typing import List
from utils.log_tool import set_color

# =============================================================================================================

def create_py_file(file_name: str, code: str):
    """
    Create a Python file with the given code.

    :param file_name: The name of the Python file to be created (without extension).
    :param code: The LLM-generated code to write into the file.
    :return: None
    """
    with open(f"{file_name}.py", "w") as file:
        file.write(code)

# =============================================================================================================

def create_requirements_file(file_name: str, requirements: List[str]) -> bool:
    """
    Create a requirements file with the given list of requirements.

    :param file_name: The name of the requirements file to be created (without extension).
    :param requirements: A list of requirements to be written into the file.
    :return: True if the file was created, False otherwise.
    """
    if len(requirements) == 1 and "none" in requirements:
        return False
    else:
        with open(f"{file_name}.txt", "w") as file:
            for requirement in requirements:
                file.write(requirement + "\n")
        return True

# =============================================================================================================

def clean(mount_path: str, files=None):
    """
    Clean up the specified directory by removing specified files.

    :param mount_path: The path of the directory to clean.
    :param files: A list of files to remove from the directory. Defaults to ["main.py"].
    :return: None
    """
    if files is None:
        files = ["main.py"]

    for file in files:
        if file in os.listdir(mount_path):
            subprocess.run(["rm", f"{mount_path}/{file}"])

    print(set_color("success", "Volume cleaned up"))

# =============================================================================================================

def parse_library_names(code: str) -> List[str]:
    """
    Parse library names from the given code.

    :param code: The LLM-generated code to parse for library names.
    :return: A list of library names found in the code.
    """
    return [library.split(" ")[-1].strip()
            for library in re.findall(r"(import\s\w+|from\s\w+)", code)]

# =============================================================================================================

def install_requirements():
    """
    Install the requirements listed in the requirements.txt file.

    :return: None
    """
    subprocess.run(["pip", "install", "-r", "requirements.txt"], shell=True)
    print("[INFO] Requirements installed")

# =============================================================================================================

def get_context(original_question: str, 
                question_history: List[str], 
                response_history: List[str]) -> str:
    """
    Return the context for a conversation.

    :param original_question: The original question that initiated the conversation.
    :param question_history: A list of previous questions asked in the conversation.
    :param response_history: A list of responses corresponding to the question_history.
    :return: A formatted string representing the context.
    """
    context_input = (
        "### Context: Provided is the previous conversation history "
        "for your reference\n\n"
        f"### Original Question: {original_question}\n"
        "### Past Conversation history:"
    )
    
    for question, answer in zip(question_history, response_history):
        temp = answer.replace("Response -", "").replace(".\n", "")
        context_input += (
            f"\n   Question: {question}\n\n"
            f"    Your Previous Response: \n'''python\n"
            f"{temp.strip()}\n'''\n\n"
        )

    context_input += "\n\n"

    return context_input