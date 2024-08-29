import subprocess
import os
import regex as re
from typing import List

from utils.log_tool import set_color

def create_py_file(file_name: str, code: str):
    """
    this will create a python file with the given code
    :param file_name:
    :param code: llm generated code
    :return: none
    """
    with open(f"{file_name}.py", "w") as file:
        file.write(code)

def create_requirements_file(file_name: str, requirements: List[str]) -> bool:
    """
    this will create a requirements file with the given requirements
    :param file_name:
    :param requirements: list of requirements
    :return: none
    """
    if len(requirements) == 1 and "none" in requirements:
        return False
    else:
        with open(f"{file_name}.txt", "w") as file:
            for requirement in requirements:
                file.write(requirement + "\n")
        return True

def clean(mount_path: str, files=None):
    """
    this will clean up the directory
    :return:
    """
    if files is None:
        files = ["main.py"]
    for file in files:
        if file in os.listdir(mount_path):
            subprocess.run(["rm", f"{mount_path}/{file}"])

    print(set_color("success", "Volume cleaned up"))

def parse_library_names(code: str) -> List[str]:
    """
    this will parse the library names from the code
    :param code: llm generated code
    :return: list of library names
    """
    return [library.split(" ")[-1].strip() for library in re.findall(r"(import\s\w+|from\s\w+)", code)]

def install_requirements():
    """
    this will install the requirements
    :return:
    """
    subprocess.run(["pip", "install", "-r", "requirements.txt"], shell=True)
    print("[INFO] Requirements installed")


def get_context(original_question:str, question_history:List[str], response_history:List[str]) -> str:
    """
    this will return the context
    :param original_question:
    :param histoty:
    :return:
    """
    context_input = f"### Context: Provided is the previous conversation history for your reference\n\n### Original Question: {original_question}\n### Past Conversation history:"
    for question, answer in zip(question_history, response_history):
        temp = answer.replace("Response -", "")
        temp = temp.replace(".\n", "")
        context_input += f"\n   Question: {question}\n\n    Your Previous Response: \n'''python\n{temp.strip()}\n'''\n\n"
    context_input += "\n\n"

    return context_input