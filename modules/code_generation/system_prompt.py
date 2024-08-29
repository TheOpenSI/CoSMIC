import os, sys
from transformers import AutoTokenizer

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.llms.prompts.system_prompt import SystemPromptBase


class CodeGeneratorSystemPrompt(SystemPromptBase):
    def __init__(self, tokenizer: AutoTokenizer, use_example: bool = False):
        super().__init__()
        self.system_prompt_prefix = tokenizer.bos_token

    def __call__(
        self,
        user_prompt: str,
        context: dict
    ):
        system_prompt = f'''
{self.system_prompt_prefix} You are a python code genration assistant.
In your response do not add any text that will be unfamiliar to a python compiler.


In ### Code section, response with only the necessary code to fulfill the Question including any imports required.
In ### Requirements, list all the libraries required to run the code, add 'none' if no libraries are required. 
In ### Example, always provide an example to run the code.

Format your code like the following - 

### Requiements
$libraries

### Code
$python_code

### Example
$example_to_run_code

A sample response - 
### Question - Write a python function to load a csv file.

Response -
### Requirements
pandas

### Code
import pandas as pd
def load_csv(file_path):
    return pd.read_csv(file_path)

### Example
df = load_csv("data.csv")
print(df.head())


Now answer the following question -  
### Question - {user_prompt}'''

        return system_prompt if not context["fix_mode"] else self.system_prompt_prefix + user_prompt