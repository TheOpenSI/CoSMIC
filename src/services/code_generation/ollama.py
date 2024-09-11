# -------------------------------------------------------------------------------------------------------------
# File: ollama.py
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

import os, sys, ollama

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

from typing import Dict, List
from jinja2 import Template
from src.services.base import ServiceBase
from utils.log_tool import set_color
from modules.code_generation.ollama_solver.chat_history import ChatHistory

# =============================================================================================================

class Ollama(ServiceBase):
    def __init__(
        self,
        model: str="mistral",
        enable_chat_history: bool=False
    ):
        """Ollama for code generation service.

        Args:
            model (str, optional): model name. Defaults to "mistral".
            enable_chat_history (bool, optional): keep chat history. Defaults to False.
        """
        super().__init__()
        self.root = f"{os.path.dirname(os.path.abspath(__file__))}/../../.."
        self.model = model
        self.enable_chat_history = enable_chat_history
        self.chat_history = None

        self.prompt_template_path = os.path.join(
            self.root,
            "scripts/code_generation/default_chat_template.jinja"
        )

        self.system_prompt = f"Always answer the question to the best of your ability " \
            f"even if the context is not useful."

    def pull_model(self):
        """ Pull the model from the server.
        """
        ollama.pull_model(self.model)

    def init_chat_history(
        self,
        original_question: str,
        max_history: int=3
    ):
        """ Initialize the chat history with the original question.

        Args:
            original_question (str): The initial question to start the chat.
            max_history (int): Maximum number of interactions to store in history.
                Default to 3.
        """
        self.chat_history = ChatHistory(original_question, max_history)

    def cleanup(self):
        """ Nothing to cleanup for Ollama.
        """
        pass

    def set_system_prompt(
        self,
        prompt: str
    ):
        """ Set system prompt and reset chat history.

        Args:
            prompt (str): System prompt.
        """
        self.system_prompt = prompt

        # Reset chat history when system prompt changes
        if self.enable_chat_history:
            self.chat_history = None

    def generate_prompt(
        self,
        messages: List[Dict],
        bos_token: str=""
    ):
        """Generate a prompt from the provided messages using the template.

        Args:
            messages (List[str]): List of messages to include in the prompt.
            bos_token (str): The BOS token to use in the prompt.
        """
        filtered_messages = [msg for msg in messages if msg.get("content").strip()]

        with open(self.prompt_template_path, "r") as jinja_file:
            template_str = jinja_file.read()

        # Create a Jinja template object.
        template = Template(template_str)

        # Render the template with the provided messages and bos_token.
        rendered_prompt = template.render(
            messages=filtered_messages,
            bos_token = bos_token
        )

        return rendered_prompt

    def __call__(
        self,
        user_query: str,
        context: str=""
    ):
        """ Generate a response from the user query using the model, including chat history.

        Args:
            user_query (str): user query.
            context (str): context.
        """
        try:
            conversation_history = ""  # no conversation history by default.

            if self.enable_chat_history:
                if self.chat_history is None:
                    # User_query is the original question and max_history is 3 by default.
                    self.init_chat_history(user_query) 

                # Prepare the context from chat history.
                conversation_history = "\n" + "\n".join([(
                    f"\tPrevious Question {index + 1}: {q}\n"
                    f"\tPrevious Answer {index + 1}: {a}\n") 
                    for index, (q, a) in enumerate(self.chat_history.conversation_history
                )])

            # Prepare the messages to generate the prompt.
            # Keep the sequece of messages as follows: System, Context, Conversation, User.
            messages = [
                {"role": "System", "content": self.system_prompt},
                {"role": "Context", "content": context},
                {"role": "Conversation", "content": conversation_history},
                {"role": "User", "content": user_query}
            ]

            # System prompt.
            full_query = self.generate_prompt(messages)  # bos_token is empty by default.

            # Generate response from the model.
            response = ollama.generate(model=self.model, prompt=full_query)

            # Add the interaction to chat history.
            if self.enable_chat_history and self.chat_history and response:
                # For chat it's response["message"]["content"].
                self.chat_history.add_interaction(user_query, response["response"])

            return response["response"]
        except ollama.ResponseError as e:
            print(set_color("error", e.error))

            if e.status_code == 404:
                print(set_color("info", "Attempting to pull the model."))
                self.pull_model()