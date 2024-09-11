# -------------------------------------------------------------------------------------------------------------
# File: chat_history.py
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

from collections import deque
from utils.log_tool import set_color

# =============================================================================================================

class ChatHistory:
    def __init__(
        self,
        original_question: str,
        max_history: int=3
    ):
        """Chat history for Ollama.

        Args:
            original_question (str): question.
            max_history (int, optional): maximum number of cached history chat. Defaults to 3.
        """
        if not isinstance(original_question, str) or original_question.strip() == "":
            print(set_color("error", "Question must be non-empty."))
            exit

        if not isinstance(max_history, int) or max_history < 1:
            print(set_color("error", "The number of chat history must be positive."))
            exit

        self.original_question = original_question
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)

    def get_original_question(self):
        """Get original question.

        Returns:
            original_question (str): original question.
        """
        if self.original_question is None or self.original_question.strip() == "":
            print(set_color("warning", "Question is required."))

        return self.original_question

    def get_conversation_history(self):
        """Get conversation history.

        Returns:
            conversation_history (list): conversation history.
        """
        return list(self.conversation_history)

    def get_history_length(self):
        """Get the number of cached conversation.

        Returns:
            num_chats (int): the number of cached conversation.
        """
        num_chats = len(self.conversation_history)

        return num_chats

    def get_last_question(self):
        """Get the last question.

        Returns:
            last_question (str): the last question.
        """
        if self.conversation_history:
            last_question = self.conversation_history[-1][0]
        else:
            last_question = self.original_question 

        return last_question

    def add_interaction(
        self, question: str,
        answer: str
    ):
        """Add a chat.

        Args:
            question (str): question.
            answer (str): answer to the question.
        """
        if not isinstance(question, str) or question.strip() == "":
            print(set_color("error", "Question must be non-empty."))
            exit

        if not isinstance(answer, str) or answer.strip() == "":
            print(set_color("error", "Answer must be non-empty."))
            exit

        self.conversation_history.append((question, answer))

    def clear_history(self):
        """Remove all cached chats.
        """
        self.original_question = None
        self.conversation_history = deque(maxlen=self.max_history)

    def get_history_chat_dict(self):
        """Get question and cached chats in dictionary.

        Returns:
            history_dict (dict): a dictionary for question and cached chats.
        """
        history_dict = {
            "original_question": self.original_question,
            "conversation_history": list(self.conversation_history)
        }

        return history_dict

    def get_history_chat_tuple(self):
        """Get cached chats in tuple.

        Returns:
            history_tuple (tuple): chat history.
        """

        history_tuple = (
            f"ChatHistory(original_question='{self.original_question}', "
            f"history_length={len(self.conversation_history)})"
        )

        return history_tuple

    def get_history_chat_string(self):
        """Get cached chats in string.

        Returns:
            history_string: cached chats in string.
        """
        history_string = (
            f"ChatHistory("
            f"original_question={repr(self.original_question)} "
            f"(type={type(self.original_question).__name__}), "
            f"max_history={self.max_history} (type={type(self.max_history).__name__}), "
            f"conversation_history={list(self.conversation_history)} "
            f"(type={type(self.conversation_history).__name__}), "
            f"history_length={len(self.conversation_history)}"
            f")"
        )

        return history_string