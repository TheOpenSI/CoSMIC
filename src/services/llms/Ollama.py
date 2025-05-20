# -------------------------------------------------------------------------------------------------------------
# File: Ollama.py
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
import os, sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

import os
import sys
import requests
import subprocess
import ollama

from typing import List, Dict, Any, Union
from ollama import Client

from src.services.llms.llm import LLMBase

class Ollama(LLMBase):
    def __init__(self,
                 llm_name: str = "llama3.2",
                 container_name: str = "ollama",
                 local_port: int = 11434,
                 **kwargs) -> None:
        """
        LLMs using the official Ollama container.

        Args:
            llm_name (str, optional): Ollama supported LLMs available at https://ollama.com/library.
            container_name (str, optional): Name of the Ollama container. Defaults to "ollama".
            local_port (int, optional): Local port for the ollama container. Defaults to 11434.
        """
        model_name = llm_name.replace("ollama:", "")
        super().__init__(llm_name=model_name, **kwargs)
        self._tag_model() # adds :latest if not present
        self.ollama_client = self._set_local_client(container_name, local_port) # local client instance
        self.available_models = self._get_available_models() # will need to refresh periodically
        self._check_availability()
        
    def _tag_model(self) -> None:
        if ":" not in self.llm_name:
            self.llm_name = f"{self.llm_name}:latest"
        
        
    def _set_local_client(self,
                          container_name: str,
                          port: int) -> Client:
        """
        Set the local client for the Ollama container.

        Args:
            container_name (str): Name of the Ollama container.        self._check_model_availability()
        """
        client = ollama.Client(
            host = f"http://{container_name}:{port}",
            headers = {"Content-Type": "application/json"}
        )
        return client


    def _get_available_models(self) -> List[str]:
        """
        Get the list of available models in the Ollama container.

        Returns:
            List[str]: List of available model names.
        """
        available_models = self.ollama_client.list()
        return [model.model for model in available_models.models]
    
    
    def _is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available in the Ollama container.

        Args:
            model_name (str): Name of the model to check.

        Returns:
            bool: True if the model is available, False otherwise.
        """
        return model_name in self.available_models
    
    
    def _check_availability(self) -> None:
        """
        Check if the Ollama container is available and the specified model is available.
        Raises an exception if the container or model is not available.
        """
        is_available = self._is_model_available(self.llm_name)
        if is_available:
            print(f"Model {self.llm_name} is available in the Ollama container.")
        else:
            print(f"Pulling model {self.llm_name} from Ollama.")
            self.ollama_client.pull(self.llm_name)
            
    
    def __call__(self,
                 question: str,
                 context: Union[str, Dict[str, Any]] = {}) -> tuple[str, str]:
        """
        Process the question and generate a response using Ollama.
        
        Args:
            question (str): User question in string.
            context (Union[str, Dict[str, Any]], optional): Context for the question. Defaults to {}.
            
        Returns:
            Tuple of (response, raw_response)
        """
        # Generate user prompt with question and context
        user_prompt: str = self.user_prompter(question, context=context)
        
        # Combine system prompt with user prompt
        combined_prompt: list[dict] = self.system_prompter(user_prompt, context=context)
        
        # Chat
        chat_response = self.ollama_client.chat(
            model = self.llm_name,
            messages = combined_prompt,
        )
        raw_response = chat_response["message"]["content"]
        
        # Apply truncation if enabled
        response = self.truncate_response(raw_response) if self.is_truncate_response else raw_response
        
        return response, raw_response    
            
    
    def quit(self):
        """
        Clean up any resources. No specific cleanup needed for API-based implementation.
        """
        pass  # No resources to clean up for REST API implementation
    
# if __name__ == "__main__":
#     ollama_instance = OllamaContainer(container_name = "localhost")
#     response, raw_respose = ollama_instance.__call__("what is the capital of greece?")
#     print("Response:", response)
#     print("Raw Response:", raw_respose)