# -------------------------------------------------------------------------------------------------------------
# File: llm.py
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
import json
import requests
import subprocess
from typing import Optional, List, Dict, Any

from src.services.llms.llm import LLMBase

class Ollama(LLMBase):
    def __init__(
        self,
        llm_name: str,
        container_name: str = "ollama",
        local_port: int = 11434,
        **kwargs
    ):
        """For Ollama supported LLMs using the official Ollama container.

        Args:
            llm_name (str, optional): Ollama supported LLMs available at https://ollama.com/library.
            container_name (str, optional): Name of the Ollama container. Defaults to "ollama".
            local_port (int, optional): Local port for the ollama container. Defaults to 11434.
        """
        # Extract the model name from "ollama:model_name" format if needed
        model_name = llm_name.replace("ollama:", "")
        
        # Call parent constructor with the clean model name
        super().__init__(llm_name=model_name, **kwargs)
        
        # Container configuration
        self.container_name = container_name
        self.local_port = local_port
        # self.api_url = f"http://{self.container_name}:{self.local_port}/api"
        self.api_url = f"http://localhost:{self.local_port}/api/generate" # use chat for chat mode
        
        # Check if model is available, pull if needed
        self._check_model_availability()


    def _process_model_list(self, model_list: str) -> List[str]:
        """
        Process the model list to extract model names.

        Args:
            model_list (str): The output of the model list command.
        """
        models = []
        # Skip header line and get just the names
        for line in model_list.strip().split('\n')[1:]:  # Skip header
            if line:  # Skip empty lines
                # Split by whitespace and take the first column
                parts = line.split()
                if parts:
                    # Extract model name without tag if present
                    model_name = parts[0].split(':')[0]
                    models.append(model_name)
            
        return models
    
    def _get_model_list(self) -> List[str]:
        """
        Get the list of models available on the ollama server.
        """
        shell_command = ["docker", "exec", self.container_name, "ollama", "list"]
        
        model_list = subprocess.run(shell_command,
                                    text=True,
                                    capture_output=True)
        
        if model_list.returncode != 0:
            print(f"Error executing docker command: {model_list.stderr}")
            return []
            
        return self._process_model_list(model_list.stdout)
    
    def _pull_model(self):
        """
        Pull the model from the server using secure subprocess execution
        """
        print(f"{self.llm_name} not found on the ollama server. Pulling the model...")
        
        pull_command = ["docker", "exec", self.container_name, "ollama", "pull", self.llm_name]
        
        pull_process = subprocess.run(pull_command, 
                                      text=True, 
                                      capture_output=True)
        
        if pull_process.returncode != 0:
            print(f"Error pulling model: {pull_process.stderr}")
            raise RuntimeError(f"Failed to pull model {self.llm_name}")
        
        print(f"Model {self.llm_name} pulled successfully.")
    
    def _check_model_availability(self):
        """
        Check if the model is available on the ollama server.
        """
        available_models = self._get_model_list()
        
        if self.llm_name in available_models:
            print(f"{self.llm_name} already exists on the ollama server.")
        else:
            self._pull_model()
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a single prompt string.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            A formatted string combining all messages
        """
        prompt = ""
        
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '')
            
            # Skip empty messages
            if not content or not content.strip():
                continue
            
            # Format based on role
            if role == 'system':
                prompt += f"System:\n{content.strip()}\n\n"
            elif role == 'user':
                prompt += f"User:\n{content.strip()}\n\n"
        
        # Add the assistant response prompt
        prompt += "Assistant Response: "
        
        return prompt
    
    def _send_chat_request(self, messages: str) -> Dict[str, Any]:
        """
        Send a chat request to the Ollama API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            The API response as a dictionary
        """
        payload = {
            "model": self.llm_name,
            "messages": messages,
            # "prompt": "what is the capital of France?",
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.api_url}", json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama API: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
    
    def __call__(
        self, 
        question: str, 
        context: dict = {}
    ):
        """
        Process the question and generate a response using Ollama.
        
        Args:
            question (str): User question in string.
            context (dict, optional): Context retrieved externally if applicable.
            
        Returns:
            Tuple of (response, raw_response)
        """
        # Generate user prompt with question and context
        user_prompt = self.user_prompter(question, context=context)
        
        # Combine system prompt with user prompt
        system_prompt = self.system_prompter(user_prompt, context=context)
        
        # Format messages for Ollama
        messages = self._format_messages(system_prompt)
        
        # Send request to Ollama
        response_data = self._send_chat_request(system_prompt[0])
        
        # Extract the response content
        raw_response = response_data.get('message', {}).get('content', '')
        
        # Apply truncation if enabled
        response = self.truncate_response(raw_response) if self.is_truncate_response else raw_response
        
        return response, raw_response
    
    def quit(self):
        """
        Clean up any resources. No specific cleanup needed for API-based implementation.
        """
        pass  # No resources to clean up for REST API implementation
    

if __name__ == "__main__":
    # Example usage
    ollama = Ollama(llm_name="ollama:llama3.2", container_name="condescending_gould", local_port=11433)
    response = ollama("What is the capital of France?")
    print(response)  # Should print the response from the model