### Core modules ###
from ollama import Client


### Type hints ###
from typing import Any


### Internal modules ###
from .llm import LLMBase
from .OllamaPullManager import OllamaPullManager



class Ollama(LLMBase):
    def __init__(
        self,
        llm_name:       str = "llama3.2",
        container_name: str = "cosmic-ollama",
        local_port:     int = 11434,
        **kwargs
    ) -> None:
        """
        LLMs using the official Ollama container.

        Args:
            llm_name        (str, optional): Ollama supported LLMs available at https://ollama.com/library.
            container_name  (str, optional): Name of the Ollama container. Defaults to "ollama".
            local_port      (int, optional): Local port for the ollama container. Defaults to 11434.
        """
        model_name = llm_name.replace("ollama:", "")
        super().__init__(llm_name=model_name, **kwargs)
        self._tag_model() # adds :latest if not present
        self.ollama_client = self._set_local_client(container_name, local_port) # local client instance
        self.ollama_pull_manager = OllamaPullManager(
            model_name=self.llm_name,
            mode="stochastic",
            interventions=[85, 95],
            max_retries=3,
            fall_back_interval=60,
            ollama_client=self.ollama_client
        )
        self._check_availability()

        return None


    def _tag_model(self) -> None:
        if ":" not in self.llm_name:
            self.llm_name = f"{self.llm_name}:latest"

        return None


    def _set_local_client(
        self,
        container_name: str,
        port:           int
    ) -> Client:
        """
        Set the local client for the Ollama container.

        Args:
            container_name (str): Name of the Ollama container.
        """
        client: Client = Client(
            host = f"http://{container_name}:{port}",
            headers = {"Content-Type": "application/json"}
        )

        return client


    def _check_availability(self) -> None:
        """
        Check if the Ollama container is available and the specified model is available.
        Raises an exception if the container or model is not available.
        """
        self.ollama_pull_manager.pull_model()

        return None


    def __call__(
        self,
        question:       str,
        context:        str | dict[str, Any]    = {},
        service_name:   str                     = "",
    ) -> tuple[str, str]:
        """
        Process the question and generate a response using Ollama.

        Args:
            question        (str):                              User question in string.
            context         (str | dict[str, Any], optional):   Context for the question. Defaults to {}.
            service_name    (str, optional):                    Name of the service. Defaults to "".
        Returns:
            Tuple of (response, raw_response)
        """
        # Generate user prompt with question and context
        user_prompt: str = self.user_prompter(
            question,
            context=context
        )

        # Combine system prompt with user prompt
        # combined_prompt: list[dict] = self.system_prompter(user_prompt, context=context)
        combined_prompt: list[dict] = self.system_prompter(
            user_prompt,
            context=context,
            service=service_name
        )

        # Chat
        chat_response = self.ollama_client.chat(
            model=self.llm_name,
            messages=combined_prompt,
        )
        raw_response = chat_response["message"]["content"]

        # Apply truncation if enabled
        response = (
            self.truncate_response(raw_response)
            if   (self.is_truncate_response)
            else (raw_response)
        )

        return (
            response,
            raw_response
        )


    def quit(self) -> None:
        """
        Clean up any resources. No specific cleanup needed for API-based implementation.
        """
        # No resources to clean up for REST API implementation
        pass
