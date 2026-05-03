### Core modules ###
from sys import stdout
from pathlib import Path
from yaml import safe_load
from httpx import (
    Client,
    Response,
    ConnectError,
    ConnectTimeout
)
from fastapi import (
    HTTPException,
    status
)
from pprint import pp


### Type hints ###
from typing import Any
from ...types.query_analyser import ServicesJsonResponse


### Internal modules ###
from . import chess as chess_instances
from .base import ServiceBase
from .llms.llm import LLMBase
from .rag import RAGBase
from ...modules.code_generation.code_generation import CodeGenerator


class QABase(ServiceBase):
    def __init__(
        self,
        query_analyser: LLMBase,
        llm: LLMBase,
        rag: RAGBase,
        code_generator: CodeGenerator,
        config: str | None = None,
        **kwargs
    ):
        """
        Base class for QA.

        Args:
            query_analyser  (LLMBase):          query analyser.
            llm             (LLMBase):          LLM instance.
            rag             (RAGBase):          RAG instance containing vector database service.
            code_generator  (CodeGenerator):    code generation service.
            config          (str, optional):    config file to extract settings. Default to None.
        """
        super().__init__( **kwargs)

        # Set config globally.
        self.query_analyser: LLMBase = query_analyser
        self.llm: LLMBase = llm
        self.rag: RAGBase = rag
        self.code_generator: CodeGenerator = code_generator
        self.config: str | None = config


    def __call__(
        self,
        query: str,
        context: str | dict = "",
        is_rag: bool = False,
        verbose: bool = False
    ):
        """
        Process each QA.

        Args:
            query   (str):                  a question.
            context (str|dict, optional):   contex associated with the question. Defaults to "".
            is_rag  (bool, optional):       if retrieve context for the question. Defaults to False.
            verbose (bool, optional):       debug mode. Default to False.

        Returns:
            response        (str): truncated answer if applicable.
            raw_response    (str): original answer from LLM.
            retrieve_score       : score of context retrieving if applicable.
        """
        # Set initial return answers.
        response = None
        raw_response = None
        retrieve_score = -1

        # Get service option through query analyser.
        (service_option, service_info_dict) = self.query_analyser(query)

        # Whether this query is related to system information.
        system_information_relevance = service_info_dict["system_information_relevance"]

        # Skip query as required or unknown service option.
        if query.find("skip") > -1:
            return (response, raw_response, retrieve_score)

        # Process query with service parsing.
        if service_option.find("0.") > -1:
            if service_option == "0.0":
                # Set game move mode.
                move_mode = "algebric"  \
                    if context == ""    \
                    else context

                # Get chess FEN.
                current_fen = service_info_dict["fen"]

                # Set up next move predictor as Stockfish.
                self.config_path: Path = Path(str(object=self.config)).resolve(strict=True)
                with self.config_path.open(
                    mode="r",
                    buffering=-1,
                    encoding="utf-8",
                    errors=None,
                    newline=None
                ) as config_file:
                    self.config_data: dict[str, Any] = safe_load(stream=config_file)

                binary_path: str = self.config_data["chess"]["stockfish_path"] if self.config else ""
                next_move_predictor = chess_instances.StockfishFENNextMove(binary_path=binary_path)

                # Predict the next move.
                next_move = next_move_predictor(
                    fen=current_fen,
                    move_mode=str(object=move_mode),
                    topk=5
                )

                # Set the response with question and next move.
                move_prediction_context = f"The current chess FEN is {[current_fen]}."

            # this is for prediction given moves, service_option == "0.1":
            else:
                # Set game move mode.
                move_mode = "algebric"  \
                    if context == ""    \
                    else context

                # Get moves.
                current_moves = service_info_dict["moves"]

                # Set up next move predictor as Stockfish.
                self.config_path: Path = Path(str(object=self.config)).resolve(strict=True)
                with self.config_path.open(
                    mode="r",
                    buffering=-1,
                    encoding="utf-8",
                    errors=None,
                    newline=None
                ) as config_file:
                    self.config_data: dict[str, Any] = safe_load(stream=config_file)

                binary_path: str = self.config_data["chess"]["stockfish_path"] if self.config else ""
                next_move_predictor = chess_instances.StockfishSequenceNextMove(binary_path=binary_path)

                # Predict the next move.
                next_move = next_move_predictor(
                    moves=current_moves,
                    move_mode=str(object=move_mode),
                    topk=5
                )

                # Set the response with question and next move.
                move_prediction_context = f"The previous chess moves are {[current_moves]}."

            # Explain why these moves are feasible.
            user_prompt = f"Select the best next move from {next_move} and explain why it is the best."
            (response, raw_response) = self.llm(
                question=user_prompt,
                context=move_prediction_context
            )

            # Attach all the moves in case LLM cannot select the best one.
            response = f"The next moves are from {next_move}.\n{response}"
            raw_response = f"The next moves are from {next_move}.\n{raw_response}"

        elif service_option == "1":
            # Check if context is a .pdf.
            is_a_document = service_info_dict["is_a_document"]

            if is_a_document:
                # Get absolute document path.
                document_path = service_info_dict["document_path"]

                if document_path is not None:
                    # Update the knowledge database and return the status.
                    self.rag.vector_database.update_database_from_document(document_path=document_path)
            else:
                # Get text.
                text = service_info_dict["text"]

                if text is not None:
                    # Add text to database.
                    self.rag.vector_database.update_database_from_text(text=text)

            response = raw_response = "Vector database updated."

        elif service_option == "2":
            raw_response, response = self.code_generator(query)

        else:
            if is_rag:# General question has RAG activated
                # Check if context is chat hostory
                chat_history_context = context              \
                    if "Conversation History:" in context   \
                    else ""

                rag_context = ""                            \
                    if "Conversation History:" in context   \
                    else context

                # If retrieving context, first generate the user prompt given the
                # user prompter format.
                user_prompt = self.llm.user_prompter(
                    query,
                    context=rag_context
                )

                # Get the retrieved context.
                (context_retrieved, retrieve_score) = self.rag(query)

                # Remain the other variables in context if it is a dictionary,
                # otherwise overwrite it.
                suffix = ""                                 \
                    if context_retrieved == ""              \
                    else "\nContext:\n" + context_retrieved

                context = context.update({"context": (chat_history_context + suffix)})  \
                    if isinstance(context, dict)                                        \
                    else chat_history_context + suffix

            else:
                user_prompt = query
                retrieve_score = -1

            # If the question is related to system information,
            # add system information to context.
            if system_information_relevance:
                # Get system information.
                system_information = service_info_dict["system_information"]

                # Add the information to existing context.
                # This context is likely to be chat history.
                if isinstance(context, dict):
                    context["context"] = "{0:s}\n{1:s}\n\n{2:s}".format(
                        "OpenSI System Information:",
                        system_information,
                        context["context"]
                    )
                else:
                    context = "{0:s}\n{1:s}\n\n{2:s}".format(
                        "OpenSI System Information:",
                        system_information,
                        context
                    )

            # Get the response with retrieved context if applicable.

            # NOTE:
            # Due to the way current logic checking for selected service after
            # receiving the response from LLM based on Query Analyser's refined
            # question from user query, we've to do this kind of workaround until
            # it get updates to handle 'int' type correctly (which it should be).
            services_name: dict[int, str] = self._get_service_name(verbose=True)

            # No active services available in the system
            if len(services_name) == 0:
                # TODO: do we want to deny using query analyser or default to using 
                # service 3 - general QA answering?
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "status": "500 - Internal Server Error",
                        "message": "{trig:s}: {cond:s}".format(
                            trig="ServiceNotFoundError",
                            cond="No active services found. Query Analyser requires at least 1 service enabled for usages"
                        )
                    }
                )

            # There is/are active services available in the system
            else:
                legacy_services_name: dict[str, str] = {
                    str(str_id): name \
                    for (str_id, name) in services_name.items()
                }

                (response, raw_response) = self.llm(
                    question=user_prompt,
                    context=context,
                    service_name=legacy_services_name[service_option]
                )

        # Print service name (NOTE: DEBUG only)
        if verbose:
            if  (response is not None) \
            and (service_option in self.query_analyser["full_services"].keys()):
                response += "{0:s} ; {1:s}".format(
                    f"[service: {self.query_analyser["full_services"][service_option]}",
                    f"system info relevance: {system_information_relevance}]"
                )

            else:
                # Found no LLM response due to unknown circumstances
                pass

        else:
            # No need to mind about extra infomation for debugging
            pass

        return (
            response,
            raw_response,
            retrieve_score
        )


    def _get_service_name(
        self,
        # TODO: util to dynamically check for valid endpoint format
        endpoint:           str                     = "http://backend:8000/api/v1/services/",
        endpoint_params:    dict[str, bool] | None  = {"active": True},
        lifetime:           float                   = 10.0,
        verbose:            bool                    = False
    ) -> dict[int, str]:
        """
        Retrieve service names from the backend API with 0-based indexing.

        This method fetches service data from the specified endpoint and returns
        a dictionary mapping 0-based indices to service names. The transformation
        subtracts 1 from the API's 1-based IDs to create 0-based indexing.

        Args:
            endpoint: Base URL of the services API endpoint.
                Defaults to "http://backend:8000/api/v1/services/".
            endpoint_params: optional query parameter for provided endpoint.
                Defaults to {"active": True} to get active only services.
            lifetime: HTTP client timeout in seconds.
                Defaults to 10.0 seconds.
            verbose: Enable pretty-printed debug output of service data.
                When True, prints formatted service data using 'pprint'.

        Returns:
            Dictionary mapping 0-based indices to service names.
            Example: {0: "memory", 1: "code_generation"}

        Raises:
            HTTPException: With status code 500 if any connection error occurs
                (ConnectError, ConnectTimeout) or other unexpected exceptions.

        Example:
            >>> services = obj._get_services_name(verbose=True)
            >>> services[0]  # First service name
            'academic_governance'
        """
        services_name_dict: dict[int, str] = {}

        with Client(
            base_url=endpoint,
            params=endpoint_params,
            timeout=lifetime
        ) as client:
            try:
                response:   Response                    = client.get(url="/")
                data:       list[ServicesJsonResponse]  = response.json()["result"]

                if len(data) == 0:
                    # No active services available
                    if verbose:
                        print(
                            "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                                head_sep=f"{'=' * 80}",
                                body_msg="[DEBUG]   SERVICES DATA ('NAME' ONLY)   [DEBUG]",
                                foot_sep=f"{'=' * 80}"
                            )
                        )
                        print(
                            "{debug_msg:s}\n{foot_sep:s}".format(
                                debug_msg="No active services available...",
                                foot_sep=f"{'=' * 80}"
                            )
                        )
                        return services_name_dict

                    else:
                        return services_name_dict

                else:
                    # There is/are active services available
                    for service_data in data:
                        # NOTE: for matching the hard-coded style until updating the logic
                        service_id:     int = service_data["id"] - 1
                        service_name:   str = service_data["name"]

                        services_name_dict.update({service_id: service_name})

                    if verbose:
                        print(
                            "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                                head_sep=f"{'=' * 80}",
                                body_msg="[DEBUG]   SERVICES DATA ('NAME' ONLY)   [DEBUG]",
                                foot_sep=f"{'=' * 80}"
                            )
                        )
                        pp(
                            object=services_name_dict,
                            stream=stdout,
                            indent=4 # Prefer tab over spaces indentation
                        )
                        print(f"{'=' * 80}")
                        return services_name_dict

                    else:
                        return services_name_dict

            except ConnectError as httpx_err:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{httpx_err}"
                )

            except ConnectTimeout as httpx_err:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{httpx_err}"
                )
