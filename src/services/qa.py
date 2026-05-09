### Core modules ###
from pathlib import Path
from httpx import Client
from yaml import safe_load


### Type hints ###
from typing import Any



### Internal modules ###
from . import chess as chess_instances
from .base import ServiceBase
from .llms.llm import LLMBase
from .rag import RAGBase
from ...modules.code_generation.code_generation import CodeGenerator
from ...utils.log_tool import set_color



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
            RAG_ENABLED_SERVICES = ["4"] # Academic QA triggers retrieval
            execute_rag = is_rag and (service_option in RAG_ENABLED_SERVICES)

            if execute_rag:
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

            # TODO: needs to read the services from the DataBase and pass the service name to the prompter.
            services_names = self._get_services_name()
            print(f"Mapped services from DB (Name only): {services_names}")

            (response, raw_response) = self.llm(
                question=user_prompt,
                context=context,
                service_name=services_names[service_option]
            )


        # Print service name.
        # if  verbose                                                             \
        # and (response is not None)                                              \
        # and (service_option in self.query_analyser["full_services"].keys()):
        #     response += "{0:s} ; {1:s}".format(
        #         f"[service: {self.query_analyser["full_services"][service_option]}",
        #         f"system info relevance: {system_information_relevance}]"
        #     )

        return (response, raw_response, retrieve_score)


    def _get_services_name(
        self
    ) -> dict[str, str]:
        try:
            with Client(
                base_url="http://backend:8000/api/v1/services/",
                params={"active": True},
                timeout=10.0
            ) as client:
                response = client.get(url="")
                response.raise_for_status()
                datas = response.json()

            services: dict[str, str] = {}
            for data in datas.get("result", []):
                # NOTE:
                # For legacy purposes. Change to normal when update the checking
                # logic to handle `int` properly
                services[str(data["id"] - 1)] = data["name"]
            return services

        except Exception as fastapi_err:
            print(
                set_color(
                    status="warning",
                    information=f"Could not fetch active services from backend: {fastapi_err}. Fallback to empty..."
                )
            )
            return {}
