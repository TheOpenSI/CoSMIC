### Core modules ###
from pathlib import Path
import logging
import uuid


### Type hints ###
from typing import Any


### Internal modules ###
from . import chess as chess_instances
from .base import ServiceBase
from .document_index import DocumentMetadata
from .llms.LLMBase import LLMBase
from .rag import RAGBase
from ...modules.code_generation.code_generation import CodeGenerator
from .system_information_service import SystemInformationService
from .fallback_service import FallbackService



logger: logging.Logger = logging.getLogger(__name__)


class EmptyServiceError(Exception):
    """
    Raised when no active service is available for the Query Analyser.
    """


class QABase(ServiceBase):
    def __init__(
        self,
        query_analyser: LLMBase,
        llm:            LLMBase,
        rag:            RAGBase,
        code_generator: CodeGenerator,
        system_information_service: SystemInformationService,
        fallback_service: FallbackService,
        config:         str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Base class for QA.

        Args:
            query_analyser (LLMBase):
                query analyser.

            llm (LLMBase):
                LLM instance.

            rag (RAGBase):
                RAG instance containing vector database service.

            code_generator (CodeGenerator):
                code generation service.

            system_information_service (SystemInformationService):
                system information service.

            fallback_service (FallbackService):
                fallback service.

            config (str, optional):
                config file to extract settings. Defaults to None.
        """
        super().__init__(**kwargs)

        # Set config globally.
        self.query_analyser             = query_analyser
        self.llm                        = llm
        self.rag                        = rag
        self.code_generator             = code_generator
        self.system_information_service = system_information_service
        self.fallback_service           = fallback_service

        return None


    @staticmethod
    def _parse_file_refs(
        file_refs: list[str] | None
    ) -> tuple[list[str], list[str]]:
        """
        Split the attached file refs sent by the frontend.

        The frontend sends each attached file as "<file_id>_<file_name>", where
        file_id is the 8-char hex id returned by `/memory/upload`. A ref without
        a valid file_id is skipped, because it would filter retrieval down to
        nothing.

        Args:
            file_refs (list[str] | None):
                attached file refs.

        Returns:
            document_ids (list[str]):
                file ids used to limit retrieval to the attached files.

            attached_file_names (list[str]):
                file names used to tell the LLM which file the question is about.
        """
        document_ids:           list[str] = []
        attached_file_names:    list[str] = []

        for ref in (file_refs or []):
            parts:      list[str]   = ref.split("_", 1)
            file_id:    str         = parts[0]

            if len(file_id) == 8 and all(c in "0123456789abcdef" for c in file_id):
                document_ids.append(file_id)
                attached_file_names.append(parts[1] if len(parts) > 1 else ref)

            else:
                logger.warning(
                    f"Unparseable file ref '{ref}' (no 8-hex file_id); skipping."
                )

        return document_ids, attached_file_names


    def __call__(
        self,
        query:      str,
        # NOTE:
        # for legacy purposes. Change to `dict[int, dict[str, str]]` type when
        # update to handle `int` properly
        services:               dict[str, dict[str, str]],
        context:                str | dict          = "",
        is_rag:                 bool                = False,
        verbose:                bool                = False,
        user_id:                str | None          = None,
        session_id:             str | None          = None,
        global_service_names:   list[str] | None    = None,
        memory_service_active:  bool                = False,
        has_files:              bool                = False,
        file_refs:              list[str] | None    = None
    ) -> tuple[str, str, int, int, float | int]:
        """
        Process each QA.

        Args:
            query (str):
                a question.

            services (dict[str, dict[str, str]]):
                all available services for Query Analyser, which then being used
                to re-structure question with its prompting techniques.

            context (str | dict, optional):
                context associated with the question. Defaults to "".

            is_rag (bool, optional):
                if retrieve context for the question. Defaults to False.

            verbose (bool, optional):
                debug mode. Defaults to False.

            user_id (str | None, optional):
                owner of the session and user memory. Defaults to None.

            session_id (str | None, optional):
                chat session of the session memory. Defaults to None.

            global_service_names (list[str] | None, optional):
                services whose global memory is searched. Defaults to None.

            memory_service_active (bool, optional):
                if the memory service is enabled. Defaults to False.

            has_files (bool, optional):
                if the user attached files to the question. Defaults to False.

            file_refs (list[str] | None, optional):
                attached file references in "<8-hex file_id>_<name>" form.
                Defaults to None.

        Returns:
            response (str):
                truncated answer if applicable.

            raw_response (str):
                original answer from LLM.

            input_token (int):
                total amount of input tokens captured by LLM during each
                QA process.

            output_token (int):
                total amount of output tokens generated by LLM during each
                QA process.

            retrieve_score (float | int):
                highest score among the retrieved chunks (if applicable).
                Defaults to -1 for non-RAG services or when nothing is retrieved.

        Raises:
            EmptyServiceError: no active service is available.
        """
        # Set initial return answers.
        response:       str | None  = None
        raw_response:   str | None  = None
        input_token:    int         = 0
        output_token:   int         = 0
        retrieve_score: float | int = -1

        # Add default service 0.
        if "0" not in services:
            services["0"] = {
                "name": "system_information",
                "desc": (
                    "Answer questions about the AI assistant itself such as who "
                    "created it, what OpenSI-CoSMIC is, and what it can do."
                )
            }

        # Cut down to just "name" of the services not its "desc".
        services_name: dict[str, str] = {}
        for (service_id, service_info) in services.items():
            services_name[service_id] = service_info["name"]

        # No active services found from fetched API endpoint.
        if len(services_name) == 0:
            raise EmptyServiceError(
                "No active services found from fetched API endpoint. At least 1 "
                "service is required for Query Analyser."
            )

        # Get service option & minimal extra info about selected service through
        # `query_analyser.py`.
        (
            service_option,
            service_info_dict
        ) = self.query_analyser(
            query,
            services=services # pyright: ignore[reportCallIssue]
        )

        # If a file is attached but the query analyser routes to "0" (system info)
        # or "-1" (fallback), neither of which use RAG, force it into the generic
        # RAG branch so the attached file actually gets used.
        if has_files and service_option in ("0", "-1"):
            service_option = "4"

        # Skip query as required or unknown service option.
        if query.find("skip") > -1:
            return (
                str(response),
                str(raw_response),
                input_token,
                output_token,
                retrieve_score
            )

        # Split attached file refs into ids (to limit retrieval to those files)
        # and names (to tell the LLM which file the question is about).
        (
            document_ids,
            attached_file_names
        ) = self._parse_file_refs(file_refs=file_refs)

        # Process query with service parsing.
        if service_option.find("1.") > -1:
            if service_option == "1.0":
                # Set game move mode.
                move_mode = (
                    "algebric"
                    if   (context == "")
                    else (context)
                )

                # Get chess FEN.
                current_fen = service_info_dict["fen"]

                # TODO:
                # this will be updated again once we've service-specific configs
                # implemented
                binary_path: str = str(
                    Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(
                        "third_party",
                        "stockfish",
                        "stockfish-ubuntu-x86-64-avx2"
                    )
                )

                next_move_predictor = chess_instances.StockfishFENNextMove(binary_path=binary_path)

                # Predict the next move.
                next_move = next_move_predictor(
                    fen=current_fen,
                    move_mode=str(object=move_mode),
                    topk=5
                )

                # Set the response with question and next move.
                move_prediction_context = f"The current chess FEN is {[current_fen]}."

            # this is for prediction given moves, service_option == "1.1":
            else:
                # Set game move mode.
                move_mode = (
                    "algebric"
                    if   (context == "")
                    else (context)
                )

                # Get moves.
                current_moves = service_info_dict["moves"]

                # TODO:
                # this will be updated again once we've service-specific configs
                # implemented
                binary_path: str = str(
                    Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(
                        "third_party",
                        "stockfish",
                        "stockfish-ubuntu-x86-64-avx2"
                    )
                )

                next_move_predictor = chess_instances.StockfishSequenceNextMove(binary_path=binary_path)

                # Predict the next move.
                next_move = next_move_predictor(
                    moves=current_moves,
                    move_mode=str(object=move_mode),
                    topk=5
                )

                move_prediction_context = f"The previous chess moves are {[current_moves]}."

            # Explain why these moves are feasible.
            user_prompt = f"Select the best next move from {next_move} and explain why it is the best."
            (
                response,
                raw_response,
                input_token,
                output_token
            ) = self.llm(
                question=user_prompt,
                context=move_prediction_context
            )

            # Attach all the moves in case LLM cannot select the best one.
            response        = f"The next moves are from {next_move}.\n{response}"
            raw_response    = f"The next moves are from {next_move}.\n{raw_response}"

        elif service_option == "2":
            selected_service_name = services_name.get(service_option, "").lower()

            if (
                memory_service_active
                and (selected_service_name == "memory")
                and (user_id)
            ):
                payload = DocumentMetadata(
                    document_id=uuid.uuid4().hex[:8],
                    user_id=user_id,
                    memory_type="user",
                ).to_vector_payload()

                self.rag.vector_database.update_database_from_text(
                    text=query,
                    extra_metadata=payload
                )

                response = raw_response = "Saved to your memory."

            # Memory service is disabled or the user is unknown, so it cannot answer.
            else:
                response, raw_response = self.fallback_service(services=services)

        elif service_option == "3":
            (
                raw_response,
                response
            ) = self.code_generator(query)

        # Add system information service.
        elif service_option == "0":
            (
                response,
                raw_response,
                input_token,
                output_token
            ) = self.system_information_service(
                query=query,
                services=services,
                context=context
            )

        # When all services are disabled and service 0 cannot answer, query analyser
        # will return -1.
        elif service_option == "-1":
            response, raw_response = self.fallback_service(services=services)

        # The query analyser returned an option that is not an active service.
        # Attached files still go to the RAG branch below.
        elif (service_option not in services) and (not has_files):
            logger.warning(
                f"Query analyser returned unknown service option '{service_option}'; "
                "using fallback service."
            )
            response, raw_response = self.fallback_service(services=services)

        else:
            is_rag_eligible_service = (
                services.get(service_option, {}).get("memory_capability") is True
            )

            execute_rag = (
                (is_rag) and
                (is_rag_eligible_service or has_files)
            )

            if execute_rag:
                # Check if context is chat history.
                chat_history_context = (
                    context
                    if   ("Conversation History:" in context)
                    else ("")
                )

                rag_context = (
                    ""
                    if   ("Conversation History:" in context)
                    else (context)
                )

                # If retrieving context, first generate the user prompt given the
                # user prompter format.
                user_prompt = self.llm.user_prompter(
                    query,
                    context=rag_context
                ) # pyright: ignore[reportCallIssue]

                # Get the retrieved context, scoped to the active memory tiers.
                (
                    context_retrieved,
                    retrieved_context_score
                ) = self.rag(
                    query,
                    user_id=user_id,
                    session_id=session_id,
                    global_service_names=global_service_names,
                    include_session=True,
                    include_user=(memory_service_active and is_rag_eligible_service),
                    include_global=is_rag_eligible_service,
                    document_ids=document_ids or None
                )

                # Report the best chunk score, or -1 if nothing was retrieved.
                retrieve_score = (
                    max(retrieved_context_score)
                    if   (retrieved_context_score)
                    else (-1)
                )

                # Tell the LLM which file the question is about.
                file_note = (
                    f"The user attached the file(s): {', '.join(attached_file_names)}. "
                    "Answer using the retrieved context from that file.\n"
                    if attached_file_names else ""
                )

                # Remain the other variables in context if it is a dictionary,
                # otherwise overwrite it.
                suffix = (
                    ""
                    if   (context_retrieved == "")
                    else (f"\n{file_note}Context:\n {context_retrieved}")
                )

                if isinstance(context, dict):
                    context.update({"context": f"{chat_history_context}{suffix}"})

                else:
                    context = f"{chat_history_context}{suffix}"

            # Non-RAG services (or when RAG is disabled) fall through here.
            else:
                user_prompt     = query
                retrieve_score  = -1

            # Get the response with retrieved context if applicable.
            (
                response,
                raw_response,
                input_token,
                output_token
            ) = self.llm(
                question=user_prompt,
                context=context,
                service_name=services_name.get(service_option, "")
            )

        # NOTE:
        # Last step to transfer final response with some extra info (I/O
        # token, retrieve score, etc) over to `opensi_cosmic.py`. This's the
        # 2nd time QA sent user query + selected service prompt (from
        # `query_analyser.py`). The latter is different based on which condition
        # above get hit.

        # Sum up I/O tokens from 1st Ollama call (in `service_info_dict`) with
        # I/O tokens from 2nd Ollama call (in either condition above).
        total_input_token:  int = service_info_dict["input_token"] + input_token
        total_output_token: int = service_info_dict["output_token"] + output_token

        return (
            str(response),
            str(raw_response),
            total_input_token,
            total_output_token,
            retrieve_score
        )
