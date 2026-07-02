### Core modules ###
from pathlib import Path
from fastapi import (
    HTTPException,
    status
)


### Type hints ###
from typing import Any, Optional, List

### Internal modules ###
from . import chess as chess_instances
from .base import ServiceBase
from .document_index import DocumentMetadata
from .llms.llm import LLMBase
from .rag import RAGBase
from ...modules.code_generation.code_generation import CodeGenerator
from .system_information_service import SystemInformationService
from ...utils.log_tool import set_color
import uuid


class QABase(ServiceBase):
    def __init__(
        self,
        query_analyser: LLMBase,
        llm:            LLMBase,
        rag:            RAGBase,
        code_generator: CodeGenerator,
        system_information_service: SystemInformationService,
        config:         str | None = None,
        **kwargs
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

            config (str, optional):
                config file to extract settings. Default to None.
        """
        super().__init__( **kwargs)

        # Set config globally.
        self.query_analyser     = query_analyser
        self.llm                = llm
        self.rag                = rag
        self.code_generator     = code_generator
        self.system_information_service = system_information_service


        return None


    def __call__(
        self,
        query:      str,
        # NOTE:
        # for legacy purposes. Change to `dict[int, dict[str, str]]` type when
        # update to handle `int` properly
        services:   dict[str, dict[str, str]],
        context:    str | dict  = "",
        is_rag:     bool        = False,
        verbose:    bool        = False,
        user_id:                Optional[str]       = None,
        session_id:             Optional[str]       = None,
        global_service_names:   Optional[List[str]] = None,
        memory_service_active:  bool                = False,
        has_files:              bool                = False,
        file_refs:              Optional[List[str]] = None,
    ) -> tuple:
        """
        Process each QA.

        Args:
            query (str):
                a question.

            services (dict[str, dict[str, str]]):
                all available services for Query Analyser, which then being used
                to re-structure question with its prompting techniques.

            context (str | dict, optional):
                contex associated with the question. Defaults to "".

            is_rag  (bool, optional):
                if retrieve context for the question. Defaults to False.

            verbose (bool, optional):
                debug mode. Default to False.

        Returns:
            response (str):
                truncated answer if applicable.

            raw_response (str):
                original answer from LLM.

            retrieve_score (float):
                score of context retrieving if applicable.
        """
        # Set initial return answers.
        response        = None
        raw_response    = None
        retrieve_score  = -1

        # NOTE:
        # for legacy purposes. Change to `dict[int, str]` type when
        # update to handle `int` properly
        
        
        # Add default service 0
        if '0' not in services:
            services["0"] = {
                "name": "system_information",
                "desc": "Answer questions about the AI assistant itself such as who created it, what OpenSI-CoSMIC is, and what it can do."
                }

        # cut down to just "name" of the services not its "desc"
        services_name: dict[str, str] = {}
        for (service_id,service_info) in services.items():
            services_name[service_id]= service_info['name']

        # No active services found from fetched API endpoint
        if len(services_name) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "404 - Not Found",
                    "message": "{trig:s}: {cond:s}".format(
                        trig="EmptyServiceError",
                        cond="No active services found from fetched API endpoint. At least 1 service is required to for Query Analyser."
                    )
                }
            )

        # Get service option through query analyser.
        (
            service_option,
            service_info_dict
        ) = self.query_analyser(
            query,
            services=services # pyright: ignore
        )

        # Skip query as required or unknown service option.
        if query.find("skip") > -1:
            return (
                response,
                raw_response,
                retrieve_score
            )

        # Validate the prefix is an 8-char hex file_id to avoid malformed ref silently breaks retrieval 
        # document_ids target retrieval at the attached file
        # attached_file_names tell the LLM which file the question is about
        document_ids: List[str] = []
        attached_file_names: List[str] = []
        for ref in (file_refs or []):
            parts = ref.split("_", 1)
            doc_id = parts[0]
            if len(doc_id) == 8 and all(c in "0123456789abcdef" for c in doc_id):
                document_ids.append(doc_id)
                raw_name = parts[1] if len(parts) > 1 else ref
                attached_file_names.append(raw_name.replace("_", " "))
            else:
                print(f"[qa] WARNING: unparseable file ref '{ref}' (no 8-hex file_id); skipping.")
        
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

                # Set the response with question and next move.
                move_prediction_context = f"The previous chess moves are {[current_moves]}."

            # Explain why these moves are feasible.
            user_prompt = f"Select the best next move from {next_move} and explain why it is the best."
            (
                response,
                raw_response
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
                and selected_service_name == "memory"
                and user_id
            ):
                payload = DocumentMetadata(
                    document_id=uuid.uuid4().hex[:8],
                    user_id=user_id,
                    memory_type="user",
                ).to_vector_payload()
                self.rag.vector_database.update_database_from_text(
                    text=query, extra_metadata=payload
                )
                response = raw_response = "Saved to your memory."
                return (response, raw_response, retrieve_score)

        elif service_option == "3":
            (
                raw_response,
                response
            )= self.code_generator(query)


        # Add system information service 
        elif service_option == "0":
            response, raw_response = self.system_information_service(
                query=query, services=services, context=context)
            
        # When all services are disabled and service 0 cannot answer, query analyser will return -1
        elif service_option == "-1":
            response = raw_response = (
                "I can't help with that right now — it may be outside what I'm currently set up to answer, or the right service isn't enabled." "\n"
                "Try rephrasing, or ask about something else I can help with. Otherwise, contact OpenSI team.")

        else:
            RAG_ENABLED_SERVICES = ["5"] # Academic QA triggers retrieval
            execute_rag = (
                (is_rag) and
                (service_option in RAG_ENABLED_SERVICES or has_files)
            )

            if execute_rag:
                # Check if context is chat hostory
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
                ) # pyright: ignore

                # Get the retrieved context, scoped to the active memory tiers
                (
                    context_retrieved,
                    retrieve_score
                ) = self.rag(
                    query,
                    user_id=user_id,
                    session_id=session_id,
                    global_service_names=global_service_names,
                    include_session=True,
                    include_user=memory_service_active,
                    include_global=True,
                    document_ids=document_ids or None,
                )

                # Tell the LLM which file the question is about
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

                context = (
                    context.update({"context": f"{chat_history_context}{suffix}"})
                    if   (isinstance(context, dict))
                    else (f"{chat_history_context}{suffix}")
                ) # pyright: ignore
            
            # Non-RAG services (or when RAG is disabled) fall through here.
            else:
                user_prompt     = query
                retrieve_score  = -1

            # Get the response with retrieved context if applicable.
            (
                response,
                raw_response
            ) = self.llm(
                question=user_prompt,
                context=context,
                service_name=services_name.get(service_option, "")
            )


        # Print service name.
        # if (
        #     (verbose)               and
        #     (response is not None)  and
        #     (service_option in self.query_analyser["full_services"].keys()) # pyright: ignore
        # ):
        #     response += "{0:s}\n{1:s}".format(
        #         f"[Service: {self.query_analyser["full_services"][service_option]}", # pyright: ignore
        #         # f"System info relevance: {system_information_relevance}]"
        #     )

        return (
            response,
            raw_response,
            retrieve_score
        )
