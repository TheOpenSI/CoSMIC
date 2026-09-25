### Core modules ###
from pathlib import Path
import uuid


### Type hints ###


### Internal modules ###
from . import chess as chess_instances
from .base import ServiceBase
from .document_index import DocumentMetadata
from .llms.LLMBase import LLMBase
from .rag import RAGBase
from ...modules.code_generation.code_generation import CodeGenerator
from .system_information_service import SystemInformationService
from .fallback_service import FallbackService
from ...utils.log_tool import set_color



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

            system_information_service (SystemInformationService):
                system information service.

            fallback_service (FallbackService):
                fallback service.

            config (str, optional):
                config file to extract settings. Default to None.
        """
        super().__init__(**kwargs)

        # Set config globally.
        self.query_analyser             = query_analyser
        self.llm                        = llm
        self.rag                        = rag
        self.code_generator             = code_generator
        self.system_information_service = system_information_service
        self.fallback_service           = fallback_service


    def _add_service_0(self, services: dict[str, dict[str, str]]) -> None:
        """
        Add default service 0 to the services dictionary if not present.
        NOTE: This is an inplace mutation of the `services` dictionary.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services to be updated.
        """
        if "0" not in services:
            services["0"] = {
                "name": "system_information",
                "desc": (
                    "Answer questions about the AI assistant itself such as who created it, "
                    "what OpenSI-CoSMIC is, and what it can do."
                )
            }


    def _get_services_name(self, services: dict[str, dict[str, str]]) -> dict[str, str]:
        """
        Extracts the names of the services from the services dictionary.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services.

        Returns:
            dict[str, str]: Dictionary mapping service IDs to their names.
        """
        return {service_id: service_info["name"] for service_id, service_info in services.items()}


    def _force_route_check(self, service_option: str, has_files: bool) -> str:
        """
        If a file is attached but the query analyser routed to "0" (system info)
        or "-1" (fallback) neither of which use RAG, force it into the
        generic RAG (else) branch so the attached file actually gets used.

        Args:
            service_option (str): The current service option selected by the query analyser.
            has_files (bool): Whether files are attached to the query.

        Returns:
            str: The updated service option, if necessary.
        """
        if has_files and service_option in ("0", "-1"):
            return "rag_fallback"
        return service_option


    def _parse_file_refs(self, 
                         file_refs: list[str] | None) -> tuple[list[str], list[str]]:
        """
        Parse `file_refs` (each formatted as "<8-char-hex-file_id>_<original_filename>")
        into the file_id and filename lists used downstream. A ref whose prefix
        isn't a valid 8-char hex file_id is dropped rather than passed through,
        so a malformed ref can't silently corrupt retrieval.

        Args:
            file_refs (list[str] | None): Raw file references attached to the query, 
                e.g. ["a1b2c3d4_report.pdf"].

        Returns:
            document_ids (list[str]): Valid file_ids, used to scope RAG retrieval to 
                the attached files.

            attached_file_names (list[str]): Original filenames, used to tell the LLM 
                which file(s) the question is about.
        """
        document_ids: list[str] = []
        attached_file_names: list[str] = []

        for ref in (file_refs or []):
            parts = ref.split("_", 1)
            doc_id = parts[0]

            if len(doc_id) == 8 and all(c in "0123456789abcdef" for c in doc_id):
                document_ids.append(doc_id)
                raw_name = parts[1] if len(parts) > 1 else ref
                attached_file_names.append(raw_name)
            else:
                print(
                    set_color(
                        "warning",
                        f"[qa] WARNING: unparseable file ref '{ref}' (no 8-hex file_id); skipping."
                    )
                )


        return document_ids, attached_file_names


    def _get_stockfish_binary_path(self) -> str:
        """
        Get the path to the Stockfish binary.
        # TODO: this will be updated once we've service-specific configs implemented

        Returns:
            str: Absolute path to the Stockfish binary.
        """
        return str(
            Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(
                "third_party",
                "stockfish",
                "stockfish-ubuntu-x86-64-avx2"
            )
        )


    def _handle_chess_service(
        self,
        service_option: str,
        service_info_dict: dict,
        context: str | dict,
    ) -> tuple[str, str, int, int]:
        """
        Predict the next chess move ("1.0": from a FEN, "1.1": from a move
        sequence) and ask the LLM to explain why it's the best choice.

        Args:
            service_option (str): "1.0" or "1.1", selecting which Stockfish predictor to use.

            service_info_dict (dict): info dict from the query analyser; carries "fen" for "1.0"
                or "moves" for "1.1".

            context (str | dict): context associated with the question; used to derive the
                move notation mode.

        Returns:
            response (str): truncated answer if applicable.

            raw_response (str): original answer from LLM.

            input_token (int): total amount of input tokens captured by LLM.

            output_token (int): total amount of output tokens generated by LLM.
        """
        binary_path = self._get_stockfish_binary_path()
        move_mode = "algebric" if context == "" else context

        if service_option == "1.0":
            current_fen = service_info_dict["fen"]
            next_move_predictor = chess_instances.StockfishFENNextMove(binary_path=binary_path)
            next_move = next_move_predictor(
                fen=current_fen,
                move_mode=str(object=move_mode),
                topk=5
            )
            move_prediction_context = f"The current chess FEN is {[current_fen]}."

        else: # this is for prediction given moves, service_option == "1.1":
            current_moves = service_info_dict["moves"]
            next_move_predictor = chess_instances.StockfishSequenceNextMove(binary_path=binary_path)
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

        return response, raw_response, input_token, output_token


    # TODO: extract into a dedicated MemoryService class, mirroring
    # system_information_service/fallback_service/code_generator
    def _handle_memory_update_service(
        self,
        query: str,
        services: dict[str, dict[str, str]],
        services_name: dict[str, str],
        memory_service_active: bool,
        user_id: str | None,
    ) -> tuple[str, str]:
        """
        Save the query to the user's long-term memory when the memory
        service is active and enabled; otherwise fall back to a generic
        response.

        Args:
            query (str): the question, saved verbatim as the memory text.

            services (dict[str, dict[str, str]]): dictionary of services, passed through 
                to the fallback service if memory can't be used.

            services_name (dict[str, str]): mapping of service id to service name, used 
                to confirm that service "2" is configured as "memory".

            memory_service_active (bool): whether the memory service is enabled for this request.

            user_id (str | None): the current user, required to scope the saved memory.

        Returns:
            response (str): truncated answer.

            raw_response (str): original answer.
        """
        selected_service_name = services_name.get("2", "").lower()

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

            return "Saved to your memory.", "Saved to your memory."

        # fallback to generic response if memory service is not active or not configured
        return self.fallback_service(services=services)


    def _is_rag_eligible(self, 
                         services: dict[str, dict[str, str]], 
                         service_option: str) -> bool:
        """
        Check if the selected service is eligible for RAG.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services.
            service_option (str): Selected service option.

        Returns:
            bool: True if the service is eligible for RAG, False otherwise.
        """
        try:
            target_service = services[service_option]
            return target_service.get("memory_capability") is True
        
        except KeyError:
            print(
                set_color(
                    "info",
                    f"[qa] Service option '{service_option}' has no entry in services "
                    "(expected for the forced-route fallback); treating as not RAG-eligible."
                )
            )
            return False


    def _use_rag(self,
                 services: dict[str, dict[str, str]],
                 service_option: str,
                 is_rag: bool,
                 has_files: bool) -> bool:
        """
        Determine whether to use RAG based on the service option, RAG eligibility,
        and whether files are attached.

        Args:
            services (dict[str, dict[str, str]]): Dictionary of services.
            service_option (str): Selected service option.
            is_rag (bool): Whether RAG is enabled for this request.
            has_files (bool): Whether files are attached to the query.

        Returns:
            bool: True if RAG should be used, False otherwise.
        """
        is_rag_eligible_service = self._is_rag_eligible(services, service_option)
        use_rag = (
                (is_rag) and
                (is_rag_eligible_service or has_files)
            )

        return use_rag


    def _split_chat_history_context(self, context: str) -> tuple[str, str]:
        """
        Split context into its chat-history and non-history parts, based on
        whether it carries the "Conversation History:" marker.

        Args:
            context (str): context associated with the question.

        Returns:
            chat_history_context (str): context if it holds chat history, else "".
            rag_context (str): context if it doesn't hold chat history, else "".
        """
        is_chat_history = "Conversation History:" in context
        return (context, "") if is_chat_history else ("", context)


    def _merge_retrieved_context(
        self,
        context: str | dict,
        chat_history_context: str,
        context_retrieved: str,
        attached_file_names: list[str],
    ) -> str | dict:
        """
        Fold RAG-retrieved context back into `context`, alongside chat
        history and a note about which attached file(s) the question is
        about.

        Args:
            context (str | dict): context associated with the question.
            chat_history_context (str): chat history part of the context, if any.
            context_retrieved (str): context retrieved by RAG.
            attached_file_names (list[str]): original filenames of attached files.

        Returns:
            str | dict: context updated with the retrieved information.
        """
        file_note = (
            f"The user attached the file(s): {', '.join(attached_file_names)}. "
            "Answer using the retrieved context from that file.\n"
            if attached_file_names else ""
        )

        suffix = (
            ""
            if   (context_retrieved == "")
            else (f"\n{file_note}Context:\n {context_retrieved}")
        )

        if isinstance(context, dict):
            context.update({"context": f"{chat_history_context}{suffix}"})
        else:
            context = f"{chat_history_context}{suffix}"

        return context


    def _retrieve_rag_context(
        self,
        query: str,
        context: str | dict,
        user_id: str | None,
        session_id: str | None,
        global_service_names: list[str] | None,
        memory_service_active: bool,
        document_ids: list[str],
        attached_file_names: list[str],
    ) -> tuple[str, str | dict, float | int]:
        """
        Retrieve RAG context for the query and fold it into `context`.

        Args:
            query (str): the question.
            context (str | dict): context associated with the question.
            user_id (str | None): the current user, used to scope retrieval.
            session_id (str | None): the current session, used to scope retrieval.
            global_service_names (list[str] | None): global services to include in retrieval.
            memory_service_active (bool): whether the user memory tier should be included.
            document_ids (list[str]): file_ids to scope retrieval to attached files.
            attached_file_names (list[str]): original filenames of attached files.

        Returns:
            user_prompt (str): prompt built from the query and non-history context.
            context (str | dict): context updated with the retrieved information.
            retrieve_score (float | int): score of the context retrieval.
        """
        chat_history_context, rag_context = self._split_chat_history_context(context=context)

        # If retrieving context, first generate the user prompt given the
        # user prompter format.
        user_prompt = self.llm.user_prompter(
            query,
            context=rag_context
        ) # pyright: ignore[reportCallIssue]

        # Get the retrieved context, scoped to the active memory tiers
        (
            context_retrieved,
            retrieved_context_score
        ) = self.rag( # pyright: ignore[reportAssignmentType]
            query,
            user_id = user_id,
            session_id = session_id,
            global_service_names = global_service_names,
            include_session = True,
            include_user = memory_service_active,
            include_global = True,
            document_ids = document_ids or None
        )

        retrieve_score = (
            max(retrieved_context_score)
            if   (retrieved_context_score)
            else (-1)
        )

        context = self._merge_retrieved_context(
            context = context,
            chat_history_context = chat_history_context,
            context_retrieved = context_retrieved,
            attached_file_names = attached_file_names
        )

        return user_prompt, context, retrieve_score


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
        file_refs:              list[str] | None    = None,
    ) -> tuple[str, str, int, int ,float | int]:
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

            is_rag  (bool, optional):
                if retrieve context for the question. Defaults to False.

            verbose (bool, optional):
                debug mode. Default to False.

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
        """
        # Set initial return answers.
        response:       str | None  = None
        raw_response:   str | None  = None
        input_token:    int         = 0
        output_token:   int         = 0
        retrieve_score: float | int = -1

        
        # Add default service 0
        self._add_service_0(services=services)

        # cut down to just "name" of the services not its "desc"
        services_name: dict[str, str] = self._get_services_name(services=services)

        # Get service option & minimal extra info about selected service through
        # `query_analyser.py`.
        (
            service_option,
            service_info_dict
        ) = self.query_analyser(
            query,
            services=services # pyright: ignore[reportCallIssue]
        )

        # Check if force route is required.
        service_option = self._force_route_check(service_option, has_files)

        # Parse file_refs into validated document_ids / attached_file_names
        document_ids, attached_file_names = self._parse_file_refs(file_refs)

        # TODO: Remove chess service
        if service_option.startswith("1."):
            (
                response,
                raw_response,
                input_token,
                output_token
            ) = self._handle_chess_service(
                service_option = service_option,
                service_info_dict = service_info_dict,
                context = context
            )

        elif service_option == "2":
            (
                response,
                raw_response
            ) = self._handle_memory_update_service(
                query = query,
                services = services,
                services_name = services_name,
                memory_service_active = memory_service_active,
                user_id = user_id
            )

        elif service_option == "3":
            (
                response,
                raw_response
            )= self.code_generator(query)


        # system information service 
        elif service_option == "0":
            (
                response,
                raw_response,
                input_token,
                output_token
            ) = self.system_information_service(
                query = query,
                services = services,
                context = context
            )

        # When all services are disabled and service 0 cannot answer, query analyser will return -1
        elif service_option == "-1":
            response, raw_response = self.fallback_service(services = services)

        else:
            if self._use_rag(services, service_option, is_rag, has_files):
                (
                    user_prompt,
                    context,
                    retrieve_score
                ) = self._retrieve_rag_context(
                    query = query,
                    context = context,
                    user_id = user_id,
                    session_id = session_id,
                    global_service_names = global_service_names,
                    memory_service_active = memory_service_active,
                    document_ids = document_ids,
                    attached_file_names = attached_file_names
                )

            # Non-RAG services (or when RAG is disabled) fall through here.
            else:
                user_prompt = query
                retrieve_score  = -1

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
        # token, retrieve score, etc) over to opensi_cosmic.py. This's the
        # 2nd time QA sent user query + selected service prompt (from
        # query_analyser.py). The latter is different based on which condition
        # above get hit

        # Sum up I/O tokens from 1st Ollama call (in `service_info_dict`) with
        # I/O tokens from 2nd Ollama call (in either condition above)
        total_input_token:  int = service_info_dict["input_token"] + input_token
        total_output_token: int = service_info_dict["output_token"] + output_token

        return (
            str(response),
            str(raw_response),
            total_input_token,
            total_output_token,
            retrieve_score
        )
