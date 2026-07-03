### Core modules ###
from sys import exit, stdout
from pathlib import Path
from typing import Any
from fastapi import HTTPException, status
from torch import cuda
from httpx import Client, ConnectError, ConnectTimeout, Response
from pprint import pp

### Type hints ###


### Internal modules ###
from .services.llms import llm as llm_instances
from .maps import LLM_INSTANCE_DICT
from .services.vector_database import VectorDatabase
from .services.qa import QABase
from .services.rag import RAGBase
from ..modules.chess.chess_qa_puzzle import PuzzleAnalyse
from ..modules.chess.chess_qa_quality import QualityEval
from ..modules.chess.chess_genfen import FENGenerator
from ..modules.chess.chess_gencot import CotGenerator
from ..modules.code_generation.code_generation import CodeGenerator
from .services.system_information_service import SystemInformationService
from ..utils.log_tool import set_color
from ..utils.module import get_instance
from .query_analyser.query_analyser import QueryAnalyser


class OpenSICoSMIC:
    def __init__(
        self, general_slm: str = "", qa_slm: str = "", user: dict | None = None
    ) -> None:
        """
        Construct OpenSICoSMIC instance. It contains SLMs and services including
        vector database and RAG, where RAG includes context retriever and vector
        database update. Chess services are included in `PuzzleAnalyse()` and
        `QualityEval()` classes (called on demand, not as global instance).

        Args:
            general_slm (str):
                General SLM name used in every chat session (check default list
                of SLMs in `src/maps.py`). If empty, the entry is a combination
                format with value from `provider` & `model` fields (accessible
                through `self.general_config_data`).

            qa_slm (str):
                Query Analyser SLM name (check default list of SLMs in
                `src/maps.py`). If empty, the entry is a combination format with
                value from `provider` & `model` fields (accessible through
                `self.qa_config_data`).

            user (dict, optional):
                user information including ID, name, etc. Default to None.
        """
        # NOTE:
        # Because of how db table works, we could add multiple preset of CoSMIC
        # default configs. While there're no usecase for this feature yet (there
        # aren't any extra ones either beside the only inserted default configs
        # data through Alembic script), it's good to mentioned here so that
        # future devs can work on this once CoSMIC is getting more complexed and
        # in need of this feature. For now, we'll assume to use the first result
        # only.

        # For-loop here would be too expensive so I used this trick instead.
        # Inspired from:
        # https://stackoverflow.com/questions/61105986/how-to-access-elements-in-a-dict-values
        self.config_data: dict[str, Any] = list((self.get_configs()[0]).values())[0]
        self.general_config_data: dict[str, Any] = self.config_data["general"]
        self.qa_config_data: dict[str, Any] = self.config_data["query_analyser"]

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Default General configs: {self.general_config_data}"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"Default QA configs: {self.qa_config_data}"
        #     )
        # )

        # Set user info
        self.user_id = (
            str(user["id"])
            if ((user is not None) and ("id" in user) and (user["id"] != ""))
            else (None)
        )

        # Set model device.
        self.device: str = "cuda" if (cuda.is_available()) else ("cpu")

        # Check OpenAI API key.
        self.openai_api_status: str | None = self.check_openai_key()

        # Read SLMs from Configs API endpoint if not specified
        self.general_slm = general_slm

        if self.general_slm == "":
            # Match model detection format in `src/services/llms/LLMBase.py`.
            # For example: "ollama:qwen2.5:7b"
            self.general_slm = f"{self.general_config_data["provider"]}:{self.general_config_data["model"]}"

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Set General SLM: [{self.general_slm}]"
        #     )
        # )

        self.llm = self.get_llm(
            llm_name=self.general_slm,
            seed=self.general_config_data["seed"],
            is_quantised=self.general_config_data["is_quantised"],
            device=self.device,
        )

        # Read SLMs from Configs API endpoint if not specified
        self.qa_slm = qa_slm

        if self.qa_slm == "":
            # Match model detection format in `src/services/llms/LLMBase.py`.
            # For example: "ollama:qwen2.5:7b"
            self.qa_slm = (
                f"{self.qa_config_data["provider"]}:{self.qa_config_data["model"]}"
            )

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Set QA SLM: [{self.qa_slm}]"
        #     )
        # )

        self.query_analyser: QueryAnalyser = QueryAnalyser(
            llm_name=self.qa_slm,
            seed=self.qa_config_data["seed"],
            is_quantised=self.qa_config_data["is_quantised"],
            service_index=-1,  # Default to 'auto' mode
            device=self.device,
        )

        # Code generation service.
        self.code_generator: CodeGenerator = CodeGenerator()
        self.system_information_service: SystemInformationService = (
            SystemInformationService(llm=self.llm)
        )

        # Initialise & setup QA instance.
        self.qa: QABase | None = None

        self.set_up_qa(user_id=str(self.user_id), user_name=None)

        return None

    def __call__(
        self, question: str, context: str = "", log_file: str | None = None
    ) -> tuple:
        """
        Execute QA.

        Args:
            question (str):
                a question or a .csv containing multiple questions.

            context (str, optional):
                context for this question. Defaults to "".

            log_file (str, optional):
                whether to print the result in a .txt file. Defaults to None.

        Returns:
            response (str):
                (truncated) response.

            raw_response (str):
                raw response from LLM without truncations.

            retrieve_score (float):
                context retrieve score if `is_rag=True`.
        """
        # print(
        #     set_color(
        #         status="info",
        #         information=f"General configs used when query received: {self.general_config_data}"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"QA configs used when query received: {self.qa_config_data}"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"General SLM used when query received: [{self.general_slm}]"
        #     )
        # )
        # print(
        #     set_color(
        #         status="info",
        #         information=f"QA SLM used when query received: [{self.qa_slm}]"
        #     )
        # )

        # Set initial output to return.
        response: str | None = None
        raw_response: str | None = None
        retrieve_score: float | int = -1

        # Check if OpenAI API key is valid.
        if self.openai_api_status != "":
            return (str(response), str(raw_response), retrieve_score)

        # Chat-mode LLM do not need example in the system prompt.
        if self.llm.llm_name in ["mistral-7b-instruct-v0.1", "gemma-7b-it"]:
            use_example = False

        else:
            use_example = True

        # Check if the question is a string or a .csv file containing multiple sub-questions.
        if question.find(".csv") > -1:
            # Batch process for puzzles (move prediction and analysis) and 4 other quality evaluations.
            if question.find("puzzle") > -1:
                # Do not truncate from the LLM base class but instead using
                # PuzzleAnalyse.is_truncate_response=True.
                self.llm.set_truncate_response(False)

                # Truncation external to the system_prompter does not require keywords, no example
                # in the system prompt is required.
                self.llm.system_prompter.set_use_example(False)

                # Build PuzzleAnalyse QA service.
                puzzle_analyser = PuzzleAnalyse(
                    llm=self.llm,
                    rag=self.rag,
                    is_rag=False,
                    log_file=log_file,
                    is_truncate_response=True,
                    next_move_predict_backend="stockfish",
                )

                # Batch process the question file.
                average_score = puzzle_analyser.batch_process(question)

                # Return how many questions get correct best move prediction
                # using Stockfish or GPT API.
                response = f"Success rate of {question} is {average_score:.2f}."

            elif (
                (question.find("attention") > -1)
                or (question.find("memory") > -1)
                or (question.find("perception") > -1)
            ):
                # Manually switch on and off RAG for specific questions.
                if (question.find("attention") > -1) or (
                    question.find("memory_update") > -1
                ):
                    is_rag = True

                else:
                    is_rag = False

                # Truncate the response according to the system_prompt format.
                self.llm.set_truncate_response(True)

                # Use example's keywords for truncation for non-chat LLM.
                self.llm.system_prompter.set_use_example(use_example)

                # Build quality evaluation service.
                quality_evaluator = QualityEval(
                    llm=self.llm, rag=self.rag, is_rag=is_rag, log_file=log_file
                )

                # Batch process the question file.
                average_score = quality_evaluator.batch_process(question)

                # Return how many questions get correct answer from LLM compared to GT answer.
                response = f"Success rate of {question} is {average_score:.2f}."

            elif question.find("checkmate_moves") > -1:
                # Generate FEN with moves
                fen_generator = FENGenerator(log_file=log_file)

                # Batch process the question file.
                fen_generator.batch_process(question)

            elif question.find("finetune_dataset") > -1:
                # Generate CoT analysis for finetune dataset.
                cot_generator = CotGenerator(
                    log_file=log_file, is_truncate_response=True
                )

                # Batch process the question file.
                cot_generator.batch_process(question)

        else:
            # General question needs truncation according the system prompt to avoid hallucination.
            self.llm.set_truncate_response(True)

            # Truncation needs keywords from the example of system prompt.
            self.llm.system_prompter.set_use_example(True)

            # Process each question.
            response, raw_response, retrieve_score = self.qa(
                query=question,
                services=self.get_services(),
                context=context,  # Chat history context (see `backend/routers/cosmic.py`)
                is_rag=True,
                verbose=False,
            )  # pyright: ignore

        # Return answers with and without truncation, and retrieve score (if
        # applicable). Otherwise, -1.
        return (response, raw_response, retrieve_score)

    def set_up_qa(self, user_id: str, user_name: str | None = None):
        """
        Set up QA instance by user ID.

        Args:
            user_id     (str):              user ID through front-end.
            user_name   (str, optional):    user name. Defaults to None.
        """
        # Invalid user ID.
        if (user_id == "") and (not isinstance(user_id, str)):
            self.user_id = None

        # For a default user (no user ID), always use the same QA instance.
        if (user_id is None) and (self.qa is not None):
            return -1

        if (user_id != self.user_id) or (self.qa is None):
            # Change the global user ID.
            self.user_id = user_id

            # Create vector database service which will be included in RAG for retrieve and information updates.
            # vector_db_path: Path = Path(self.config_data["rag"]["vector_db_path"]).resolve(strict=True)

            # # If index.faiss exists, it is user selected path; do not change the path.
            # # Otherwise, create a new directory.
            # if not Path.exists(
            #     vector_db_path.joinpath("index.faiss"),
            #     follow_symlinks=True
            # ):
            #     if self.user_id is not None:
            #         # User ID specific.
            #         vector_db_path: Path = vector_db_path.joinpath(self.user_id)
            #     else:
            #         # Set to default folder for easy management.
            #         vector_db_path: Path = vector_db_path.joinpath("default")

            #     # Create the data folder if not exist.
            #     vector_db_path.mkdir(
            #         mode=0o777,
            #         parents=False,
            #         exist_ok=True
            #     )
            # Since Qdrant is now managed, we don't need to check for index.faiss
            # The VectorDatabase service will handle the connection.

            # TODO:
            # I know that we had a different path implmented for RAG works on
            # another branch right now. However, I need to match what already
            # there in the YAML file so COSMIC-225 PR can be merged. Once this
            # merged, we can modify the path again with the current branch
            # working on RAG to test out.

            vector_database = VectorDatabase(local_database_path="", device=self.device)

            # Add a directory of documents.
            document_path: Path = (
                Path(__file__)
                .resolve(strict=True)
                .parent.parent.joinpath("data", "docs")
            )
            documents: str | list[str] = []

            if document_path.exists(follow_symlinks=True):
                vector_database.add_document_directory(str(document_path))

            # Add documents.
            if (documents != "") or (len(documents) > 0):
                vector_database.add_documents(documents)

            # Base RAG service with vector_database, the database can be changed using
            # self.rag.set_vector_database().
            self.rag = RAGBase(vector_database=vector_database)

            # QA module to handle basic types of questions, such __next__move__, __update__store__, and
            # general questions.
            self.qa = QABase(
                query_analyser=self.query_analyser,  # pyright: ignore
                llm=self.llm,
                rag=self.rag,
                code_generator=self.code_generator,
                system_information_service=self.system_information_service,
                config=None,
            )

    def check_openai_key(self):
        """
        Check OpenAI API key valid.

        Returns:
            answer: status information.
        """
        llm_name = f"{self.general_config_data["provider"]}:{self.general_config_data["model"]}"
        query_analyser_llm_name = (
            f"{self.qa_config_data["provider"]}:{self.qa_config_data["model"]}"
        )

        is_llm_name_gpt = llm_name.find("gpt") > -1
        is_query_analyser_llm_name_gpt = query_analyser_llm_name.find("gpt") > -1

        llm_name_list = []

        if is_llm_name_gpt:
            llm_name_list.append(llm_name)

        if (is_query_analyser_llm_name_gpt) and (
            query_analyser_llm_name not in llm_name_list
        ):
            llm_name_list.append(query_analyser_llm_name)

        count = len(llm_name_list)

        openai_api_key: str | None = self.general_config_data["api_key"]

        if (count > 0) and (openai_api_key == ""):
            answer = ""

            if count == 1:
                answer = f"Since '{llm_name_list[0]} is' used, please add valid OPENAI_API_KEY in .env."

            elif count == 2:
                answer = f"Since '{llm_name_list[0]} and {llm_name_list[1]} are' used, please add valid OPENAI_API_KEY in .env."

        else:
            answer = ""

        return answer

    def get_llm(
        self, llm_name: str, seed: int = 0, is_quantised: bool = False, **kwargs
    ):
        """
        Construct LLM give an LLM name.

        Args:
            llm_name        (str):
                LLM name, check LLM_MODEL_DICT in src/maps.py.

            seed            (int):
                LLM content generation seed. Default to 0.

            is_quantised    (bool):
                use quantised LLM. Default to False.

        Return:
            llm (LLMBase): LLM instance.
        """
        # Build LLM instance from class defined in .py if llm_name is supported.
        if llm_name in LLM_INSTANCE_DICT.keys():
            llm_instance_name = LLM_INSTANCE_DICT[llm_name]

        elif llm_name.find("gpt") > -1:
            llm_instance_name = "GPT"

        elif llm_name.find("ollama") > -1:
            llm_instance_name = "Ollama"

        else:
            print(
                set_color(status="error", information=f"Unsupported LLM: {llm_name}.")
            )
            exit(1)

        llm = get_instance(instances=llm_instances, instance_name=llm_instance_name)(
            llm_name=llm_name, seed=seed, is_quantised=is_quantised, **kwargs
        )

        print(set_color(status="info", information=f"LLM instance created: {llm}"))

        return llm

    def quit(self):
        """
        Release memory of LLM and vector embedding model in vector_database.
        """
        self.query_analyser.quit()
        self.llm.quit()
        self.rag.vector_database.quit()

    def get_services(
        self,
        # TODO:
        # create a dedicated util to handle valid URL format
        url: str = "http://backend:8000/api/v1/services/",
        params: dict[str, bool] | None = {"active": True},
        lifetime: float = 10.0,
        verbose: bool = False,
        # NOTE:
        # for legacy purposes. Change to `dict[int, dict[str, str]]` type when update
        # to handle `int` properly
    ) -> dict[str, dict[str, str]]:
        """
        Retrieve all services from API endpoint with 0-based indexing.

        This method fetches service data from the specified endpoint and returns
        a nested dictionary mapping 0-based indices to minimal service info. The
        transformation subtracts 1 from the API's 1-based IDs to create 0-based
        indexing.

        Args:
            url:
                base URL of the services API endpoint. Defaults to
                "http://backend:8000/api/v1/services/".

            params:
                optional query parameter for provided endpoint. Defaults to
                {"active": True} to get active only services.

            lifetime:
                HTTP client timeout in seconds. Defaults to 10.0 seconds.

            verbose:
                Enable pretty-printed debug output of service data. When True,
                prints formatted service data using 'pprint'.

        Returns:
            Nested dictionary mapping 0-based indices to minimal service info.

            Example:
            {
                0: {
                    "name": "chess",
                    "desc": "<a very long description>"
                },
                1: {
                    "name": "memory",
                    "desc": "<a very long description>"
                }
            }

        Raises:
            HTTPException:
                With status code 500 if any connection error occurs
                (ConnectError, ConnectTimeout) or other unexpected exceptions.

        Example:
            >>> services = obj.get_services(verbose=True)
            >>> services[0] # 1st service
            {"name": "chess", "desc": "<a very long description>"}
        """
        services: dict[str, dict[str, str]] = {}

        try:
            with Client(base_url=url, params=params, timeout=lifetime) as client:
                response: Response = client.get(url="")
                response.raise_for_status()
                datas: list[dict[str, Any]] = response.json().get("result", [])

            if len(datas) == 0:
                # No active services available
                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   SERVICES DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}",
                        )
                    )
                    print(
                        "{debug_msg:s}\n{foot_sep:s}".format(
                            debug_msg="No active services available...",
                            foot_sep=f"{'=' * 80}",
                        )
                    )
                    return {}

                else:
                    return {}

            else:
                # There is/are active services available
                for data in datas:
                    # NOTE:
                    # for legacy purposes. Change to normal when update the checking
                    # logic to handle `int` properly
                    services.setdefault(str(data["id"]), {}).update(
                        {"name": data["name"], "desc": data["desc"]}
                    )

                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   SERVICES DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}",
                        )
                    )
                    pp(
                        object=dict(sorted(services.items())),
                        stream=stdout,
                        indent=4,  # Prefer tab over spaces indentation
                    )
                    print(f"{'=' * 80}")
                    return dict(sorted(services.items()))

                else:
                    return dict(sorted(services.items()))

        except ConnectError as httpx_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{httpx_err}"
            )

        except ConnectTimeout as httpx_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{httpx_err}"
            )

    def get_configs(
        self,
        # TODO:
        # create a dedicated util to handle valid URL format
        url: str = "http://backend:8000/api/v1/configs/",
        params: dict[str, bool] | None = None,
        lifetime: float = 10.0,
        verbose: bool = False,
    ) -> list[dict[str, dict[str, Any]]]:
        """
        Retrieve presets of configuration from API endpoint.

        This method fetches presets of configuration data from the specified
        endpoint and returns the exact data structure received for further
        CoSMIC usages.

        Args:
            url:
                base URL of the configurations API endpoint. Defaults to
                "http://backend:8000/api/v1/configs/".

            params:
                optional query parameter for provided endpoint. Defaults to None.

            lifetime:
                HTTP client timeout in seconds. Defaults to 10.0 seconds.

            verbose:
                Enable pretty-printed debug output for presets of configuration
                data. When True, prints formatted presets of configuration data
                using 'pprint'.

        Returns:
            Presets of configuration data.

            Example:
            [
                { <first configuration preset> },
                { <second configuration preset> }
            ]

        Raises:
            HTTPException:
                With status code 500 if any connection error occurs
                (ConnectError, ConnectTimeout) or other unexpected exceptions.

        Example:
            >>> configs = obj.get_configs(verbose=True)
            >>> configs[0] # 1st configuration preset
            [{"<configuration preset name>": "<default configurations>"}]
        """
        configs: list[dict[str, dict[str, Any]]] = []

        try:
            with Client(base_url=url, params=params, timeout=lifetime) as client:
                response: Response = client.get(url="")
                response.raise_for_status()
                datas: list[dict[str, Any]] = response.json().get("result", [])

            if len(datas) == 0:
                # No config presets available
                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   CONFIGURATIONS DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}",
                        )
                    )
                    print(
                        "{debug_msg:s}\n{foot_sep:s}".format(
                            debug_msg="No configuration presets available...",
                            foot_sep=f"{'=' * 80}",
                        )
                    )
                    return []

                else:
                    return []

            else:
                # There is/are config preset(s) available
                for data in datas:
                    # We only need `name` & `details` field to form minimal
                    # config presets data
                    configs.append({data["name"]: data["details"]})

                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   CONFIGURATIONS DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}",
                        )
                    )
                    pp(
                        object=configs,
                        stream=stdout,
                        indent=4,  # Prefer tab over spaces indentation
                    )
                    print(f"{'=' * 80}")
                    return configs

                else:
                    return configs

        except ConnectError as httpx_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{httpx_err}"
            )

        except ConnectTimeout as httpx_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{httpx_err}"
            )
