### Core modules ###
from sys import (
    exit,
    stdout
)
from pathlib import Path
from typing import Any
from fastapi import (
    HTTPException,
    status
)
from yaml import safe_load
from torch import cuda
from httpx import (
    Client,
    ConnectError,
    ConnectTimeout,
    Response
)
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
from ..utils.log_tool import set_color
from ..utils.module import get_instance
from .query_analyser.query_analyser import QueryAnalyser



class OpenSICoSMIC:
    def __init__(
        self,
        query_llm_name: str         = "",
        llm_name:       str         = "",
        config_path:    str         = "scripts/configs/config.yaml",
        user:           dict | None = None
    ) -> None:
        """
        Construct OpenSICoSMIC instance. It contains LLM and services including vector database
        and RAG, where RAG includes context retriever and vector database update.
        Chess services are induced in PuzzleAnalyse and QualityEval, called on demand, not as global instance.

        Args:
            query_llm_name  (str):  query analyser LLM name, check LLM_MODEL_DICT in src/maps.py,
                                    if it is empty, the entry is self.config.query_llm_name.
            llm_name        (str):  LLM name, check LLM_MODEL_DICT in src/maps.py, if it is empty, the entry
                                    is self.config.llm_name.
            config_path     (str):  path of configuration file.
            user            (dict): user information including ID, name, etc. Default to None.
        """
        # Check if required config file exists.
        self.config_path: Path = Path(config_path).resolve(strict=True)

        if not self.config_path.exists(follow_symlinks=True):
            print(
                set_color(
                    status="error",
                    information=f"Config file {config_path} not exist."
                )
            )
            exit(1)

        # Load yaml file to get the config.
        with self.config_path.open(
            mode="r",
            encoding="utf-8"
        ) as config_file:
            self.config_data: dict[str, Any] = safe_load(stream=config_file)

        self.user_id = (
            str(user["id"])
            if (
                (user is not None)  and
                ("id" in user)      and
                (user["id"] != "")
            )
            else (None)
        )

        # Set model device.
        self.device = (
            "cuda"
            if   (cuda.is_available())
            else ("cpu")
        )

        # Initialize QA instance.
        self.qa = None

        # If llm_name is not specified, read it from the config file.
        if llm_name == "":
            llm_name = self.config_data["llm_name"]

        self.llm = self.get_llm(
            llm_name=llm_name,
            seed=self.config_data["seed"],
            is_quantized=self.config_data["is_quantized"],
            device=self.device,
        )

        # Check OpenAI API key.
        self.openai_api_status = self.check_openai_key()

        # Set LLM for query analysis.
        if query_llm_name == "":
            query_llm_name = self.config_data["query_analyser"]["llm_name"]

        self.query_analyser = QueryAnalyser(
            llm_name=query_llm_name,
            seed=self.config_data["seed"],
            is_quantized=self.config_data["query_analyser"]["is_quantized"],
            service_index=self.config_data["service"],
            device=self.device,
        )

        # Code generation service.
        self.code_generator = CodeGenerator()

        # Set up QA instance.
        self.set_up_qa(
            user_id=str(self.user_id),
            user_name=None
        )

        return None


    def __call__(
        self,
        question:   str,
        context:    str         = "",
        log_file:   str | None  = None
    ) -> tuple:
        """
        Execute QA.

        Args:
            question    (str):              a question or a .csv containing multiple questions.
            context     (str, optional):    context for this question. Defaults to "".
            log_file    (str, optional):    whether to print the result in a .txt file. Defaults to None.

        Returns:
            response        (str):      (truncated) response.
            raw_response    (str):      raw response from LLM without truncations.
            retrieve_score  (float):    context retrieve score if is_rag=True.
        """
        # Set initial output to return.
        response        = None
        raw_response    = None
        retrieve_score  = -1

        # Check if OpenAI API key is valid.
        if self.openai_api_status != "":
            return (
                self.openai_api_status,
                self.openai_api_status,
                -1
            )

        # Chat-mode LLM do not need example in the system prompt.
        if self.llm.llm_name in [
            "mistral-7b-instruct-v0.1",
            "gemma-7b-it"
        ]:
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
                (question.find("attention") > -1) or
                (question.find("memory") > -1)    or
                (question.find("perception") > -1)
            ):
                # Manually switch on and off RAG for specific questions.
                if (
                    (question.find("attention") > -1) or
                    (question.find("memory_update") > -1)
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
                    llm=self.llm,
                    rag=self.rag,
                    is_rag=is_rag,
                    log_file=log_file
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
                    log_file=log_file,
                    is_truncate_response=True
                )

                # Batch process the question file.
                cot_generator.batch_process(question)

        else:
            # General question needs truncation according the system prompt to avoid hallucination.
            self.llm.set_truncate_response(True)

            # Truncation needs keywords from the example of system prompt.
            self.llm.system_prompter.set_use_example(True)

            # Process each question.
            (
                response,
                raw_response,
                retrieve_score
            ) = self.qa(
                query=question,
                services=self.get_services(),
                # The context that we are passing here is chat history. See api.py
                context=context,
                is_rag=True,
                verbose=False
            ) # pyright: ignore

        # Return answers with and without truncation, and retrieve score if applicable otherwise -1.
        return (
            response,
            raw_response,
            retrieve_score
        )


    def set_up_qa(
        self,
        user_id: str,
        user_name: str | None = None
    ):
        """
        Set up QA instance by user ID.

        Args:
            user_id     (str):              user ID through front-end.
            user_name   (str, optional):    user name. Defaults to None.
        """
        # Invalid user ID.
        if (
            (user_id == "") and
            (not isinstance(user_id, str))
        ):
            self.user_id = None

        # For a default user (no user ID), always use the same QA instance.
        if (
            (user_id is None) and
            (self.qa is not None)
        ):
            return -1

        if (
            (user_id != self.user_id) or
            (self.qa is None)
        ):
            # Change the global user ID.
            self.user_id = user_id

            # Create vector database service which will be included in RAG for retrieve and information updates.
            vector_db_path: Path = Path(self.config_data["rag"]["vector_db_path"]).resolve(strict=True)

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

            vector_database = VectorDatabase(
                document_analyser_model="gte-small",
                local_database_path=str(object=vector_db_path),
                vector_database_update_threshold=0.98,
                device=self.device
            )

            # # Add a directory of documents.
            # if Path.exists(
            #     Path(self.config_data["doc_directory"]).resolve(strict=True),
            #     follow_symlinks=True
            # ):
            #     vector_database.add_document_directory(self.config_data["doc_directory"])

            # # Add documents.
            # if self.config_data["document_path"] != "" or len(self.config_data["document_path"]) > 0:
            #     vector_database.add_documents(self.config_data["document_path"])

            # Base RAG service with vector_database, the database can be changed using
            # self.rag.set_vector_database().
            self.rag = RAGBase(
                vector_database=vector_database,
                retrieve_score_threshold=self.config_data["rag"]["retrieve_score_threshold"],
                topk=self.config_data["rag"]["topk"],
            )

            # QA module to handle basic types of questions, such __next__move__, __update__store__, and
            # general questions.
            self.qa = QABase(
                query_analyser=self.query_analyser, # pyright: ignore
                llm=self.llm,
                rag=self.rag,
                code_generator=self.code_generator,
                config=str(object=self.config_path)
            )


    def check_openai_key(self):
        """
        Check OpenAI API key valid.

        Returns:
            answer: status information.
        """
        llm_name                = self.config_data["llm_name"]
        query_analyser_llm_name = self.config_data["query_analyser"]["llm_name"]

        is_llm_name_gpt                 = llm_name.find("gpt") > -1
        is_query_analyser_llm_name_gpt  = query_analyser_llm_name.find("gpt") > -1

        llm_name_list = []

        if is_llm_name_gpt:
            llm_name_list.append(llm_name)

        if (
            (is_query_analyser_llm_name_gpt) and
            (query_analyser_llm_name not in llm_name_list)
        ):
            llm_name_list.append(query_analyser_llm_name)

        count = len(llm_name_list)

        if is_llm_name_gpt:
            openai_api_key = self.llm.get_openai_key()

        elif is_query_analyser_llm_name_gpt:
            openai_api_key = self.query_analyser.llm.get_openai_key()

        else:
            openai_api_key = ""

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
        self,
        llm_name: str,
        seed: int = 0,
        is_quantized: bool = False,
        **kwargs
    ):
        """
        Construct LLM give an LLM name.

        Args:
            llm_name        (str):  LLM name, check LLM_MODEL_DICT in src/maps.py.
            seed            (int):  LLM content generation seed. Default to 0.
            is_quantized    (bool): use quantized LLM. Default to False.

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
                set_color(
                    status="error",
                    information=f"Unsupported LLM: {llm_name}."
                )
            )
            exit(1)

        llm = get_instance(
            instances=llm_instances,
            instance_name=llm_instance_name
        )(
            llm_name=llm_name,
            seed=seed,
            is_quantized=is_quantized,
            **kwargs
        )

        print(f"LLM instance created: {llm}")

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
        url:        str                     = "http://backend:8000/api/v1/services/",
        params:     dict[str, bool] | None  = {"active": True},
        lifetime:   float                   = 10.0,
        verbose:    bool                    = False
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
            >>> services = obj.get_services_name(verbose=True)
            >>> services[0] # 1st service
            {"name": "chess", "desc": "<a very long description>"}
        """
        services: dict[str, dict[str, str]] = {}

        try:
            with Client(
                base_url=url,
                params=params,
                timeout=lifetime
            ) as client:
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
                            foot_sep=f"{'=' * 80}"
                        )
                    )
                    print(
                        "{debug_msg:s}\n{foot_sep:s}".format(
                            debug_msg="No active services available...",
                            foot_sep=f"{'=' * 80}"
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
                    services.setdefault(
                        str(data["id"] - 1),
                        {}
                    ).update(
                        {
                            "name": data["name"],
                            "desc": data["desc"]
                        }
                    )

                if verbose:
                    print(
                        "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                            head_sep=f"{'=' * 80}",
                            body_msg="[DEBUG]   SERVICES DATA   [DEBUG]",
                            foot_sep=f"{'=' * 80}"
                        )
                    )
                    pp(
                        object=dict(sorted(services.items())),
                        stream=stdout,
                        indent=4 # Prefer tab over spaces indentation
                    )
                    print(f"{'=' * 80}")
                    return dict(sorted(services.items()))

                else:
                    return dict(sorted(services.items()))


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
