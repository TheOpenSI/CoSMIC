### Core modules ###
from sys import (
    exit,
    stdout
)
from pathlib import Path
from re import search
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
from ...types.query_analyser import ServicesJsonResponse


### Internal modules ###
from ..services.llms import llm as llm_instances
from ..query_analyser import user_prompt as query_user_prompt_instances
from ..maps import LLM_INSTANCE_DICT
from ...utils.log_tool import set_color
from ...utils.module import get_instance


class QueryAnalyser:
    def __init__(
        self,
        llm_name: str = "mistral-7b-instruct-v0.1",
        seed: int = 0,
        is_quantized: bool = False,
        service_index: int = -1,
        device: str = "cuda"
    ):
        """
        Query analyser to select a service.

        Args:
            llm_name        (str, optional):    LLM name for analyser. Defaults to "mistral-7b-instruct-v0.1".
            seed            (int, optional):    response generation seed. Defaults to 0.
            is_quantized    (bool, optional):   use quantized LLM. Defaults to False.
            service_index   (int, optional):    use selected service, otherwise automatically select.
            device          (str, optional):    use cuda or cpu for LLM. Defaults to "cuda".
        """
        # Set config.
        self.root = Path(__file__).resolve(strict=True).parent.parent.parent
        self.device = device

        # Set a list of services.

        # NOTE:
        # Due to the way current logic checking for selected service after
        # receiving the response from LLM based on Query Analyser's refined
        # question from user query, we've to do this kind of workaround until
        # it get updates to handle 'int' type correctly (which it should be).
        services_desc: dict[int, str] = self._get_service_desc(verbose=True)

        # No active services available in the system
        if len(services_desc) == 0:
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
            self.legacy_services_desc: dict[str, str] = {
                str(str_id): desc \
                for (str_id, desc) in services_desc.items()
            }

            # Set chess subservices.
            self.chess_subservices_desc: dict[str, str] = {
                "0.0": "predict next move given a chess FEN",
                "0.1": "predict next move given a sequence of moves"
            }

            # Get full services.
            self.full_services = {**self.legacy_services_desc, **self.chess_subservices_desc}

            # Get the number of services.
            self.num_services = len(self.legacy_services_desc)

            # Set provided service.
            self.service_index = service_index

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

            # Build LLM instance from class defined in .py
            self.llm = get_instance(
                instances=llm_instances,
                instance_name=llm_instance_name
            )(
                llm_name=llm_name,
                seed=seed,
                is_quantized=is_quantized,
                use_example=False,
                is_truncate_response=True,
                device=device
            )

            # Set user prompter for service option.
            self.user_prompter_service = get_instance(
                instances=query_user_prompt_instances,
                instance_name="QueryAnalyserService"
            )(
                services=self.legacy_services_desc
            )

            # Set user prompter for system information.
            self.user_prompter_system_info = get_instance(
                instances=query_user_prompt_instances,
                instance_name="QueryAnalyserSystemInfo"
            )(
                services=self.legacy_services_desc
            )


    def quit(self):
        """
        Quit by releasing model memory and instance.
        """
        # Release memory of LLM.
        if self.llm: self.llm.quit()


    def mapping(
        self,
        response: str
    ):
        """
        Parse response to get service option.

        Args:
            response (str): response from LLM analysis.

        Returns:
            option (int): index of the service option if avaiable; otherwise, None.
        """
        # Set to lower case.
        response = response.lower()

        # Truncate to get the option index.
        option = search(
            pattern=r"service (\d{1,3}\.\d{1,3}|\d{1,3})",
            string=response,
            flags=0
        )

        if option:
            option = option.group(1)

            if option not in self.full_services.keys():
                print(
                    set_color(
                        status="error",
                        information=f"Unknown service '{option}' from '{response}'."
                    )
                )

                return "-1"
        else:
            return "-1"

        return option


    def get_service(
        self,
        index: int
    ):
        """
        Get the description of the service.

        Args:
            index (int): index of the service.

        Returns:
            service (str): description of the service.
        """
        # Option not found.
        if index not in self.full_services.keys():
            return None

        return self.legacy_services_desc[str(object=index)]


    def chess_parse(
        self,
        query: str,
        service_info_dict: dict
    ):
        """
        Parse query to get chess service.

        Args:
            query               (str):  question.
            service_info_dict   (dict): dictionary to contain parsed information.

        Returns:
            service_option      (str):  service option.
            service_info_dict   (dict): updated information dictionary.
        """
        # Default service option.
        service_option = "-1"

        # Parse move string.
        move_match = search(
            pattern=r"[\[,\:](.*?[,\s].*?)[\.,\]]?$",
            string=query,
            flags=0
        )

        # Parse FEN string.
        fen_match = search(
            pattern="{0:s}{1:s}".format(
                r"(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)",
                r"\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$"
            ),
            string=query,
            flags=0
        )

        if fen_match:
            # Given FEN.
            current_fen = fen_match.group()
            service_option = "0.0"
            service_info_dict.update({"fen": current_fen})

        elif move_match:
            # Given a sequence of moves.
            current_moves = move_match.group(1)
            service_option = "0.1"
            service_info_dict.update({"moves": current_moves})

        else:
            # Invalid inputs.
            if query.find("predict") > -1 or query.find("next move") > -1:
                print(
                    set_color(
                        status="hint",
                        information="For chess move prediction, index a sequence of moves or FEN with \":\"."
                    )
                )

        return (service_option, service_info_dict)


    def update_vector_database_parse(
        self,
        query: str,
        service_info_dict: dict
    ):
        """
        Parse query to get text or document path to update vector database.

        Args:
            query               (str):  question.
            service_info_dict   (dict): dictionary to contain parsed information.

        Returns:
            service_option      (str):  service option.
            service_info_dict   (dict): updated information dictionary.
        """
        service_option = "1"

        # Check if context is a .pdf.
        is_a_document = query.find(".pdf") > -1

        # Update information dictionary.
        service_info_dict.update({
            "is_a_document": is_a_document,
            "text": None,
            "document_path": None
        })

        if is_a_document:
            # Parse move string
            document_path = search(
                pattern=r"(?<=\:\s)(.*?)+\.pdf",
                string=query,
                flags=0
            )

            if document_path:
                document_path = document_path.group()

            else:
                print(
                    set_color(
                        status="warning",
                        information="Invalid document query [tip: pdf file(s) is required.]"
                    )
                )

                return (service_option, service_info_dict)

            # Check if not an absolute path, convert to an absoluate path.
            if not Path(document_path).resolve(strict=True).is_absolute():
                document_path = self.root.joinpath(document_path)

            # Update information dictionary.
            service_info_dict["document_path"] = document_path
        else:
            # Extract text.
            text = search(
                pattern=r"\:((\"|\')?(.*?)[\",\']?$)",
                string=query,
                flags=0
            )

            if text:
                text = text.group(0)
                text = text.replace(": ", "").replace(":", "")

                # Update information dictionary.
                service_info_dict["text"] = text
            else:
                print(
                    set_color(
                        status="warning",
                        information="Invalid text query [tip: index the text with :]"
                    )
                )

        return (service_option, service_info_dict)


    def get_system_information_relevance(
        self,
        response: str
    ):
        """
        Get whether the question is related to system information from response.

        Args:
            response (str): LLM response.

        Returns:
            relevance (bool): whether being related to.
        """
        relevance = response.lower().find("yes") > -1

        return relevance


    def __call__(
        self,
        query: str,
        verbose: bool = False
    ):
        """
        Analyse query to get service option.

        Args:
            query   (str):              question.
            verbose (bool, optional):   debug mode. Default to False.

        Returns:
            service_option      (str):  service option.
            service_info_dict   (dict): updated information dictionary.
        """
        # Create an initial information dictionary.
        service_info_dict = {
            "query": query,
            "system_information_relevance": False,
            "system_information": ""
        }

        if self.service_index >= 0:
            service_option = str(self.service_index)

        else:
            # Set the user prompter for service option.
            self.llm.set_user_prompter(self.user_prompter_service)

            # Get raw anlysis from LLM to select a service.
            service_analysis = self.llm(query)[0]

            # Get the service option.
            service_option = self.mapping(service_analysis)

            # Analysis information.
            if verbose:
                print(set_color(
                    status="info",
                    information=f"Query: {query}, analysis: {service_analysis}, service: {service_option}."
                ))

        if service_option == "0":
            # Remove last symbol.
            if query[-1] in [",", ".", "!", "?"]: query = query[:-1]

            # Predict the next move in chess game.
            service_option, service_info_dict = self.chess_parse(query, service_info_dict)

        elif service_option == "1":
            # Update the vector database.
            service_option, service_info_dict = self.update_vector_database_parse(
                query,
                service_info_dict
            )

        else:
            # Set the user prompter for system information relevance.
            self.llm.set_user_prompter(self.user_prompter_system_info)

            # Get the response for whether the query is related to system information.
            relevance_analysis: str = self.llm(query)[0]

            # Get whether the question is related to system information.
            relevance: bool = self.get_system_information_relevance(relevance_analysis)

            # Update system information relevance.
            service_info_dict["system_information_relevance"] = relevance

            # Add the system information if it is related to the question.
            if relevance:
                service_info_dict["system_information"] = self.user_prompter_system_info.system_information

        return (service_option, service_info_dict)


    def _get_service_desc(
        self,
        # TODO: util to dynamically check for valid endpoint format
        endpoint:           str                     = "http://backend:8000/api/v1/services/",
        endpoint_params:    dict[str, bool] | None  = {"active": True},
        lifetime:           float                   = 10.0,
        verbose:            bool                    = False
    ) -> dict[int, str]:
        """
        Retrieve service descriptions from the backend API with 0-based indexing.

        This method fetches service data from the specified endpoint and returns
        a dictionary mapping 0-based indices to service descriptions. The
        transformation subtracts 1 from the API's 1-based IDs to create 0-based
        indexing.

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
            Dictionary mapping 0-based indices to service descriptions.
            Example: {0: "<service 1 description>", 1: "<service 2 description>"}

        Raises:
            HTTPException: With status code 500 if any connection error occurs
                (ConnectError, ConnectTimeout) or other unexpected exceptions.

        Example:
            >>> services = obj._get_services_desc(verbose=True)
            >>> services[0]  # First service description
            'Answer question about Academic Governance.'
        """
        services_desc_dict: dict[int, str] = {}

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
                                body_msg="[DEBUG]   SERVICES DATA ('DESC' ONLY)   [DEBUG]",
                                foot_sep=f"{'=' * 80}"
                            )
                        )
                        print(
                            "{debug_msg:s}\n{foot_sep:s}".format(
                                debug_msg="No active services available...",
                                foot_sep=f"{'=' * 80}"
                            )
                        )
                        return services_desc_dict

                    else:
                        return services_desc_dict

                else:
                    # There is/are active services available
                    for service_data in data:
                        # NOTE: for matching the hard-coded style until updating the logic
                        service_id:     int = service_data["id"] - 1
                        services_desc:  str = service_data["desc"]

                        services_desc_dict.update({service_id: services_desc})

                    if verbose:
                        print(
                            "{head_sep:s}\n{body_msg:s}\n{foot_sep:s}".format(
                                head_sep=f"{'=' * 80}",
                                body_msg="[DEBUG]   SERVICES DATA ('DESC' ONLY)   [DEBUG]",
                                foot_sep=f"{'=' * 80}"
                            )
                        )
                        pp(
                            object=services_desc_dict,
                            stream=stdout,
                            indent=4 # Prefer tab over spaces indentation
                        )
                        print(f"{'=' * 80}")
                        return services_desc_dict

                    else:
                        return services_desc_dict

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
