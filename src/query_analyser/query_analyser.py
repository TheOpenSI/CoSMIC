### Core modules ###
from sys import exit
from pathlib import Path
from re import search
from fastapi import HTTPException, status

### Type hints ###


### Internal modules ###
from ..services.llms import llm as llm_instances
from ..query_analyser import user_prompt as query_user_prompt_instances
from ..maps import LLM_INSTANCE_DICT
from ...utils.log_tool import set_color
from ...utils.module import get_instance


class QueryAnalyser:
    def __init__(
        self,
        llm_name: str = "qwen2.5:7b",
        seed: int = 0,
        is_quantised: bool = False,
        service_index: int = -1,
        device: str = "cuda",
    ) -> None:
        """
        Query analyser to select a service.

        Args:
            llm_name (str, optional):
                LLM name for analyser. Defaults to "qwen2.5:7b".

            seed (int, optional):
                response generation seed. Defaults to 0.

            is_quantised (bool, optional):
                use quantized LLM. Defaults to False.

            service_index (int, optional):
                use selected service, otherwise automatically select.

            device (str, optional):
                use cuda or cpu for LLM. Defaults to "cuda".
        """
        # Set config.
        self.root = Path(__file__).resolve(strict=True).parent.parent.parent
        self.device = device

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
                set_color(status="error", information=f"Unsupported LLM: {llm_name}.")
            )
            exit(1)

        # Build LLM instance from class defined in .py
        self.llm = get_instance(
            instances=llm_instances, instance_name=llm_instance_name
        )(
            llm_name=llm_name,
            seed=seed,
            is_quantised=is_quantised,
            use_example=False,
            is_truncate_response=True,
            device=device,
        )

        return None

    def __call__(
        self,
        query: str,
        # NOTE:
        # for legacy purposes. Change to `dict[int, dict[str, str]]` type when
        # update to handle `int` properly
        services: dict[str, dict[str, str]],
        verbose: bool = False,
    ) -> tuple[str, dict[str, str | bool]]:
        """
        Analyse query to get service option.

        Args:
            query (str):
                question.

            services (dict[str, dict[str, str]]):
                all available services for Query Analyser, which then being used
                to re-structure question with its prompting techniques.

            verbose (bool, optional):
                debug mode. Default to False.

        Returns:
            service_option (str):
                service option.

            service_info_dict (dict[str, str | bool]):
                updated information dictionary.
        """
        # Manile - add default service 0 and then add other services in
        self.services: dict[str, str] = {
            service_id: service_info["desc"]
            for (service_id, service_info) in services.items()
        }

        self.services["0"] = (
            "Answer questions about the AI assistant itself such as who created it, what OpenSI-CoSMIC is, and what it can do."
        )

        # print(self.services)

        # There is/are active services from fetched API endpoint
        if len(self.services) != 0:
            pass

        # No active services found from fetched API endpoint
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "404 - Not Found",
                    "message": "{trig:s}: {cond:s}".format(
                        trig="EmptyServiceError",
                        cond="No active services found from fetched API endpoint. At least 1 service is required to for Query Analyser.",
                    ),
                },
            )

        # Chect if chess is one of the services, if yes, then add chess subservices to full services.
        self.full_services: dict[str, str] = dict(self.services)

        if "1" in self.services:
            self.chess_subservices: dict[str, str] = {
                "1.0": "predict next move given a chess FEN",
                "1.1": "predict next move given a sequence of moves",
            }

            self.full_services.update(self.chess_subservices)

        # print(self.full_services)

        # print(
        #     set_color(
        #         status="info",
        #         information=f"Full services from Query Analyser: {self.full_services}"
        #     )
        # )

        # Get the number of services.
        self.num_services: int = len(self.services)

        # Set user prompter for service option.
        self.user_prompter_service = get_instance(
            instances=query_user_prompt_instances, instance_name="QueryAnalyserService"
        )(services=self.services)

        # Create an initial information dictionary.
        service_info_dict: dict[str, str | bool] = {
            "query": query,
        }

        if self.service_index >= 0:
            service_option = str(self.service_index)

        else:
            # Set the user prompter for service option.
            self.llm.set_user_prompter(self.user_prompter_service)

            # Get raw anlysis from LLM to select a service.
            service_analysis = self.llm(query)[0]

            # Get the service option.
            service_option = self.mapping(response=service_analysis)

            # print(service_option)

            # print(
            #     set_color(
            #         status="info",
            #         information=f"Selected service from SLM response (user query has been re-prompted by Query Analyser): {service_option}"
            #     )
            # )

            # Analysis information.
            if verbose:
                print(
                    set_color(
                        status="info",
                        information="{0:s}\n{1:s}\n{2:s}".format(
                            f"Query: {query}",
                            f"Analysis: {service_analysis}",
                            f"Service: {service_option}",
                        ),
                    )
                )

        if service_option == "1":
            # Remove last symbol.
            if query[-1] in [",", ".", "!", "?"]:
                query = query[:-1]

            # Predict the next move in chess game.
            service_option, service_info_dict = self.chess_parse(
                query=query, service_info_dict=service_info_dict
            )

        elif service_option == "2":
            # Update the vector database.
            service_option, service_info_dict = self.update_vector_database_parse(
                query=query, service_info_dict=service_info_dict
            )

        return (service_option, service_info_dict)

    def quit(self):
        """
        Quit by releasing model memory and instance.
        """
        # Release memory of LLM.
        if self.llm:
            self.llm.quit()

    def mapping(self, response: str):
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
            pattern="service (\d{1,3}\.\d{1,3}|\d{1,3})", string=response, flags=0
        )

        if option:
            option = option.group(1)

            if option not in self.full_services.keys():
                print(
                    set_color(
                        status="error",
                        information=f"Unknown service '{option}' from '{response}'.",
                    )
                )
                return "-1" 

            else:
                return option

        else:
            return "4" 

    def get_service(self, index: int):
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

        return self.services[str(object=index)]

    def chess_parse(self, query: str, service_info_dict: dict):
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
            pattern="[\[,\:](.*?[,\s].*?)[\.,\]]?$", string=query, flags=0
        )

        # Parse FEN string.
        fen_match = search(
            pattern="{0:s}{1:s}".format(
                "(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)",
                "\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$",
            ),
            string=query,
            flags=0,
        )

        if fen_match:
            # Given FEN.
            current_fen = fen_match.group()
            service_option = "1.0"
            service_info_dict.update({"fen": current_fen})

        elif move_match:
            # Given a sequence of moves.
            current_moves = move_match.group(1)
            service_option = "1.1"
            service_info_dict.update({"moves": current_moves})

        else:
            # Invalid inputs.
            if query.find("predict") > -1 or query.find("next move") > -1:
                print(
                    set_color(
                        status="hint",
                        information='For chess move prediction, index a sequence of moves or FEN with ":".',
                    )
                )

        return (service_option, service_info_dict)

    def update_vector_database_parse(self, query: str, service_info_dict: dict):
        """
        Parse query to get text or document path to update vector database.

        Args:
            query               (str):  question.
            service_info_dict   (dict): dictionary to contain parsed information.

        Returns:
            service_option      (str):  service option.
            service_info_dict   (dict): updated information dictionary.
        """
        service_option = "2"

        # Check if context is a .pdf.
        is_a_document = query.find(".pdf") > -1

        # Update information dictionary.
        service_info_dict.update(
            {"is_a_document": is_a_document, "text": None, "document_path": None}
        )

        if is_a_document:
            # Parse move string
            document_path = search(
                pattern="(?<=\:\s)(.*?)+\.pdf", string=query, flags=0
            )

            if document_path:
                document_path = document_path.group()

            else:
                print(
                    set_color(
                        status="warning",
                        information="Invalid document query [tip: pdf file(s) is required.]",
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
            text = search(pattern="\:((\"|')?(.*?)[\",']?$)", string=query, flags=0)

            if text:
                text = text.group(0)
                text = text.replace(": ", "").replace(":", "")

                # Update information dictionary.
                service_info_dict["text"] = text
            else:
                print(
                    set_color(
                        status="warning",
                        information="Invalid text query [tip: index the text with :]",
                    )
                )

        return (service_option, service_info_dict)
