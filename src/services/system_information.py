from ...src.query_analyser.user_prompt import QueryAnalyserSystemInfo
from ...src.services.llms.llm import LLMBase
from ...src.services.base import ServiceBase


class SystemInformationService(ServiceBase):
    def __init__(self, llm: LLMBase, **kwargs) -> None:
        """
        Service 0: answer questions about the OpenSI-CoSMIC system itself.

        Args:
            llm (LLMBase): LLM instance used to answer system-info queries.
        """

        super().__init__(**kwargs)

        self.llm = llm

        return None

    def __call__(
        self, query: str, services: dict, context: str | dict = ""
    ) -> tuple[str, str]:
        """
        Answer a system-information query about OpenSI-CoSMIC.

        Args:
            query (str): user query.
            services (dict): current services dict, used to describe capabilities.
            context (str | dict, optional): existing context. Defaults to "".

        Returns:
            response (str): truncated answer.
            raw_response (str): original answer from LLM.
        """
        system_information = QueryAnalyserSystemInfo(
            services=services
        ).get_system_information()

        print("system_information", system_information)

        response, raw_response = self.llm(question=query, context=system_information)

        return (response, raw_response)
