from ..query_analyser.user_prompt import QueryAnalyserSystemInfo
from .llms.llm import LLMBase
from .base import ServiceBase


class SystemInformationService(ServiceBase):
    def __init__(self, llm: LLMBase, **kwargs) -> None:
        """
        Service 0: answer questions about the OpenSI-CoSMIC system itself.

        Args:
            llm (LLMBase): initiate with LLM instance to answer system-info queries.
        """

        super().__init__(**kwargs)

        self.llm = llm

        return None

    def __call__(
        self, query: str, services: dict, context: str | dict = "") -> tuple[str, str]:
        """
        Answer a system-information query about OpenSI-CoSMIC.

        Args:
            query (str): user query.
            services (dict): current services dict, used to describe capabilities.
            context (str | dict, optional): existing context (e.g. chat history). Defaults to "".

        Returns:
            response (str): truncated answer.
            raw_response (str): original answer from LLM.
        """
        user_prompt = query

        # If the question is related to system information,
        # add system information to context.

        system_information = QueryAnalyserSystemInfo(services).system_information


        # Add the information to existing context.
        # This context is likely to be chat history.

        if isinstance(context, dict):
            context["context"] = "OpenSI System Information:\n" + system_information + "\n\n" + context["context"]
        else:
            context = "OpenSI System Information:\n" + system_information + "\n\n" + context

        # Get the response with retrieved context if applicable.
        response, raw_response = self.llm(user_prompt, context=context)

        return (response, raw_response)



