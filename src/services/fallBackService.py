from typing import Any

from .base import ServiceBase

class FallBackService(ServiceBase):
    def __init__(self,
                 services: list[dict[str, Any]], 
                 **kwargs) -> None:
        """
        Service -1: fallback service for when no other service can answer the query.

        Args:
            services (list[dict[str, Any]]): List of available services.
            **kwargs: additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.services = services

    
    def _get_service_names(self) -> str:
        """
        Get the names of the available services.

        Returns:
            service_names (str): names of the available services.
        """
        return ', '.join([service.get("name") for service in self.services])


    def __call__(self) -> tuple[str, str]:
        """
        Answer a query when no other service can answer it.

        Returns:
            response (str): truncated answer.
            raw_response (str): original answer from LLM.
        """
        raw_response = response = (f"I am unable to assist with that request at this time. "
                                   f"It may be outside my current capabilities, or the required service is not enabled. "
                                   f"For your reference, I currently have access to the following {len(self.services)} services: "
                                   f"{self._get_service_names()}. "
                                   f"Please feel free to rephrase your request, or contact the OpenSI team for further support.")

        return (response, raw_response)