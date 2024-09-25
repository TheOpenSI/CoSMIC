# -------------------------------------------------------------------------------------------------------------
# File: user_prompt.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
# 
# Copyright (c) 2024 Open Source Institute
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import os, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from src.services.llms.prompts.user_prompt import UserPromptBase

# =============================================================================================================

class QueryAnalyserService(UserPromptBase):
    def __init__(
        self,
        services: dict,
        **kwargs
    ):
        """Initialize the instance.

        Args:
            services (dict): a dictionary of services.
        """
        super().__init__(**kwargs)

        # Set services.
        self.services = services
        self.num_services = len(self.services)

        # Get the service string and option string for user prompt.
        service_string = ""
        option_string = ""

        # Get strings for user prompt.
        for idx, service_tag in enumerate(self.services):
            service = self.services[service_tag]
            service_string += f"service {service_tag}: {service}"
            option_string += f"service {service_tag}"

            if idx < self.num_services - 1:
                service_string += ", "
                option_string += " or "

        # Set as global variables.
        self.service_string = service_string
        self.option_string = option_string

    def __call__(
        self,
        question: str,
        context: dict={}
    ):
        """Build user prompt to analyse the question.

        Args:
            question (str): the question.
            context (dict): context, not used but reserve for interface uniform. Default to "".

        Returns:
            user_prompt (str): question with instruction.
        """
        user_prompt = f"Given {self.num_services} services:" \
            f" '{self.service_string}'," \
            f" which service is the question '{question}' highly related to?" \
            f" Just return {self.option_string}."

        return user_prompt

# =============================================================================================================

class QueryAnalyserSystemInfo(QueryAnalyserService):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """Initialize the instance.
        """
        super().__init__(*args, **kwargs)

        # Get the service string and option string for user prompt.
        service_string = ""

        # Get a string of services for user prompt.
        for idx, service_tag in enumerate(self.services):
            service = self.services[service_tag]
            service_string += f"{service}"

            if idx < self.num_services - 1:
                service_string += ", "

        # Replace self.service_string.
        self.service_string = service_string

        # Get system information.
        self.system_information = self.get_system_information()

    def get_system_information(self):
        """Get system information.

        Returns:
            system_information (str): system information.
        """
        system_information = \
            f"This system is built and maintained by the team of the Open Source" \
            f" Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)." \
            f" It provides {len(self.services)} services," \
            f" including {self.service_string}."

        return system_information

    def __call__(
        self,
        question: str,
        context: dict={}
    ):
        """Build user prompt to analyse the question.

        Args:
            question (str): the question.
            context (dict): context, not used but reserve for interface uniform. Default to "".

        Returns:
            user_prompt (str): question with instruction.
        """
        user_prompt = f"Given the system information" \
            f" '{self.system_information}'," \
            f" is the question '{question}' highly related to the system information?" \
            f" Just return yes or no without any explainations."

        return user_prompt