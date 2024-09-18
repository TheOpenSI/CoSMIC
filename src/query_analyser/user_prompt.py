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

class QueryAnalyser(UserPromptBase):
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
            service_string += f"option {service_tag}. {service}"
            option_string += f"option {service_tag}"

            if idx < self.num_services - 1:
                service_string += ", "
                option_string += " or "

        # Set as global variables.
        self.service_string = service_string
        self.option_string = option_string

    def __call__(
        self,
        question,
        context: dict={}
    ):
        """Build user prompt to analyse the question.

        Args:
            query (str): the question.
            context (dict): context, not used but reserve for interface uniform. Default to "".

        Returns:
            user_prompt (str): question with instruction.
        """
        user_prompt = f"Given {self.num_services} options: " \
            f"{self.service_string}. " \
            f"Which option does '{question}' belong to? " \
            f"Just return {self.option_string}."

        return user_prompt