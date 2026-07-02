### Core modules ###


### Type hints ###


### Internal modules ###
from ..services.llms.prompts.user_prompt import UserPromptBase


class QueryAnalyserService(UserPromptBase):
    def __init__(self, services: dict, **kwargs):
        """
        Initialize the instance.

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

    def __call__(self, question: str, context: dict = {}):
        """Build user prompt to analyse the question.

        Args:
            question    (str):  the question.
            context     (dict): context, not used but reserve for interface uniform. Default to "".

        Returns:
            user_prompt (str): question with instruction.
        """
        user_prompt: str = "{0:s}, {1:s}. {2:s}; {3:s}. {4:s}. {5:s}.".format(
            f"Given {self.num_services} services: '{self.service_string}'",
            f"which service can answer the following query? The query is '{question}'",
            "For instance, if the query is asking about who you are, who created you, or what you can do, then select service 0",
            "otherwise, if the query is to predict the next chess move, then select service 1",
            "otherwise, if the query is to generate or modify a code, then select service 3",
            "otherwise, if the query is about Academic Governance, then select service 5",
            "otherwise, if the query is a general question that doesn't fit the other services, then select service 4",
            "Just return which service without any explainations",
        )

        return user_prompt


class QueryAnalyserSystemInfo(QueryAnalyserService):
    def __init__(self, *args, **kwargs):
        """
        Initialize the instance.
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
        # system_information = (
        #     f"My name is OpenSI-CoSMIC."
        #     f" I am an AI system."
        #     f" OpenSI-CoSMIC stands for the Open Source"
        #     f" Institute-Cognitive System of Machine Intelligent Computing."
        #     f" I am created, developed, and maintained by OpenSI,"
        #     f" which is an institute at the University of Canberra."
        #     f" At the moment, I can provide {len(self.services)} services,"
        #     f" including {self.service_string}."
        #     f" I can design and provide more services under an agreement."
        #     f" To request more services, please find the contact information in my profile."
        #     f" My profile and project repository can be found at"
        #     f" <https://github.com/TheOpenSI/CoSMIC>."
        #     f" You take my role."
        # )
        system_information = (
            f"You are OpenSI-CoSMIC (Open Source Institute – Cognitive System of "
            f"Machine Intelligent Computing), an AI assistant created, developed, "
            f"and maintained by OpenSI, an institute at the University of Canberra. "
            f"You can currently provide {len(self.services)} services, including "
            f"{self.service_string}. "
            f"More services can be added under agreement — to request one, users "
            f"can find contact information in your profile. Your profile and "
            f"project repository are at https://github.com/TheOpenSI/CoSMIC. "
            f"Answer the user's question as yourself, using this information."
        )

        return system_information

    def __call__(self, question: str, context: dict = {}):
        """Build user prompt to analyse the question.

        Args:
            question (str): the question.
            context (dict): context, not used but reserve for interface uniform. Default to "".

        Returns:
            user_prompt (str): question with instruction.
        """
        # user_prompt = f"Given that '{self.system_information}'," \
        #     f" is the question '{question}' a general question related to the system" \
        #     f" information or OpenSI-CoSMIC?" \
        #     f" Just answer yes or no without any explainations."

        user_prompt = (
            f"A user has asked the following question - '{question}', "
            f"is the user asking information about you (the AI assistant called OpenSI-CoSMIC) or "
            f"requesting information about how you work/what you can do? "
            f"Answer only 'YES' or 'NO' without any explanations."
        )

        return user_prompt
