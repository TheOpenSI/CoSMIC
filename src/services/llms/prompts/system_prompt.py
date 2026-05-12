# # -------------------------------------------------------------------------------------------------------------
# # File: system_prompt.py
# # Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# # Contributors:
# #     Danny Xu <danny.xu@canberra.edu.au>
# #     Muntasir Adnan <adnan.adnan@canberra.edu.au>
# #
# # Copyright (c) 2024 Open Source Institute
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# # documentation files (the "Software"), to deal in the Software without restriction, including without
# # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# # the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# # conditions:
# #
# # The above copyright notice and this permission notice shall be included in all copies or substantial
# # portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# # LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# # WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# # -------------------------------------------------------------------------------------------------------------

# # =============================================================================================================


# class SystemPromptBase:
#     def __init__(self, use_example: bool = False, prefix: str = ""):
#         """System prompt base.

#         Args:
#             use_example (bool, optional): use example in system prompt to detect keywords for
#                 response truncation. Defaults to False.
#             prefix (str): prefix to start the prompt. Default to "".
#         """
#         self.use_example = use_example
#         self.prefix = prefix

#     def set_prefix(self, prefix: str):
#         """Set prefix externally.

#         Args:
#             prefix (str): prefix for system prompt.
#         """
#         self.prefix = prefix

#     def get_context(self, context: str = ""):
#         """Get context based on the input type.

#         Args:
#             context (str|dict, optional): context, string or dictionary.
#                 Defaults to "".

#         Returns:
#             context: extract context or an empty string.
#         """
#         if isinstance(context, dict):
#             if "context" in context:
#                 context = context["context"]
#             else:
#                 context = ""

#         return context

#     def set_use_example(self, use_example: bool):
#         """Set use_example externally.

#         Args:
#             use_example (bool): use example in system prompt.
#         """
#         self.use_example = use_example

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Merge user_prompt in system prompt as the question containing context.

#         Args:
#             user_prompt (str): user prompt.
#             context (str|dict, optional): context retrieved if applicable. Defaults to "".
#         """
#         # Need to be implemented, otherwise raise error.
#         raise NotImplementedError


# # =============================================================================================================


# class Mistral7bv01(SystemPromptBase):
#     def __init__(self, prefix="<s>", **kwargs):
#         """For Mistral 7B.

#         Args:
#             prefix (str): prompt prefix. Default to "<s>".
#         """
#         super().__init__(prefix=prefix, **kwargs)

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         # Get context.
#         context = self.get_context(context)

#         system_prompt = self.prefix

#         if self.use_example:
#             if context == "":
#                 system_prompt += " [INST] What is the capital of Australia? [/INST]\n"
#             else:
#                 system_prompt += (
#                     " [INST] Given that 'Canberra is the capital of Australia',"
#                     " what is the capital of Australia? [/INST]\n"
#                 )

#             system_prompt += "Canberra</s>\n" f"[INST] {user_prompt} [/INST]"
#         else:
#             if context != "":
#                 system_prompt += " Always answer the question briefly even if the context isn't useful."

#             system_prompt += f" {user_prompt}"

#         return system_prompt


# # =============================================================================================================


# class Mistral7bInstructv01(SystemPromptBase):
#     def __init__(self, **kwargs):
#         """For Mistral 7B Instruction."""
#         super().__init__(**kwargs)

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         # Get context.
#         context = self.get_context(context)

#         system_prompt = []

#         if self.use_example:
#             if context == "":
#                 system_prompt.append(
#                     {"role": "user", "content": "What is the capital of Australia?"}
#                 )
#             else:
#                 system_prompt.append(
#                     {
#                         "role": "user",
#                         "content": "Given that 'Canberra is the capital of Australia', what is the capital of Australia?",
#                     }
#                 )

#             system_prompt.append({"role": "assistant", "content": "Canberra"})

#         system_prompt.append({"role": "user", "content": user_prompt})

#         return system_prompt


# # =============================================================================================================


# class Gemma7b(SystemPromptBase):
#     def __init__(self, prefix="<bos>", **kwargs):
#         """For Gemma 7B.

#         Args:
#             prefix (str): prompt prefix. Default to "<bos>".
#         """
#         super().__init__(prefix=prefix, **kwargs)

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         # Get context.
#         context = self.get_context(context)

#         system_prompt = self.prefix

#         if self.use_example:
#             if context == "":
#                 system_prompt += (
#                     "<start_of_turn>user\n"
#                     "What is the capital of Australia?<end_of_turn>\n"
#                 )
#             else:
#                 system_prompt += (
#                     "<start_of_turn>user\n"
#                     "Given that 'Canberra is the capital of Australia',"
#                     " what is the capital of Australia?<end_of_turn>\n"
#                 )

#             system_prompt += (
#                 "<start_of_turn>model\n"
#                 "Canberra<end_of_turn><eos>\n"
#                 "<start_of_turn>user\n"
#                 f"{user_prompt}<end_of_turn>\n"
#                 "<start_of_turn>model"
#             )
#         else:
#             if context != "":
#                 system_prompt += " Always answer the question briefly even if the context isn't useful."

#             system_prompt += f" {user_prompt}"

#         return system_prompt


# # =============================================================================================================


# class Gemma7bIt(Mistral7bInstructv01):
#     def __init__(self, **kwargs):
#         """For Gemma 7B Instruction."""
#         super().__init__(**kwargs)


# # =============================================================================================================


# class GPT(SystemPromptBase):
#     def __init__(self, **kwargs):
#         """For GPT API."""
#         super().__init__(**kwargs)

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         system_prompt = [
#             {
#                 "role": "system",
#                 # "content": "You are a helpful assistant. Always answer the question " \
#                 "content": "You are OpenSI-CoSMIC, a helpful assistant developed at OpenSI."
#                 "Always answer the question even if the context is not helpful. "
#                 "You would have access to conversation history, this is for your context only. "
#                 "Do not mention conversation history unless you are specidically asked to do so.",
#             },
#             {"role": "user", "content": user_prompt},
#         ]

#         return system_prompt


# # =============================================================================================================


# class Ollama(GPT):
#     def __init__(self, **kwargs):
#         """For Ollama model."""
#         super().__init__(**kwargs)


# # =============================================================================================================


# class MistralFinetuned(SystemPromptBase):
#     def __init__(self, prefix="<s>", **kwargs):
#         """For Mistral 7B Finetuned LLM.

#         Args:
#             prefix (str): prompt prefix. Default to "<s>".
#         """
#         super().__init__(prefix=prefix, **kwargs)

#     def __call__(self, question: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         # Get context.
#         context = self.get_context(context)

#         system_prompt = f"{self.prefix}### Instruction:\n{question}\n### Context: \n{context}\n### Response:"

#         return system_prompt


# # =============================================================================================================


# class FENNextMoveAnalyse(SystemPromptBase):
#     def __init__(self, **kwargs):
#         """For analysis of next move prediction given a FEN."""
#         super().__init__(**kwargs)

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         # Get context.
#         context = self.get_context(context)

#         if context == "":
#             system_prompt = user_prompt
#         else:
#             system_prompt = f"{user_prompt}\nIf the context is useless, ignore it."

#         return system_prompt


# # =============================================================================================================


# class FENNextMoveAnalyseMistralFinetuned(SystemPromptBase):
#     def __init__(self, prefix="<s>", **kwargs):
#         """For analysis of next move prediction given a FEN and a finetuned LLM.

#         Args:
#             prefix (str): prompt prefix. Default to "<s>".
#         """
#         super().__init__()

#     def __call__(self, user_prompt: str, context: str = ""):
#         """Apply system prompt with user prompt and context.

#         Args:
#             user_prompt (str): question with context.
#             context (str|dict, optional): context retrieved. Defaults to "".

#         Returns:
#             system_prompt (str): system prompt with question and context under LLM query format.
#         """
#         system_prompt = f"{self.prefix}### Instruction:\n{user_prompt}\n### Context: \n \n### Response:"

#         return system_prompt


# -------------------------------------------------------------------------------------------------------------
# File: system_prompt.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
# -------------------------------------------------------------------------------------------------------------

# =============================================================================================================


class SystemPromptBase:
    def __init__(self, use_example: bool = False, prefix: str = ""):
        self.use_example = use_example
        self.prefix = prefix

    def set_prefix(self, prefix: str):
        self.prefix = prefix

    def get_context(self, context: str = ""):
        if isinstance(context, dict):
            context = context.get("context", "")
        return context

    def set_use_example(self, use_example: bool):
        self.use_example = use_example

    def __call__(self, user_prompt: str, context: str = ""):
        raise NotImplementedError


# =============================================================================================================


class ServiceSelectorPrompt(SystemPromptBase):
    """System prompt for the service-selection / orchestrator task.

    Instructs the LLM to act as a deterministic router and reply
    with only a service tag.  Compatible with Ollama (chat-style
    message list) and the existing CoSMIC LLM interface.
    """

    def __init__(self, service_tags: list, **kwargs):
        super().__init__(**kwargs)
        self.service_tags = service_tags

    def __call__(self, user_prompt: str, context: str = "") -> list:
        valid_tags = ", ".join(self.service_tags)
        system_content = (
            "You are a deterministic query-routing assistant for OpenSI-CoSMIC. "
            "Your only job is to read the user query and return the single service "
            "tag that best matches it. "
            f"You MUST reply with exactly one tag from: {valid_tags}. "
            "Reply with the tag only — no explanation, no punctuation, no extra words."
        )

        messages = [{"role": "system", "content": system_content}]

        if self.use_example:
            messages += [
                {
                    "role": "user",
                    "content": (
                        "Given 2 services: 'service mathematics: General maths problems; "
                        "service anatomy: Human body structure questions',"
                        " which service can best answer the following query?"
                        " The query is 'What is the integral of x squared?'."
                        " Just return the service tag without any explanation."
                    ),
                },
                {"role": "assistant", "content": "mathematics"},
            ]

        messages.append({"role": "user", "content": user_prompt})
        return messages


class Mistral7bv01(SystemPromptBase):
    def __init__(self, prefix="<s>", **kwargs):
        super().__init__(prefix=prefix, **kwargs)

    def __call__(self, user_prompt: str, context: str = ""):
        context = self.get_context(context)
        system_prompt = self.prefix
        if self.use_example:
            if context == "":
                system_prompt += " [INST] What is the capital of Australia? [/INST]\n"
            else:
                system_prompt += (
                    " [INST] Given that 'Canberra is the capital of Australia',"
                    " what is the capital of Australia? [/INST]\n"
                )
            system_prompt += "Canberra</s>\n" f"[INST] {user_prompt} [/INST]"
        else:
            if context != "":
                system_prompt += " Always answer the question briefly even if the context isn't useful."
            system_prompt += f" {user_prompt}"
        return system_prompt


class Mistral7bInstructv01(SystemPromptBase):
    def __call__(self, user_prompt: str, context: str = ""):
        context = self.get_context(context)
        system_prompt = []
        if self.use_example:
            if context == "":
                system_prompt.append(
                    {"role": "user", "content": "What is the capital of Australia?"}
                )
            else:
                system_prompt.append(
                    {
                        "role": "user",
                        "content": "Given that 'Canberra is the capital of Australia', what is the capital of Australia?",
                    }
                )
            system_prompt.append({"role": "assistant", "content": "Canberra"})
        system_prompt.append({"role": "user", "content": user_prompt})
        return system_prompt


class Gemma7b(SystemPromptBase):
    def __init__(self, prefix="<bos>", **kwargs):
        super().__init__(prefix=prefix, **kwargs)

    def __call__(self, user_prompt: str, context: str = ""):
        context = self.get_context(context)
        system_prompt = self.prefix
        if self.use_example:
            if context == "":
                system_prompt += "<start_of_turn>user\nWhat is the capital of Australia?<end_of_turn>\n"
            else:
                system_prompt += (
                    "<start_of_turn>user\n"
                    "Given that 'Canberra is the capital of Australia',"
                    " what is the capital of Australia?<end_of_turn>\n"
                )
            system_prompt += (
                "<start_of_turn>model\nCanberra<end_of_turn><eos>\n"
                f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
                "<start_of_turn>model"
            )
        else:
            if context != "":
                system_prompt += " Always answer the question briefly even if the context isn't useful."
            system_prompt += f" {user_prompt}"
        return system_prompt


class Gemma7bIt(Mistral7bInstructv01):
    pass


class GPT(SystemPromptBase):
    def __call__(self, user_prompt: str, context: str = ""):
        return [
            {
                "role": "system",
                "content": (
                    "You are OpenSI-CoSMIC, a helpful assistant developed at OpenSI. "
                    "Always answer the question even if the context is not helpful. "
                    "You would have access to conversation history, this is for your context only. "
                    "Do not mention conversation history unless you are specifically asked to do so."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]


class Ollama(GPT):
    pass


class MistralFinetuned(SystemPromptBase):
    def __init__(self, prefix="<s>", **kwargs):
        super().__init__(prefix=prefix, **kwargs)

    def __call__(self, question: str, context: str = ""):
        context = self.get_context(context)
        return f"{self.prefix}### Instruction:\n{question}\n### Context: \n{context}\n### Response:"


class FENNextMoveAnalyse(SystemPromptBase):
    def __call__(self, user_prompt: str, context: str = ""):
        context = self.get_context(context)
        if context == "":
            return user_prompt
        return f"{user_prompt}\nIf the context is useless, ignore it."


class FENNextMoveAnalyseMistralFinetuned(SystemPromptBase):
    def __init__(self, prefix="<s>", **kwargs):
        super().__init__()

    def __call__(self, user_prompt: str, context: str = ""):
        return f"{self.prefix}### Instruction:\n{user_prompt}\n### Context: \n \n### Response:"
