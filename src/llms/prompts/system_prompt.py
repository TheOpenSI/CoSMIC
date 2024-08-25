# -------------------------------------------------------------------------------------------------------------
# File: system_prompt.py
# Project: OpenSI AI System
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
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

class SystemPromptBase:
    def __init__(
        self,
        use_example: bool=False
    ):
        """System prompt base.

        Args:
            use_example (bool, optional): use example in system prompt to detect keywords for
            response truncation. Defaults to False.
        """
        self.use_example = use_example

    def set_use_example(
        self,
        use_example: bool
    ):
        """Set use_example externally.

        Args:
            use_example (bool): use example in system prompt.
        """
        self.use_example = use_example

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Merge user_prompt in system prompt as the question containing context.

        Args:
            user_prompt (str): user prompt.
            context (str, optional): context retrieved if applicable. Defaults to "".
        """
        # Need to be implemented, otherwise raise error.
        raise NotImplementedError

# =============================================================================================================

class Mistral7bv01(SystemPromptBase):
    def __init__(self, **kwargs):
        """For Mistral 7B.
        """
        super().__init__(**kwargs)

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = "<s>"

        if self.use_example:
            if context == "":
                system_prompt += " [INST] What is the capital of China? [/INST]\n"
            else:
                system_prompt += \
                    " [INST] Given that 'Beijing is the capital of China'," \
                    " what is the capital of China? [/INST]\n"

            system_prompt += \
                "Beijing</s>\n" \
                f"[INST] {user_prompt} [/INST]"
        else:
            if context != "":
                system_prompt += \
                    " Always answer the question briefly even if the context isn't useful."

            system_prompt += f" {user_prompt}"

        return system_prompt

# =============================================================================================================

class Mistral7bInstructv01(SystemPromptBase):
    def __init__(self, **kwargs):
        """For Mistral 7B Instruction.
        """
        super().__init__(**kwargs)

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = []

        if self.use_example:
            if context == "":
                system_prompt.append({"role": "user", "content": "What is the capital of China?"})
            else:
                system_prompt.append({
                    "role": "user",
                    "content": "Given that 'Beijing is the capital of China', what is the capital of China?"
                })

            system_prompt.append({"role": "assistant", "content": "Beijing"})

        system_prompt.append({"role": "user", "content": user_prompt})

        return system_prompt

# =============================================================================================================

class Gemma7b(SystemPromptBase):
    def __init__(self, **kwargs):
        """For Gemma 7B.
        """
        super().__init__(**kwargs)

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = "<bos>"
        if self.use_example:
            if context == "":
                system_prompt += \
                    "<start_of_turn>user\n" \
                    "What is the capital of China?<end_of_turn>\n"
            else:
                system_prompt += \
                    "<start_of_turn>user\n" \
                    "Given that 'Beijing is the capital of China'," \
                    " what is the capital of China?<end_of_turn>\n"

            system_prompt += \
                "<start_of_turn>model\n" \
                "Beijing<end_of_turn><eos>\n" \
                "<start_of_turn>user\n" \
                f"{user_prompt}<end_of_turn>\n" \
                "<start_of_turn>model"
        else:
            if context != "":
                system_prompt += \
                    " Always answer the question briefly even if the context isn't useful."

            system_prompt += f" {user_prompt}"

        return system_prompt

# =============================================================================================================

class Gemma7bIt(Mistral7bInstructv01):
    def __init__(self, **kwargs):
        """For Gemma 7B Instruction.
        """
        super().__init__(**kwargs)

# =============================================================================================================

class GPT35Turbo(SystemPromptBase):
    def __init__(self, **kwargs):
        """For GPT 3.5-Turbo API.
        """
        super().__init__(**kwargs)

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Always answer the question " \
                    f"even if the context is not helpful"
            },
            {"role": "user", "content": user_prompt}
        ]

        return system_prompt

# =============================================================================================================

class GPT4o(GPT35Turbo):
    def __init__(self, **kwargs):
        """For GPT 4-o API.
        """
        super().__init__(**kwargs)

# =============================================================================================================

class MistralFinetuned(SystemPromptBase):
    def __init__(self, **kwargs):
        """For Mistral 7B Finetuned LLM.
        """
        super().__init__(**kwargs)

    def __call__(
        self,
        question: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = \
            f"<s>### Instruction:\n{question}\n### Context: \n{context}\n### Response:"

        return system_prompt

# =============================================================================================================

class FENNextMoveAnalyse(SystemPromptBase):
    def __init__(self):
        """For analysis of next move prediction given a FEN.
        """
        super().__init__()

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        if context == "":
            system_prompt = user_prompt
        else:
            system_prompt = f"{user_prompt}\nIf the context is useless, ignore it."

        return system_prompt

# =============================================================================================================

class FENNextMoveAnalyseMistralFinetuned(SystemPromptBase):
    def __init__(self):
        """For analysis of next move prediction given a FEN and a finetuned LLM.
        """
        super().__init__()

    def __call__(
        self,
        user_prompt: str,
        context: str=""
    ):
        """Apply system prompt with user prompt and context.

        Args:
            user_prompt (str): question with context.
            context (str, optional): context retrieved. Defaults to "".

        Returns:
            system_prompt (str): system prompt with question and context under LLM query format.
        """
        system_prompt = \
            f"<s>### Instruction:\n{user_prompt}\n### Context: \n \n### Response:"

        return system_prompt