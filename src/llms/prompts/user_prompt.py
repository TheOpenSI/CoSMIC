# -------------------------------------------------------------------------------------------------------------
# File: user_prompt.py
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

class UserPromptBase:
    def __init__(self):
        """Initialize the instance.
        """
        pass

    def get_context(
        self,
        context: str=""
    ):
        """Get context based on the input type.

        Args:
            context (str|dict, optional): context, string or dictionary.
            Defaults to "".

        Returns:
            context: extract context or an empty string.
        """
        if isinstance(context, dict):
            if "context" in context:
                context = context["context"]
            else:
                context = ""

        return context

# =============================================================================================================

class GeneralUserPrompt(UserPromptBase):
    def __init__(self):
        """Initialize the instance.
        """
        super().__init__()

    def __call__(
        self,
        question: str,
        context: str=""
    ):
        """Build user prompt.

        Args:
            question (str): user question.
            context (str|dict, optional): external context. Defaults to "".

        Returns:
            user_prompt (str): user prompt.
        """
        # Get context.
        context = self.get_context(context)

        if context == "":
            user_prompt = question
        else:
            user_prompt = f"Given that '{context}', {question}"

        return user_prompt

# =============================================================================================================

class FENNextMovePredict(UserPromptBase):
    def __init__(self):
        """Initialize the instance.
        """
        super().__init__()

    def __call__(
        self,
        fen: str,
        player: str
    ):
        """Build user prompt.

        Args:
            question (str): user question.

        Returns:
            user_prompt (str): user prompt.
        """
        user_prompt = \
            f"Given chess board FEN '{fen}'," \
            f" give the next move of player {player} indexed by ** without any analysis?"

        return user_prompt

# =============================================================================================================

class FENNextMoveAnalyse(UserPromptBase):
    def __init__(self):
        """Initialize the instance.
        """
        super().__init__()

    def __call__(
        self,
        fen: str,
        player: str,
        move: str,
        context: str=""
    ):
        """Build user prompt to analyse next move action.

        Args:
            fen (str): Chess FEN.
            player (str): White or Black for the next move.
            move (str): next move taken by player given FEN.
            context (str|dict, optional): context retrieved if applicable. Defaults to "".

        Returns:
            user_prompt (str): user prompt.
        """
        # Get context.
        context = self.get_context(context)

        if context == "":
            user_prompt = \
                f"Given chess board FEN '{fen}', explain briefly why {player} takes {move}?"
        else:
            user_prompt = \
                f"Given chess board FEN '{fen}' and context that '{context}', " \
                f"explain briefly why {player} takes {move}?"

        return user_prompt

# =============================================================================================================

class FENNextMoveAnalyseMistralFinetuned(UserPromptBase):
    def __init__(self):
        """Initialize the instance.
        """
        super().__init__()

    def __call__(
        self,
        fen: str,
        player: str,
        move: str,
        context: str=""
    ):
        """Build user prompt to analyse next move action for Mistral 7B finetuned LLM.

        Args:
            fen (str): Chess FEN.
            player (str): White or Black for the next move.
            move (str): next move taken by player given FEN.
            context (str|dict, optional): context retrieved if applicable. Defaults to "".

        Returns:
            user_prompt (str): user prompt.
        """
        # Get context.
        context = self.get_context(context)

        if context == "":
            user_prompt = \
                f"Given chess board FEN '{fen}', explain briefly why {player} takes {move}?"
        else:
            user_prompt = \
                f"Given chess board FEN '{fen}' and context that '{context}', " \
                f"explain briefly why {player} takes {move}?"

        return user_prompt

# =============================================================================================================

class CoTGeneration(UserPromptBase):
    def __init__(self):
        """Initialize the instance.
        """
        super().__init__()

    def __call__(
        self,
        fen: str,
        player: str,
        move: str,
        with_cot_instruct: bool=True
    ):
        """Build user prompt to generate CoT analysis for next move action.

        Args:
            fen (str): Chess FEN.
            player (str): White or Black for the next move.
            move (str): next move taken by player given FEN.
            with_cot_instruct (bool, optional): use CoT as instruction. Defaults to True.

        Returns:
            user_prompt (str): user prompt.
        """
        if with_cot_instruct:
            user_prompt = \
                f"Given chess board FEN '{fen}' and context that '', " \
                f"explain briefly why {player} takes {move}?" \
                " If the context is useless, ignore it."
        else:
            user_prompt = \
                f"Question: 'Given chess board FEN '{fen}' and context that '', " \
                f"explain briefly why {player} takes {move}?" \
                " If the context is useless, ignore it.'\n" \
                "- Provide step-by-step reasoning to answer the question.\n" \
                "- Give a succinct answer starting with <ANSWER>: $answer." \

        return user_prompt