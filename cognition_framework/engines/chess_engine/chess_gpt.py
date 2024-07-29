import sys, os

from openai import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from utils.log_tool import set_color


sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")


# Set up OpenAI model name
LLM_MODEL_DICT = {
    "gpt4": "gpt-4o"
}

# =============================================================================================================

class ChessEngineGPT:
    def __init__(self, llm_model='gpt4'):
        # Set up query prompt
        prompt_template = "Given chess board FEN '{fen}'," \
            " give the next move of player {player} indexed by ** without any analysis?"

        self.prompt = PromptTemplate(
            input_variables=['player', 'fen'],
            template=prompt_template,
        )

        # To get the full player name from the shortname
        self.PLAYER_DICT = {'w': 'White', 'b': 'Black'}

        # Login OpenAI
        # Cannot hard-code this key as it is forbidden by GitHub
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."

        # Set the key stored file
        load_dotenv(f"{self.root}/.env")

        # Variable openai_key stores OpenAI key
        openai_key = os.getenv('openai_key')

        # Login OpenAI account
        self.model = OpenAI(api_key=openai_key)

        # Set up GPT chat template
        chat_template_context = lambda query: [
            {"role": "system", "content": "You are a helpful assistant. Always answer the question even if the context is not helpful"},
            {"role": "user", "content": query}
        ]

        # Build GPT object
        self.llm_reader = lambda query: \
            self.model.chat.completions.create(
                model=LLM_MODEL_DICT[llm_model],
                max_tokens=2048,
                temperature=0.0,
                messages=chat_template_context(query)
            ).choices[0].message.content

    def quit(self):
        self.model.close()
        del self.llm_reader

    def get_player(self, fen):
        # Get player shortname from FEN
        color_to_be_checked = fen.split(' ')[1]
        assert color_to_be_checked in ['w', 'b']

        return self.PLAYER_DICT[color_to_be_checked]

    def __call__(
        self,
        current_move='',
        move_mode='coordinate',
        is_last_move=False,
        is_puzzle=False
    ):
        print(set_color('error', 'Not implemented.'))

    def puzzle_solve(
            self,
            fen,
            move_mode=None
        ):
        # Get the player name
        player = self.get_player(fen)

        # Ask the question
        prompt = self.prompt.format(fen=fen, player=player)

        # Get the response
        response = self.llm_reader(prompt)

        # Truncate and set as a list
        next_move = response.split('**')[1].replace('.', '').split(' ')[-1]
        next_move = [next_move]

        return next_move