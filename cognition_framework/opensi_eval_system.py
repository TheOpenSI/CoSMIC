from engines.chess_engine.chess import ChessEngine
from engines.llm_engine.llm import LLMEngine


# =============================================================================================================

class OpenSIEvalSystem():
    """
    Define an evaluation system for OpenSI.
    """
    def __init__(self):
        # Set up chess engine for __next_move__
        self.chess_engine = ChessEngine()

        # Set up LLM for QA interaction
        prompt_template = "Give the answer of question {question} based on context {context}."

        self.llm_engine = LLMEngine(
            model_name='mistral',
            prompt_variables=['question', 'context'],
            prompt_template=prompt_template
        )

    def finish(self):
        self.chess_engine.finish()

    def run(self, query, context=''):
        if query.find('__next_move__') > -1:
            # Set a new board
            self.chess_engine.initialize_engine()

            # Parse move string
            current_move = query.split('__next_move__')[-1]

            # Use context to indicate the move mode
            move_mode = context
            if move_mode == '': move_mode = 'algebric'

            # Predict the best move, a string style can be automatically parsed in the chess engine
            next_move = self.chess_engine.run(
                current_move=current_move,
                move_mode=move_mode,
                is_last_move=True
            )

            # Display as an answer
            result = f"The next move of {[current_move]} is {next_move}."
        elif query.find('__update_store__') > -1:
            result = None
        else:
            # Run the LLM engine to get the answer
            result = self.llm_engine.run(query, context)

            # Parse the answer
            result = result.strip()

        return result

# =============================================================================================================

if __name__ == '__main__':
    # Build constructor of eval system
    qa_system = OpenSIEvalSystem()

    # Set a bunch of questions, can also read from .csv
    query_list = [
        ["What's the capital of Australia"],
        ["__next_move__ d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4"]
    ]
        # '__update__store__ Real Madrid won 15 UCL titles'

    # Loop over questions to get the answers
    for idx, query in enumerate(query_list):
        # Parse query
        question = query[0]
        context = ''
        if len(query) >= 2: context = query[1]

        # Solve the problem
        answer = qa_system.run(question, context=context)
        print(f"\n=== Question: {question}.\n==> Answer: {answer}\n")

    # Finish the system
    qa_system.finish()