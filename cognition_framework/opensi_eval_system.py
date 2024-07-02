import os
import pandas as pd

from engines.chess_engine.chess import ChessEngine
from engines.llm_engine.llm import LLMEngine


# =============================================================================================================

class OpenSIEvalSystem:
    """
    Design an evaluation system for OpenSI.
    """
    def __init__(self,
        document_dir='',
        document_paths='',  # can be a list
        retrieve_score_threshold=0
    ):
        # Set up chess engine for __next_move__
        self.chess_engine = ChessEngine()

        # Set up LLM prompt template
        prompt_template = \
        "Always answer the question, even if the context isn't helpful." \
        " Write a response that appropriately completes the request, do not" \
        " write any explanation nor response, only answer.\n\n" \
        "### Instruction:\n use the context '{context}'" \
        "and answer the question '{question}'.\n\n" \
        "### Response:"

        # Set up LLM engine
        self.llm_engine = LLMEngine(
            llm_model_name='mistral',
            document_analyser_model_name='gte-small',
            prompt_variables=['question', 'context'],
            prompt_template=prompt_template,
            retrieve_score_threshold=retrieve_score_threshold
        )

        # Update database through all .pdf in a folder
        if os.path.exists(document_dir):
            self.add_document_directory(document_dir)

        # Update database through some .pdf
        if os.path.exists(document_paths):
            self.add_documents(document_paths)

    def finish(self):
        self.chess_engine.finish()

    def add_documents(self, document_paths):
        self.llm_engine.add_documents(document_paths)

    def add_document_directory(self, document_dir):
        self.llm_engine.add_document_directory(document_dir)

    def __call__(self, query, context='', topk=1):
        if query.find('__next__move__') > -1:
            # Set a new board
            self.chess_engine.initialize_engine()

            # Parse move string
            current_move = query.split('__next__move__')[-1]

            # Use context to indicate the move mode
            move_mode = context
            if move_mode == '': move_mode = 'algebric'

            # Predict the best move, a string style can be automatically parsed in the chess engine
            next_move = self.chess_engine(
                current_move=current_move,
                move_mode=move_mode,
                is_last_move=True
            )

            # Display as an answer
            result = f"The next move of {[current_move]} is {next_move}."
        elif query.find('exit') > -1:
            result = 'exit'
        else:
            # Update context from text has no question, thus updating the database only
            if query.find('__update__store__') > -1:
                context = query.split('__update__store__')[-1]
                update_database_only = True
            else:
                update_database_only = False

            # Check if context is a .pdf
            is_context_a_document = context.find('.pdf') > -1

            # Run the LLM engine to get the answer
            result = self.llm_engine(
                query,
                context,
                is_cotext_a_document=is_context_a_document,
                update_database_only=update_database_only,
                topk=topk
            )

            # Parse the answer
            if result is not None: result = result.strip()

        return result

# =============================================================================================================

if __name__ == '__main__':
    # Switch on this to avoid massive warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get the file's absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = f"{current_dir}"

    # Build constructor of eval system
    qa_system = OpenSIEvalSystem(
        document_dir=f'{root}/doc',
        retrieve_score_threshold=0.7  # filter out low-confidence retrieved context
    )

    # Externally add other documents
    qa_system.add_documents(f'{root}/test_doc/ucl_2023.pdf')

    # Externally add a document directory
    qa_system.add_document_directory(f'{root}/test_doc')

    # Set a bunch of questions, can also read from .csv
    df = pd.read_csv(f"{root}/tests/test.csv")
    queries = df["Question"]
    answers = df["Answer"]

    # Initialize quality variables
    num_q_tests = 0
    num_q_success_tests = 0

    # Loop over questions to get the answers
    for idx, (query, gt) in enumerate(zip(queries, answers)):
        # Solve the problem
        answer = qa_system(query, topk=5)

        # Print the answer
        if answer is not None:
            if answer == 'exit':
                break  # exit as requested
            else:
                # Next move has gt which is nan so skip
                if not isinstance(gt, str): continue

                # Assign to q variables: if successful, +1.
                num_q_tests += 1

                if answer.find(gt) > -1: num_q_success_tests += 1

                print(f"\n=== Question: {query}.\n==> Answer: {answer}\n")

    # Release the system
    qa_system.finish()

    # Print out the success rate
    success_ratio = num_q_success_tests / num_q_tests
    print(f"Success ratio: {success_ratio * 100.:.2f}% ({num_q_success_tests} out of {num_q_tests}).")