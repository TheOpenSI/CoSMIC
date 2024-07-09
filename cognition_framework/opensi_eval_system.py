import os
import pandas as pd

from engines.chess_engine.chess import ChessEngine
from engines.llm_engine.llm import LLMEngine
from utils.log_tool import set_color


# =============================================================================================================

class OpenSIEvalSystem:
    """
    Design an evaluation system for OpenSI.
    """
    def __init__(self,
        document_dir='',
        document_paths='',  # can be a list
        retrieve_score_threshold=0,
        llm_back_end='instance',
        chess_back_end='stockfish'
    ):
        # Set root for the location of this file relative to the repository
        self.root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
        self.chess_back_end = chess_back_end

        # Set up chess engine for __next_move__
        self.chess_engine = ChessEngine(back_end=chess_back_end)

        # Set up LLM engine
        self.llm_engine = LLMEngine(
            llm_model_name='mistral',
            document_analyser_model_name='gte-small',
            retrieve_score_threshold=retrieve_score_threshold,
            back_end=llm_back_end  # options: 'instance'/'chat'
        )

        # Update database through all .pdf in a folder
        self.add_document_directory(document_dir)

        # Update database through some .pdf
        self.add_documents(document_paths)

    def quit(self):
        self.chess_engine.quit()

    def add_documents(self, document_paths):
        self.llm_engine.add_documents(document_paths)

    def add_document_directory(self, document_dir):
        self.llm_engine.add_document_directory(document_dir)

    def parse_puzzle_csv(self, puzzle_path):
        # Parse .csv to get all FENs
        df = pd.read_csv(puzzle_path)
        fens = df['FEN']
        best_cases = df['best_case']
        players = df['player']
        best_move_list = df['moves']

        # When start from black, the number of moves is across two blobs, thus minus 1
        best_case_updated = []

        for idx, player in enumerate(players):
            # Convert to the number of moves, not blobs
            if player == 'White':
                num_moves = (best_cases[idx] - 1) * 2
            else:
                num_moves = best_cases[idx] * 2

            best_case_updated.append(num_moves)

        # Parse the ground truth moves given the FEN
        # Remove namespace and . for each piece of moves
        best_move_updated = []

        for best_moves in best_move_list:
            best_move_updated.append([v for v in best_moves.split(' ') if v.find('.') <= -1])

        # Set a dictionary
        info = {'fen': fens, 'best_case': best_case_updated, 'moves': best_move_updated}

        return info

    def __call__(self, query, context='', topk=1):
        if query.find('exit') > -1:
            result = 'exit'
        elif query.find('__next__move__') > -1:
            # Set a new board
            self.chess_engine.initialize_engine()

            # Parse move string
            current_move = query.split('__next__move__')[-1]

            # Use context to indicate the move mode
            move_mode = context
            if move_mode == '': move_mode = 'algebric'

            # Predict the best move, a string style can be automatically
            # parsed in the chess engine
            next_move = self.chess_engine(
                current_move=current_move,
                move_mode=move_mode,
                is_last_move=True
            )

            # Display as an answer
            result = f"The next move of {[current_move]} is {next_move}"
        elif query.find('puzzle') > -1 and query.find('.csv') > -1:
            # Use context to indicate the move mode
            move_mode = context
            if move_mode == '': move_mode = 'algebric'

            # Remove all namespace
            puzzle_path = f"{self.root}/{query.replace(' ', '')}"

            if os.path.exists(puzzle_path):
                puzzle_info = self.parse_puzzle_csv(puzzle_path)

                # Get fens for puzzle solving
                fens = puzzle_info['fen']
                gt_move_lists = puzzle_info['moves']
                num_fens = len(fens)

                # Store the actual processed fen(s)
                puzzle_solve_info = {'fen': [], 'best_case': [], 'solution': []}

                # Set a .csv file containing multiple FEN to estimate the next move
                for idx, (fen, gt_move_list) in enumerate(zip(fens, gt_move_lists)):
                    if idx % 10 == 0 or idx == num_fens - 1:
                        print(set_color('info', f"Solving puzzles {idx + 1}/{num_fens}..."))

                    # Reset board
                    self.chess_engine.reset_board()

                    if self.chess_back_end == 'stockfish':
                        # Set initial FEN since this will be update in every round
                        current_fen = fen
                        next_moves = []

                        for idx_move, gt_move in enumerate(gt_move_list):
                            # Stockfish only returns the best move, so push FEN to get the best move
                            next_move = self.chess_engine.puzzle_solve(current_fen, move_mode=move_mode)[0]

                            # Even step is the opponent, odd step is the player
                            # Only push next_move for the player, and gt_move for the opponent
                            if idx_move % 2 == 0:
                                next_move = gt_move

                            # Push the estimate move to the board
                            self.chess_engine.push_single(next_move, move_mode=move_mode)

                            # Then update the FEN in chess engine
                            current_fen = self.chess_engine.get_current_board()

                            # Save the actual move to next_moves
                            next_moves.append(next_move)

                        # Assume each puzzle can have multiple solutions, the above is one of them
                        # thus, as a list, it is in another list
                        next_moves = [next_moves]
                    else:
                        # Call to solve each puzzle
                        next_moves = self.chess_engine.puzzle_solve(fen, move_mode)

                    # Store information
                    puzzle_solve_info['fen'].append(fen)
                    puzzle_solve_info['best_case'].append(puzzle_info['best_case'][idx])
                    puzzle_solve_info['solution'].append(next_moves)

                # Print for progress checking
                print(set_color('info', "Solving puzzles finished."))

                result = puzzle_solve_info
            else:
                result = f"!!! Error, {puzzle_path} not exist."
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

        return result

# =============================================================================================================

if __name__ == '__main__':
    # Switch on this to avoid massive warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get the file's absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = f"{current_dir}/.."

    # Set config
    llm_back_end = 'chat'  # chat/instance
    chess_back_end = 'stockfish'  # stockfish/chess-engine

    # Build constructor of eval system
    qa_system = OpenSIEvalSystem(
        retrieve_score_threshold=0.7,  # filter out low-confidence retrieved context
        llm_back_end=llm_back_end,
        chess_back_end=chess_back_end
    )

    # Externally add other documents, a string or a list of strings
    qa_system.add_documents(f'{root}/cognition_framework/doc/ucl_2023.pdf')

    # Externally add a document directory
    qa_system.add_document_directory(f'{root}/cognition_framework/doc')

    # Set a bunch of questions, can also read from .csv
    df = pd.read_csv(f"{root}/cognition_framework/tests/test.csv")
    queries = df["Question"]
    answers = df["Answer"]

    # Initialize quality variables
    num_q_tests = 0
    num_q_success_tests = 0

    # Loop over questions to get the answers
    for idx, (query, gt) in enumerate(zip(queries, answers)):
        # Skip marked questions
        if query.find('skip') > -1: continue

        # Solve the problem
        answer = qa_system(query, topk=5)

        # Print the answer
        if answer is not None:
            if answer == 'exit':
                break  # exit as requested
            elif isinstance(gt, str):  # compare with GT string
                # Assign to q variables: if successful, +1.
                num_q_tests += 1

                if answer.find(gt) > -1:
                    num_q_success_tests += 1
                    status = 'success'
                else:
                    status = 'fail'

                print(set_color(status, f"\nQuestion: '{query}' with GT: {gt}.\nAnswer: '{answer}'.\n"))
            else:
                if query.find('puzzle') > -1 and query.find('.csv') > -1:
                    # Get information from puzzle_solve_info for comparison
                    fens = answer['fen']
                    best_cases = answer['best_case']
                    solutions = answer['solution']

                    # Get puzzle success ratio
                    num_q_test_puzzle = 0
                    num_q_success_test_puzzle = 0
                    num_q_tests += 1

                    for fen, best_case, solution in zip(fens, best_cases, solutions):
                        # All the best solution for one FEN will be the same, divide by 2 for the number of blobs
                        print('###', fen, solution[0])
                        solution_length = len(solution[0])

                        # Statistic of the tests
                        num_q_test_puzzle += 1

                        # If the best solution has less blobs, it succeeds
                        if solution_length <= best_case:
                            num_q_success_test_puzzle += 1
                        else:
                            print(set_color(
                                'fail',
                                f"\nQuestion: {query}." \
                                f"\n==> The solution for FEN\n{fen}\nis" \
                                f"\n{[f'{v_idx}: {v}' for v_idx, v in enumerate(solution)]}" \
                                f"\nthe number of moves in the solution(s) is \n{[len(v) for v in solution]}.\n")
                            )

                    # Print information of puzzle success rate
                    if num_q_test_puzzle == 0:
                        puzzle_success_rate = 0.0
                    else:
                        puzzle_success_rate = float(num_q_success_test_puzzle) / float(num_q_test_puzzle)

                    # Only all puzzles are successful, this test succeeds
                    if puzzle_success_rate == 1: num_q_success_tests += 1

                    # Print locally
                    print(set_color(
                        'info',
                        f"Puzzle from {query}" \
                        f" has success rate: {puzzle_success_rate * 100.:.2f}%" \
                        f" ({num_q_success_test_puzzle} out of {num_q_test_puzzle}).")
                    )
                else:
                    print(set_color('info', f"\nQuery: {query}.\nAnswer: {answer}."))

    # Release the system
    qa_system.quit()

    # Print out the success rate
    if num_q_tests == 0:  # in case no questions in the test file
        success_ratio = 0.0
    else:
        success_ratio = num_q_success_tests / num_q_tests

    print(set_color(
        'info',
        f"Success ratio: {success_ratio * 100.:.2f}%" \
        f" ({num_q_success_tests} out of {num_q_tests}).\n")
    )