import os, csv, numbers
import pandas as pd
import numpy as np

# from difflib import SequenceMatcher
from engines.chess_engine.chess import ChessEngine
from engines.chess_engine.chess_gpt import ChessEngineGPT
from engines.llm_engine.llm import LLMEngine
from utils.log_tool import set_color
from utils.num2word import convert_number2word


# =============================================================================================================

class OpenSIEvalSystem:
    """
    Design an evaluation system for OpenSI.
    """
    def __init__(self,
        llm_model='mistral-7b-v0.1',
        document_dir='',
        document_paths='',  # can be a list
        retrieve_score_threshold=0,
        chess_back_end='stockfish',
        chess_best_move_predictor='stockfish',
        seed=0
    ):
        # Set root for the location of this file relative to the repository
        self.root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
        self.chess_back_end = chess_back_end
        self.chess_best_move_predictor = chess_best_move_predictor

        # Set up chess engine for __next_move__
        self.chess_engine = ChessEngine(back_end=chess_back_end)

        # Set up LLM engine
        self.llm_engine = LLMEngine(
            llm_model=llm_model,
            document_analyser_model='gte-small',
            retrieve_score_threshold=retrieve_score_threshold,
            seed=seed
        )

        # Set up the best move predictor
        if chess_best_move_predictor == 'stockfish':
            self.chess_best_move_engine = self.chess_engine
        else:
            self.chess_best_move_engine = ChessEngineGPT(llm_model=chess_best_move_predictor)

        # Update database through all .pdf in a folder
        self.add_document_directory(document_dir)

        # Update database through some .pdf
        self.add_documents(document_paths)

    def quit(self):
        self.chess_engine.quit()
        self.llm_engine.quit()

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
            # Truncate to best the moves only
            best_move_updated_per = [v for v in best_moves.split(' ') if v.find('.') <= -1]

            # Always add # to the last move
            best_move_updated_per[-1] += '#'

            # Assign to the entire question list
            best_move_updated.append(best_move_updated_per)

        # Set a dictionary
        info = {
            'fen': fens,
            'best_case': best_case_updated,
            'moves': best_move_updated,
            'player': players
        }

        return info

    def parse_quality_csv(self, csv_path):
        # Read data
        df = pd.read_csv(csv_path)

        # Set a dictionary
        info = {
            'question': df['Question'],
            'answer': df['Answer']
        }

        return info

    def __call__(self, query, context='', topk=1, log_file=None, is_rag=False):
        # Since all questions will go to context retriever as RAG first,
        # check the retriever score for visualization
        retriever_score = None
        raw_result = None

        if query.find('exit') > -1:
            result = 'exit'
        elif query.find('skip') > -1:
            result = None
        elif query.find('__next__move__') > -1:
            # Set a new board
            self.chess_engine.reset_board()

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

            # Get the current FEN
            current_fen = self.chess_engine.get_fen()

            # The first move from White and then Black given an initial board
            if len(current_move) % 2 == 0:
                player = 'white'
            else:
                player = 'black'

            # Convert next move to a list for loopy analysis
            if not isinstance(next_move, list): next_move = [next_move]

            # Interaction with LLM for each next move
            for next_move_per in next_move:
                analysis = self.llm_engine.chess_analysis(player=player, move=next_move_per, fen=current_fen)

                # Display the analysis
                print(set_color('info', f"The next move of {[current_move]} is {next_move_per}\nAnalysis: {analysis}.\n"))

            # Display as an answer
            result = f"The next move of {[current_move]} is one of {next_move}."
        elif query.find('.csv') > -1:
            # Remove all namespace
            query = query.replace(' ', '')

            # Return if file is invalid
            if not os.path.exists(query):
                return f"!!! Error, {query} not exist."

            if query.find('puzzle') > -1 :
                puzzle_info = self.parse_puzzle_csv(query)

                # Use context to indicate the move mode
                move_mode = context
                if move_mode == '': move_mode = 'algebric'

                # Get fens for puzzle solving
                fens = puzzle_info['fen']
                gt_move_lists = puzzle_info['moves']
                players = puzzle_info['player']
                num_fens = len(fens)
                score_list = []

                # Set a .csv file containing multiple FEN to estimate the next move
                for idx, (fen, gt_moves, player) in enumerate(zip(fens, gt_move_lists, players)):
                    if idx % 10 == 0 or idx == num_fens - 1:
                        print(set_color('info', f"Solving puzzles {idx + 1}/{num_fens}..."))

                    # Reset board
                    self.chess_engine.reset_board()

                    if self.chess_back_end == 'stockfish':
                        # Set initial FEN since this will be update in every round
                        current_fen = fen
                        next_moves = []

                        # Set the fen
                        self.chess_engine.set_fen(current_fen)

                        for idx_move, gt_move in enumerate(gt_moves):
                            # Even step is the opponent, odd step is the player
                            # Only push next_move for the player, and gt_move for the opponent
                            if idx_move % 2 == 0:
                                next_move = gt_move
                            else:
                                # Stockfish only returns the best move, so push FEN to get the best move
                                next_move = self.chess_best_move_engine.puzzle_solve(current_fen, move_mode=move_mode)[0]

                                # 20240723 Remove for the checkmate next_move, in case the model just analyse the move by #
                                # next_move = next_move.replace('#', '')

                                # Interaction between LLM and Chess engine
                                analysis = self.llm_engine.chess_analysis(
                                    player=player,
                                    move=next_move,
                                    fen=current_fen,
                                    is_rag=is_rag,
                                    topk=topk
                                )

                                # Display the analysis
                                print(set_color(
                                    'info',
                                    f"Puzzle {idx + 1} at step {idx_move + 1}:" \
                                    f" player {player} takes {next_move} given FEN '{current_fen}'.\n" \
                                    f"Analysis: {analysis}\n")
                                )

                                # Save to csv
                                if log_file is not None:
                                    log_file.writerow([
                                        f"Puzzle {idx + 1} at step {idx_move + 1}: " \
                                        f"player {player} takes {next_move} given FEN '{current_fen}'",
                                        analysis
                                    ])

                            try:
                                # Push the estimate move to the board
                                status = self.chess_engine.push_single(next_move, move_mode=move_mode)
                            except:
                                # If any error raised, the next move is invalid
                                print(set_color('fail', f"Move {next_move} for FEN {current_fen} is invalid."))
                                status = -1

                            # If it is illegal, then stop the program
                            if status < 0:
                                next_moves.append(next_move)
                                break
                            else:
                                # Then update the FEN in chess engine
                                current_fen = self.chess_engine.get_fen()

                                # Save the actual move to next_moves
                                next_moves.append(next_move)

                        # Check if next moves are the same as GT moves, only on the player which is odd
                        comparison_list = [float(next_v == gt_v) for idx_inner, (next_v, gt_v) in enumerate(zip(next_moves, gt_moves)) if idx_inner % 2 == 1]
                        score_per = np.mean(np.array(comparison_list))
                    else:
                        # Call to solve each puzzle, not yet to be used for score calculation
                        next_moves = self.chess_engine.puzzle_solve(fen, move_mode)

                    # Save to score list
                    score_list.append(score_per)

                    # Save to log file
                    if log_file is not None:
                        log_file.writerow([fen, next_moves, gt_moves, score_per])

                # Calculate the overall score of all puzzles in the puzzle file
                average_score = np.mean(score_list)

                # Print for progress checking
                print(set_color('info', "Solving puzzles finished."))

                # Save to log file
                if log_file is not None:
                    log_file.writerow([query, "", "", f"{average_score:.4f}"])

                result = average_score
            elif query.find('attention') > -1 \
                or query.find('memory') > -1 \
                or query.find('perception') > -1:
                rag_info = self.parse_quality_csv(query)
                questions = rag_info['question']
                answers = rag_info['answer']
                num_questions = len(questions)
                score_list = []

                # Set tag for information print
                if query.find('attention') > -1:
                    quality_tag = 'Attention'
                elif query.find('memory') > -1:
                    quality_tag = 'Memory'
                elif query.find('perception') > -1:
                    quality_tag = 'Perception'
                else:
                    quality_tag = 'Unknown'

                for idx, (question, gt_answer) in enumerate(zip(questions, answers)):
                    # Print progress
                    if idx % 10 == 0 or idx == num_questions - 1:
                        print(set_color("info", f"Solving {quality_tag} {idx + 1}/{num_questions}..."))

                    # Inner loop to call self.__call__ for update
                    result, retriever_score, raw_result = self.__call__(
                        question,
                        log_file=log_file,
                        topk=topk,
                        is_rag=is_rag
                    )

                    # For __update__store__, the result is None
                    if result is None: continue

                    # Change None to 'N/A' for comparison
                    if (gt_answer is None) or (isinstance(gt_answer, numbers.Number) and np.isnan(gt_answer)):
                        gt_answer = 'N/A'

                    # Case insensitive and remove line change for better readability
                    if isinstance(gt_answer, numbers.Number) or gt_answer.isdigit():
                        # Convert number to word and compare both number and string format answer
                        gt_answer = [str(int(gt_answer)), str(convert_number2word(int(gt_answer)))]
                    elif isinstance(gt_answer, str):
                        # A number can be read as a string, so convert it to a number
                        gt_answer = gt_answer.lower().replace('\n', '#newline')

                    if isinstance(result, str):
                        result = result.lower().replace('\n', '#newline')
                        raw_result = raw_result.lower().replace('\n', '#newline')

                    # Check if the answer word is in the prediction
                    if isinstance(gt_answer, list):
                        score_per = float(len(np.nonzero([float(result.find(v) > -1) for v in gt_answer])[0]) > 0)
                    else:
                        score_per = float(result.find(gt_answer) > -1)

                    # Push per question score to the list
                    score_list.append(score_per)

                    # Save to log
                    if log_file is not None:
                        log_file.writerow([question, result, gt_answer, score_per, retriever_score[0], raw_result])
                        print(set_color(
                            'info',
                            f'Question: {question}, Result: {result}, Retrieve score: {retriever_score[0]:.4f}.')
                        )

                # Average the score
                average_score = np.mean(np.array(score_list))
                result = average_score

                # Save to log
                if log_file is not None:
                    log_file.writerow([query, "", "", f"{average_score:.4f}"])
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
            result, retriever_score, raw_result = self.llm_engine(
                query,
                context,
                is_cotext_a_document=is_context_a_document,
                update_database_only=update_database_only,
                topk=topk,
                is_rag=is_rag
            )

        return result, retriever_score, raw_result

# =============================================================================================================

if __name__ == '__main__':
    # Switch on this to avoid massive warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Whether use GPT or Stockfish to predict the next move
    chess_best_move_predictor = 'stockfish'

    # Only support certain best move predictors
    assert chess_best_move_predictor in ['gpt4', 'stockfish'], \
        print(set_color('error', f'Unknown best move predictor: {chess_best_move_predictor}'))

    # Get the file's absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = f"{current_dir}/.."

    # Set llm_model_list to run all at once
    llm_model_list = [
        # "mistral-7b-v0.1",
        "mistral-7b-instruct-v0.1",
        # "gemma-7b",
        "gemma-7b-it",
        # "mistral-7b-finetuned",
        # "mistral-7b-finetuned-new",
        "gpt-4o"
    ]

    # Run all models at once
    for llm_model in llm_model_list:
        # Print head information
        print(set_color('info', f"\n######## Evaluation with {llm_model} ########\n"))

        # Build constructor of eval system
        qa_system = OpenSIEvalSystem(
            llm_model=llm_model,
            retrieve_score_threshold=0.7,  # filter out low-confidence retrieved context
            chess_best_move_predictor=chess_best_move_predictor
        )

        # Externally add a document directory
        qa_system.add_document_directory(f"{root}/cognition_framework/doc")

        # Set a bunch of questions, can also read from .csv
        df = pd.read_csv(f"{root}/data/test.csv")
        queries = df["Question"]
        answers = df["Answer"]

        # Initialize quality variables
        test_score_list = []

        # Loop over questions to get the answers
        for idx, (query, gt) in enumerate(zip(queries, answers)):
            # Skip marked questions
            if query.find('skip') > -1: continue

            # Create a log file
            if query.find(".csv") > -1:
                # 20240725 For other qualities, which excludes the reasoning (puzzle analysis), use RAG
                # For puzzle analysis, switch off RAG because from the experiments, adding RAG ruins the analysis
                if query.find('puzzle') > -1:
                    is_rag = False
                else:
                    is_rag = True

                query = f"{root}/{query.replace(' ', '')}"

                # Change the data folder to results for log file
                log_file = query.replace('/data/', f'/results/{llm_model}/')

                # Create a folder
                log_file_name = log_file.split('/')[-1]
                log_dir = log_file.replace(log_file_name, '')
                os.makedirs(log_dir, exist_ok=True)

                # Open a log file and pass the instance
                log_file = log_file.replace('.csv', '_isragTrue.csv') if is_rag else log_file.replace('.csv', '_isragFalse.csv')

                # Add a tag to the log file to distinguish Stockfish or GPT as best move predictor for Chess games
                if query.find('puzzle') > -1:
                    log_file = log_file.replace('.csv', f'_{chess_best_move_predictor}AsBestMovePredictor.csv')

                log_file_pt = open(log_file, 'w')
                log_file = csv.writer(log_file_pt)

                # Write heads
                log_file.writerow(["Question", "Answer", "Label", "Score", "Comment", "Raw Answer"])
            else:
                log_file_pt = None
                log_file = None

            # Solve the problem
            answer, _, _ = qa_system(query, topk=5, log_file=log_file, is_rag=is_rag)

            # Print the answer
            if answer is not None:
                if answer == 'exit':
                    break  # exit as requested
                elif isinstance(gt, str):  # compare with GT string
                    # Assign to q variables
                    status = 'success' if (answer.find(gt) > -1) else 'fail'

                    # Set the score to score list
                    test_score_list.append(float(answer.find(gt) > -1))

                    print(set_color(status, f"\nQuestion: '{query}' with GT: {gt}.\nAnswer: '{answer}'.\n"))
                else:
                    # The answer can be the average score from a .csv test file
                    if isinstance(answer, numbers.Number) or answer.isdigit():
                        test_score_list.append(float(answer))
                        status = 'success' if answer == 1 else 'fail'
                    else:
                        status = 'info'

                    print(set_color(status, f"\nQuery: {query}.\nAnswer: {answer}."))

            # Close log file pointer
            if log_file_pt is not None:
                log_file_pt.close()

        # Release the system
        qa_system.quit()

        # Print out the success rate
        success_ratio = np.mean(np.array(test_score_list))

        print(set_color(
            'info',
            f"Success ratio: {success_ratio * 100.:.2f}%" \
            f" on {len(test_score_list)} tests.\n")
        )