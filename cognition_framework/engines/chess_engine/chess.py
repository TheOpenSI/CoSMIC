# Install visualization tool by pip install fenToBoardImage
from fentoboardimage import fenToImage, loadPiecesFolder
from stockfish import Stockfish

# Install chess engine by pip install python-chess
# Check API docs at https://python-chess.readthedocs.io
# Check the notations at https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
# Source code at https://github.com/Monika-After-Story/MonikaModDev/blob/master/Monika%20After%20Story/
# game/python-packages/chess/__init__.py

import chess, os, imageio, sys
import chess.engine
import numpy as np

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from utils.log_tool import set_color


# =============================================================================================================

class ChessEngine:
    def __init__(
        self,
        binary_path='',
        enable_visualization=False,
        visualization_dir='',
        back_end='stockfish'
    ):
        # Set config and status
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.enable_visualization = enable_visualization
        self.visualization_dir = visualization_dir
        self.back_end = back_end
        self.valid_move_modes = ['algebric', 'coordinate']
        self.image_list = []  # to generate GIF animates if enable_visualization is on
        self.game_count = -1  # one engine can have multiple games
        self.move_count = 0  # count moves for each game
        self.is_game_over = False
        self.board = chess.Board()
    
        # Set initial path for stockfish
        if binary_path == '':
            binary_path = f'{self.root}/third_party/stockfish/stockfish-ubuntu-x86-64-avx2'

        # This will kill the entire program, better just alert, but so far use assert
        assert os.path.exists(binary_path), \
            f'!!!Error, stockfish binary file not exist: {binary_path}.'

        # Set chess engine
        if self.back_end == 'chess-engine':
            self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)
        else:
            self.engine = Stockfish(
                binary_path, depth=20,
                parameters={"Threads": 2, "Minimum Thinking Time": 30}
            )

        # Create image visualization path and list
        if enable_visualization and self.visualization_dir == '':
            self.visualization_dir = f"{self.root}/viz"

        # Initialize the engine
        self.initialize_engine()

    def quit(self):
        if self.back_end == 'chess-engine':
            self.engine.quit()

    def initialize_engine(self):  # start a new game
        # Reset board
        self.reset_board()

        # Initialize all global parameters for the current engine
        self.game_count += 1

    def reset_board(self):
        # Reset every in the board
        self.board.reset()

        # Reset stockfish
        if self.back_end == 'stockfish':
            self.engine.reset_engine_parameters()

        # Initialize all global parameters for the current board
        self.is_game_over = False
        self.move_count = 0
        self.image_list = []

    def get_move_count(self):
        return self.move_count

    def get_game_count(self):
        return self.game_count

    def check_game_over(self):
        return self.is_game_over

    def get_fen(self):
        return self.board.fen()

    def _check_move_mode(self, move_mode):
        # Only support algebric mode and coordinate mode
        assert move_mode in self.valid_move_modes, \
            f"!!!Error, unknown move mode: {move_mode}, only support {self.valid_move_modes}."

    def visualize_next_move(self, is_last_move=False):
        # For visualization.
        # Download piece images from https://github.com/ReedKrawiec/Fen-To-Board-Image.git
        # Alternatively, online FEN visualization at https://www.dailychess.com/chess/chess-fen-viewer.php
        boardImage = fenToImage(
            fen=self.board.fen(),
            squarelength=100,
            pieceSet=loadPiecesFolder(f"{self.root}/third_party/Fen-To-Board-Image/examples/pieces"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
        )

        # Save for visualization
        save_dir = f'{self.visualization_dir}/game_{self.game_count}'
        save_image_dir = f'{save_dir}/images'
        os.makedirs(save_image_dir, exist_ok=True)

        # Save next move image
        boardImage.save(f'{save_image_dir}/{self.move_count}.png')
        self.image_list.append(boardImage)

        # Generate animate
        if is_last_move:
            print('Generate animate ...')
            imageio.mimsave(
                f'{save_dir}/{self.move_count}_moves.gif',
                self.image_list,
                'GIF',
                fps=1
            )

    def convert_algebric_to_coordinate(self, move):
        return str(self.board.parse_san(move))

    def convert_coordinate_to_algebric(self, move):
        # If move is a string, then convert to chess.Move for the board to parse
        if not isinstance(move, chess.Move):
            move = chess.Move.from_uci(move)

        return str(self.board.san(move))

    def check_is_legal_move(self, move, move_mode):
        # self.board requires coordinate format
        if move_mode == 'algebric':
            move = self.convert_algebric_to_coordinate(move)

        # Check if the move is in the legal moves on the current board
        is_legal = move in [str(v) for v in self.board.legal_moves]

        return is_legal

    def push_single(self, current_move, move_mode):
        # Check if the move is legal
        is_legal = self.check_is_legal_move(current_move, move_mode)

        if not is_legal:
            print(set_color('error', f"\nMove {current_move} is illegal on FEN {self.board.fen}."))

            return -1

        # Convert from algebraic to coordinate
        if move_mode == 'algebric':
            current_move = self.convert_algebric_to_coordinate(current_move)

        # If the current move is valid, then push to the board
        self.board.push(chess.Move.from_uci(current_move))

        # Also push the move to stockfish
        if self.back_end == 'stockfish':
            self.engine.make_moves_from_current_position([current_move])

        # Add one to valid move count
        self.move_count += 1

        return 0

    def set_fen(self, fen):
        # Reset the board to set a fen
        self.reset_board()

        # Set the FEN
        self.board.set_fen(fen)

        # Set FEN to stockfish
        if self.back_end == 'stockfish':
            self.engine.set_fen_position(fen)

    def puzzle_solve(self, fen, move_mode):
        # Clean the current board and restart the game
        self.set_fen(fen)

        # Estimate the next moves for the current FEN then automatic till game over
        next_move_list = self.__call__(
            move_mode=move_mode,
            is_puzzle=True
        )

        return next_move_list

    def get_next_moves(self):
        # All next moves on the current board, unnecessary to be the best move(s)
        next_move_list = [v for v in self.board.generate_legal_moves()]

        return next_move_list

    def get_best_moves(self, info, is_puzzle=False):
        # Get the color to be checkmated from FEN
        fen = self.board.fen()
        color_to_be_checked = fen.split(' ')[1]
        assert color_to_be_checked in ['w', 'b']

        mate_list = []

        for info_per in info:
            # Get the score object for the best solution(s)
            scorer = info_per['score']

            # Get the status to be checkmated
            if is_puzzle:
                if color_to_be_checked == 'w':
                    mate = scorer.white().mate()
                else:
                    mate = scorer.black().mate()

                # None means unlikely to be checkmated
                if mate is None: mate = -np.inf
            else:
                # chess.WHITE wins, that is mate (all negative) maximum or cp maximum (if no mate)
                mate = scorer.white().mate()
                if mate is None: mate = scorer.white().cp

            mate_list.append(mate)

        # Find the solution(s) with the maximum negative value
        least_move_for_checkmate = np.max(mate_list)

        # Find the indices with the least_move_for_checkmate
        least_move_for_checkmate_indices = \
            [idx for idx, mate in enumerate(mate_list) if mate == least_move_for_checkmate]

        # Find the result with each solution or move as a list
        if is_puzzle:
            # For puzzle, return all the moves in the solution(s) as a list
            result = [info[idx]['pv'] for idx in least_move_for_checkmate_indices]
        else:
            # For next move, return the first move of the solution(s) as a list
            result = [info[idx]['pv'][0] for idx in least_move_for_checkmate_indices]

        return result

    def __call__(
        self,
        current_move='',
        move_mode='coordinate',
        is_last_move=False,
        is_puzzle=False
    ):
        # Check if move_mode is valid
        self._check_move_mode(move_mode)

        # This is coordinate format, if it is empty, no push but automatic estimate
        if current_move != '':
            # Parse moves in a string, if it is a '' also put it to a list
            if isinstance(current_move, str):
                current_move = [str(v.replace('.', '')) for v in current_move.split(' ') if v != '']

            # If push a list with multiple moves in advance
            for current_move_per in current_move:
                self.push_single(current_move_per, move_mode)

        # Check if game over
        self.is_game_over = self.board.is_game_over()

        if not self.is_game_over:
            # For puzzle, to get the best solution(s) with multiple moves; otherwise, just general next moves
            # For next moves, get_best_moves() can also return best move(s) as next moves,
            # but to the present, just general next moves from get_next_moves()
            if self.back_end == 'stockfish':
                if is_puzzle:
                    # This returns only one move
                    best_move = self.engine.get_best_move()
                    best_solution_list = [best_move]
                else:
                    # This returns topk moves
                    top_moves = self.engine.get_top_moves(5)
                    best_solution_list = [v['Move'] for v in top_moves]
            else:
                if is_puzzle:
                    # Analyse for next move, depth for the moves of a solution, multipv for multiple solutions
                    info = self.engine.analyse(self.board, chess.engine.Limit(depth=20), multipv=50)

                    # This is a customized best move estimator (since inbuilt BestMove function will be deadlock)
                    # The best_solution_list can be multiple solution with multiple moves
                    best_solution_list = self.get_best_moves(info, is_puzzle=is_puzzle)
                else:
                    best_solution_list = self.get_next_moves()

            # Convert the move format to string
            next_move = []

            for best_solution in best_solution_list:
                if is_puzzle and self.back_end == 'chess-engine':
                    # The chess-engine returns a solution of multiple moves
                    next_move.append([str(v) for v in best_solution])
                else:
                    # Eeach solution is a move
                    if move_mode == 'algebric':
                        best_solution = self.convert_coordinate_to_algebric(best_solution)
                        next_move.append(best_solution)
                    else:
                        # Append directly as it is already coordinate required
                        next_move.append(str(best_solution))
        else:
            # If game over, no 'pv' in info for analysis
            next_move = ''

        # Save image for visualization
        if self.enable_visualization:
            self.visualize_next_move(is_last_move=is_last_move or self.is_game_over)

        return next_move