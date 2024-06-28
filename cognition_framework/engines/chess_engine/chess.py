# Install visualization tool by pip install fenToBoardImage
from fentoboardimage import fenToImage, loadPiecesFolder

# Install chess engine by pip install python-chess
# Check API docs at https://python-chess.readthedocs.io
# Check the notations at https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
# Source code at https://github.com/Monika-After-Story/MonikaModDev/blob/master/Monika%20After%20Story/
# game/python-packages/chess/__init__.py

import chess, os, imageio
import chess.engine


# =============================================================================================================

class ChessEngine():
    def __init__(
        self,
        binary_path='',
        enable_visualization=False,
        visualization_dir=''
    ):
        # Set config and status
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../../.."
        self.enable_visualization = enable_visualization
        self.visualization_dir = visualization_dir
        self.valid_move_modes = ['algebric', 'coordinate']
        self.image_list = []  # to generate GIF animates if enable_visualization is on
        self.game_count = -1  # one engine can have multiple games
        self.move_count = 0  # count moves for each game
        self.is_game_over = False
        self.board = chess.Board()

        # Initialize the engine
        self.initialize_engine()
    
        # Set initial path for stockfish
        if binary_path == '':
            binary_path = f'{self.root}/third_party/stockfish/stockfish-ubuntu-x86-64-avx2'

        # This will kill the entire program, better just alert, but so far use assert
        assert os.path.exists(binary_path), \
            f'!!!Error, stockfish binary file not exist: {binary_path}.'

        # Set chess engine
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)

        # Create image visualization path and list
        if enable_visualization and self.visualization_dir == '':
            self.visualization_dir = f"{self.root}/viz"

    def finish(self):
        self.engine.quit()

    def initialize_engine(self):  # start a new game
        # Reset every in the board
        self.board.reset()

        # Initialize all global parameters
        self.image_list = []
        self.game_count += 1
        self.move_count = 0
        self.is_game_over = False

    def get_move_count(self):
        return self.move_count

    def get_game_count(self):
        return self.game_count

    def check_game_over(self):
        return self.is_game_over

    def get_current_board(self):
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

    def push_single(self, current_move, move_mode):
        # Convert from algebraic to coordinate
        if move_mode == 'algebric':
            # Check if it is legal, setting an initial board has no legal castling moves for O-O
            # assert current_move in [v for v in self.board.legal_moves], \
            #     f"!!! Error illegal move: {current_move}."

            current_move = str(self.board.parse_san(current_move))

        # if the current move is valid, then push to the board
        self.board.push(chess.Move.from_uci(current_move))

        # Add one to valid move count
        self.move_count += 1

    def run(self, current_move='', move_mode='coordinate', is_last_move=False):
        # Check if move_mode is valid
        self._check_move_mode(move_mode)

        # This is coordinate format, if it is empty, no push but automatic estimate
        if current_move != '':
            # Parse moves in a string, if it is a '' also put it to a list
            if isinstance(current_move, str):
                current_move = [str(v) for v in current_move.split(' ') if v != '']

            # If push a list with multiple moves in advance
            for current_move_per in current_move:
                self.push_single(current_move_per, move_mode)

        # Check if game over
        self.is_game_over = self.board.is_game_over()

        if not self.is_game_over:
            # Analyse for next move
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=1))

            # Predict the next move by .uci() and set to the board
            next_move = info['pv'][0].uci()

            # Conver to the same input move format
            if move_mode == 'algebric':
                next_move = self.board.san(info['pv'][0])
        else:
            # If game over, no 'pv' in info for analysis
            next_move = None

        # Save image for visualization
        if self.enable_visualization:
            self.visualize_next_move(is_last_move=is_last_move or self.is_game_over)

        return next_move