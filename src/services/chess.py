# -------------------------------------------------------------------------------------------------------------
# File: chess.py
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

import os, chess, sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from stockfish import Stockfish
from utils.log_tool import set_color
from src.llms.llm import GPT35Turbo, GPT4o
from src.services.base import ServiceBase

# =============================================================================================================

class ChessBase(ServiceBase):
    def __init__(
        self,
        binary_path: str="",
        **kwargs
    ):
        """Base class for all chess questions.

        Args:
            binary_path (str, optional): stockfish executable file path. Defaults to "".
        """
        super().__init__(**kwargs)

        # Get root of this file to set an absolute path to the stockfish executable file path.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."

        self.valid_move_modes = ["algebric", "coordinate"]

        # Get an initial chess board.
        self.board = chess.Board()

        # To get the full player name from the shortname.
        self.PLAYER_DICT = {"w": "White", "b": "Black"}

        # Set initial path for stockfish.
        if binary_path == "":
            binary_path = f"{self.root}/third_party/stockfish/stockfish-ubuntu-x86-64-avx2"

        # This will kill the entire program, better just alert, but so far use assert.
        assert os.path.exists(binary_path), \
            set_color("error", f"!!!Error, stockfish binary file not exist: {binary_path}.")

        # Set chess engine.
        self.stockfish = Stockfish(
            binary_path,
            depth=20,
            parameters={"Threads": 2, "Minimum Thinking Time": 30}
        )

        # Initialize the engine.
        self.reset_board()

    def reset_board(self):
        """Reset stockfish board for each chess game.
        """
        # Reset every in the board.
        self.board.reset()

        # Reset stockfish.
        self.stockfish.reset_engine_parameters()
        self.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", True)

    def get_fen(self):
        """Get FEN of current chess board.

        Returns:
            fen (str): FEN of the current chess board.
        """
        return self.board.fen()

    def convert_algebric_to_coordinate(
        self,
        move: str
    ):
        """Convert move mode from algebric to coordinate.

        Args:
            move (str): a chess move in algebric mode.

        Returns:
            move (str): a chess move in coodinate mode.
        """
        return str(self.board.parse_san(move))

    def convert_coordinate_to_algebric(
        self,
        move: str
    ):
        """Convert move mode from coordinate to algebric.

        Args:
            move (str): a chess move in coodinate mode.

        Returns:
            move (str): a chess move in algebric mode.
        """
        # If move is a string, then convert to chess.Move for the board to parse.
        if not isinstance(move, chess.Move):
            move = chess.Move.from_uci(move)

        return str(self.board.san(move))

    def check_is_legal_move(
        self,
        move: str,
        move_mode: str
    ):
        """Check if the next move is legal for the current chess board.

        Args:
            move (str): a chess move.
            move_mode (str): move mode.

        Returns:
            is_legal (bool): check if the move is legal.
        """
        # self.board requires coordinate format.
        if move_mode == "algebric":
            move = self.convert_algebric_to_coordinate(move)

        # Check if the move is in the legal moves on the current board.
        is_legal = move in [str(v) for v in self.board.legal_moves]

        return is_legal

    def push_single(
        self,
        current_move: str,
        move_mode: str
    ):
        """Push a chess move to the current chess board.

        Args:
            current_move (str): a chess move.
            move_mode (str): move mode.

        Returns:
            status (int): 0 for legal move and -1 for illegal move.
        """
        # Check if the move is legal.
        is_legal = self.check_is_legal_move(current_move, move_mode)

        if not is_legal:
            print(set_color("error", f"\nMove {current_move} is illegal on FEN {self.board.fen}."))

            return -1

        # Convert from algebraic to coordinate.
        if move_mode == "algebric":
            current_move = self.convert_algebric_to_coordinate(current_move)

        # If the current move is valid, then push to the board.
        self.board.push(chess.Move.from_uci(current_move))

        # Also push the move to stockfish.
        self.stockfish.make_moves_from_current_position([current_move])

        return 0

    def set_fen(
        self,
        fen: str
    ):
        """Set FEN to the current chess board.

        Args:
            fen (str): a given chess FEN.
        """
        # Reset the board to set a given FEN.
        self.reset_board()

        # Set the FEN.
        self.board.set_fen(fen)

        # Set FEN to stockfish.
        self.stockfish.set_fen_position(fen)

    def get_next_moves(self):
        """Get all legal next moves given the current chess board.

        Returns:
            next_move_list (list): a list of legal moves for the current chess board.
        """
        # All next moves on the current board, unnecessary to be the best move(s)
        next_move_list = [v for v in self.board.generate_legal_moves()]

        return next_move_list

    def _check_move_mode(
        self,
        move_mode: str
    ):
        """Internally check if the move mode is legal.

        Args:
            move_mode (str): move mode.
        """
        # Only support algebric mode and coordinate mode.
        assert move_mode in self.valid_move_modes, \
            f"!!!Error, unknown move mode: {move_mode}, only support {self.valid_move_modes}."

    def __call__(
        self,
        current_move: str="",
        move_mode: str="coordinate",
        topk: int=1
    ):
        """Predict the next move given current_move.

        Args:
            current_move (str, optional): a move string or a sequence of moves. Defaults to "".
            move_mode (str, optional): move mode. Defaults to "coordinate".
            topk (int, optional): up to topk predicted moves returned. Defaults to 1.

        Returns:
            next_move (list): a list of predicted topk next move(s).
        """
        # Check if move_mode is valid.
        self._check_move_mode(move_mode)

        # For moves in coordinate mode, if it is empty, no push but automatic estimate.
        if current_move != "":
            # Parse moves in a string, if it is a "" also put it to a list.
            if isinstance(current_move, str):
                current_move = [str(v.replace(".", "")) for v in current_move.split(" ") if v != ""]

            # If push a list with multiple moves in advance.
            for current_move_per in current_move:
                self.push_single(current_move_per, move_mode)

        # Check if game over.
        is_game_over = self.board.is_game_over()

        next_move = []

        # If game over, return empty as next move.
        if is_game_over:
            return next_move

        # For puzzle, to get the best solution(s) with multiple moves; otherwise, just general next moves.
        if topk == 1:
            # Return only one move.
            best_move = self.stockfish.get_best_move()
            best_solution_list = [best_move]
        else:
            # Returns topk moves.
            top_moves = self.stockfish.get_top_moves(topk)
            best_solution_list = [v["Move"] for v in top_moves]

        for best_solution in best_solution_list:
            # Each solution is a move.
            if move_mode == "algebric":
                best_solution = self.convert_coordinate_to_algebric(best_solution)
                next_move.append(best_solution)
            else:
                # Append directly as it is already coordinate required.
                next_move.append(str(best_solution))

        return next_move

# =============================================================================================================

class StockfishSequenceNextMove(ChessBase):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """Predict the next move given a sequence of moves.
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        moves: str,
        move_mode: str="",
        topk: int=1
    ):
        """Predict the next move for a query.

        Args:
            moves (str): a string of moves.
            move_mode (str, optional): move mode. Defaults to "".
            topk (int, optional): topk move(s) to be predicted. Defaults to 1.

        Returns:
            next_move_list (list): a list of topk next move(s).
        """
        # Clean the current board and restart the game.
        self.reset_board()

        # Estimate the next moves for puzzle.
        next_move_list = super().__call__(
            current_move=moves,
            move_mode=move_mode,
            topk=topk
        )

        return next_move_list

# =============================================================================================================

class StockfishFENNextMove(ChessBase):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """Predict the next move given a chess FEN using Stockfish as backend.
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        fen: str,
        move_mode: str=""
    ):
        """Predict the next move for a query.

        Args:
            fen (str): a chess FEN.
            move_mode (str, optional): move mode. Defaults to "".

        Returns:
            next_move_list (list): a list of topk next move(s).
        """
        # Clean the current board and restart the game
        self.set_fen(fen)

        # Estimate the next moves for puzzle
        next_move_list = super().__call__(
            move_mode=move_mode,
            topk=1
        )

        return next_move_list

# =============================================================================================================

class GPTFENNextMove(ChessBase):
    def __init__(
        self,
        llm_name: str="gpt-4o",
        is_truncate_response: bool=True
    ):
        """Predict the next move given a chess FEN using GPT as backend.

        Args:
            llm_name (str, optional): LLM name. Defaults to "gpt-4o".
            is_truncate_response (bool, optional): truncate response for next move prediction.
            Defaults to True.
        """
        # Set config.
        self.llm_name = llm_name
        self.is_truncate_response = is_truncate_response

        # Use GPT 4o or 3.5-turbo.
        if llm_name == "gpt-4o":
            self.llm = GPT4o(user_prompt_instance_name="FenNextMovePredict")
        else:
            self.llm = GPT35Turbo(user_prompt_instance_name="FenNextMovePredict")

    def quit(self):
        """Release LLM memory cached on GPU and LLM instannce.
        """
        self.llm.close()
        del self.llm

    def get_player(
        self,
        fen: str
    ):
        """Parse FEN to get the player for the next move.

        Args:
            fen (str): a given chess FEN.

        Returns:
            player (str): the player's full name, White or Black.
        """
        # Get player shortname from FEN
        color_to_be_checked = fen.split(" ")[1]

        assert color_to_be_checked in ["w", "b"], \
            set_color("error", f"Unsupported player: {color_to_be_checked}.")

        return self.PLAYER_DICT[color_to_be_checked]

    def truncate_response(
        self,
        response: str
    ):
        """Truncate the response from LLM.

        Args:
            response (str): response from LLM.

        Returns:
            response (str): truncated response if applicable.
        """
        # Return full response if no truncation is required.
        if not self.is_truncate_response:
            return response

        # Truncate according to keywords of LLM.
        try:
            if self.llm_name  == "gpt-4o":
                response = response.split("**")[1].replace(".", "").split(" ")[-1]
            elif self.llm_model == "gpt-3.5-turbo":
                response = response.split("is ")[-1].replace(".", "").replace("*", "").split(" ")[-1]
        except:
            response = response.split("** is")[-1]

        response = response.replace("\n", "").strip()

        return response

    def __call__(
        self,
        fen: str,
        move_mode: str=""
    ):
        """Main entry to predict next move for a query.

        Args:
            fen (str): a chess FEN.
            move_mode (str, optional): move mode. Defaults to "".

        Returns:
            next_move: the next move predicted for the given FEN.
        """
        # Get the player's full name.
        player = self.get_player(fen)

        # Get the next move response.
        response = self.llm(fen, player)

        # Truncate to get the next move and set the next move in a list.
        next_move = self.truncate_response(response)
        next_move = [next_move]

        return next_move