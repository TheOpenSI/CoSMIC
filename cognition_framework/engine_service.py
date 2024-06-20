from stockfish import Stockfish
import time

def get_best_move(list_of_moves:str):
    # use stockfish to predict next move
    moves = list_of_moves.split(" ")
    print("[INFO] Generating next best move using Chess Engine Service STOCKFISH")
    time.sleep(5)
    print("[STOCKFISH] Next best move will be __test__move__")