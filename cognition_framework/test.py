from stockfish import Stockfish
stockfish = Stockfish("/home/s448780/workspace/cognitive_ai/cognition_framework/stockfish/stockfish-ubuntu-x86-64-avx2")
# print(stockfish.get_best_move())
FEN = "1Q6/5ppk/3p4/4r3/4q3/2P5/P5PP/R5K1 w - - 3 25"
stockfish.set_fen_position(FEN)
print(stockfish.get_parameters())

# print(stockfish.get_best_move())