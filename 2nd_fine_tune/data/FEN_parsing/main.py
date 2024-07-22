from util import fen_to_board, get_piece_positions, count_pieces


def get_fen_state_explanation(fen: str) -> str:
    explanation = count_pieces(fen.split(" ")[0]) + "\n"
    board_state = fen_to_board(fen)
    explanation += f"Board State is:\n{board_state}\n"
    explanation += "Piece Positions are:\n"
    piece_positions = get_piece_positions(board_state)
    for piece, position in piece_positions.items():
        explanation += f"{piece} at {position}\n"

    return explanation


if __name__ == "__main__":
    all_fens = []
    with open("/workspaces/cognitive_AI_experiments/2nd_fine_tune/FEN_parsing/FEN.txt", "r") as f:
        all_fens = f.readlines()
        all_fens = all_fens[:50]
        all_fens = [fen.replace("-", "/").replace("\n", "")+" _" for fen in all_fens]

    f.close()

    # FEN = "4r1k1/1pR3p1/p2pn1qp/8/PPBP4/1QP1n3/3N2PP/5RK1 w - - 1 26"
    # print(get_fen_state_explanation(FEN))

    fen_parsing = open("fen_parsing.txt", "w")
    for fen in all_fens:
        fen_parsing.write(fen.split(" ")[0]+"\n")
        fen_parsing.write("-"*5+"\n")
        fen_parsing.write("Empty chess board square is denoted by '1'\n")
        fen_parsing.write(get_fen_state_explanation(fen)+"\n")
        fen_parsing.write("-"*5+"\n")

    fen_parsing.close()
    
