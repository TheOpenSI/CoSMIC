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
    FEN = "4r1k1/1pR3p1/p2pn1qp/8/PPBP4/1QP1n3/3N2PP/5RK1 w - - 1 26"
    print(get_fen_state_explanation(FEN))
