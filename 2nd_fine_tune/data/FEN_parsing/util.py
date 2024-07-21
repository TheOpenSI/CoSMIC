import re


def count_characters(input_string):
    character_count = {}
    for char in input_string:
        if char in character_count:
            character_count[char] += 1
        else:
            character_count[char] = 1
    return character_count


def fen_to_board(fen: str) -> str:
    """
    Converts a FEN string into a board state
    :param fen: fen string, e.g. 2r3k1/p4p1p/4pp2/1b6/pP1P3P/P3PN2/2rN1PP1/R3K2R w KQ - 1 20
    :return: the board state, with empty square denoted as 1
    """
    rows = fen.split(" ")[0].split("/")
    board = ""
    for row in rows:
        board_row = ""
        for char in row:
            if char.isdigit():
                board_row += " 1" * int(char)
            else:
                board_row += " " + char
        board += board_row + "\n"
    return board


def count_pieces(fen: str) -> str:
    """
    Counts the number of pieces in a FEN string
    :param fen: the original FEN string
    :return: number of chess pieces present in the game
    """
    white_piece_pattern = r"[A-Z]"
    black_piece_pattern = r"[a-z]"
    piece_dictionary = {"P": "White Pawn", "N": "White Knight",
                        "B": "White Bishop", "R": "White Rook",
                        "Q": "White Queen", "K": "White King",
                        "p": "Black Pawn", "n": "Black Knight",
                        "b": "Black Bishop", "r": "Black Rook",
                        "q": "Black Queen", "k": "Black King"}

    white_pieces_count = len(re.findall(white_piece_pattern, fen))
    white_pieces = count_characters(re.findall(white_piece_pattern, fen))
    black_pieces_count = len(re.findall(black_piece_pattern, fen))
    black_pieces = count_characters(re.findall(black_piece_pattern, fen))
    return (f'''Total {white_pieces_count + black_pieces_count} pieces present in the board\n
            White Pieces: {white_pieces_count} pieces which are {", ".join([f"{white_pieces[p]} x {piece_dictionary[p]}" for p in white_pieces.keys()])}\n
            Black Pieces: {black_pieces_count} pieces which are {", ".join([f"{black_pieces[p]} x {piece_dictionary[p]}" for p in black_pieces])}
            ''')


def get_piece_positions(board: str) -> dict[str, str]:
    """
    Gets the positions of all pieces in Algebraic Notation
    :param board: output of fen_to_board
    :return: Algebraic Notation for each piece present in a dictionary
    """
    position = {}
    # each row will have 16 chars since I added space before everything
    rank_dictionary = {0: "a", 2: "b", 4: "c", 6: "d", 8: "e", 10: "f", 12: "g", 14: "h"}
    file_dictionary = {0: "8", 1: "7", 2: "6", 3: "5", 4: "4", 5: "3", 6: "2", 7: "1"}
    piece_dictionary = {"P": "White Pawn", "N": "White Knight",
                        "B": "White Bishop", "R": "White Rook",
                        "Q": "White Queen", "K": "White King",
                        "p": "Black Pawn", "n": "Black Knight",
                        "b": "Black Bishop", "r": "Black Rook",
                        "q": "Black Queen", "k": "Black King"}
    pattern_to_detect_pieces = r"[a-zA-Z]"
    board_state = board.split("\n")[:-1]

    for file, row in enumerate(board_state):
        row = row.strip()
        for rank, piece in enumerate(row):
            if re.fullmatch(pattern_to_detect_pieces, piece):
                prev_string = position[piece_dictionary[piece]] if position.get(piece_dictionary[piece]) else ""
                position[piece_dictionary[piece]] = f"{prev_string} {rank_dictionary[rank]}{file_dictionary[file]}".strip()
                # print(f"[DEBUG] Piece {piece_dictionary[piece]}, File {file_dictionary[file]}, Rank {rank_dictionary[rank]}")

    return position


# FEN = "1R6/p3k1p1/2p2b2/2Pn4/1BQPB3/P7/6PP/3q2K1 w - - 1 33"
# board_state = fen_to_board(FEN)
# piece_position = get_piece_positions(board_state)
# print(piece_position)

