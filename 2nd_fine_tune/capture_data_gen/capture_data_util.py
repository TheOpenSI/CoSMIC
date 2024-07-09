import typing


# defining chess piece dictionary
CHESS_PIECE = {
    "a": "Pawn",
    "b": "Pawn",
    "c": "Pawn",
    "d": "Pawn",
    "e": "Pawn",
    "f": "Pawn",
    "g": "Pawn",
    "h": "Pawn",
    "R": "Rook",
    "N": "Knight",
    "B": "Bishop",
    "Q": "Queen",
    "K": "King"
}

# initial position
# small letters are all pawns
INITIAL_POSITIONS = {
  "a1": "R", "b1": "N", "c1": "B", "d1": "Q", "e1": "K", "f1": "B", "g1": "N", "h1": "R",
  "a2": "a", "b2": "a", "c2": "a", "d2": "a", "e2": "a", "f2": "a", "g2": "a", "h2": "a",
  "a8": "R", "b8": "N", "c8": "B", "d8": "Q", "e8": "K", "f8": "B", "g8": "N", "h8": "R",
  "a7": "a", "b7": "a", "c7": "a", "d7": "a", "e7": "a", "f7": "a", "g7": "a", "h7": "a"
}


def get_pair(moves : str) -> typing.Tuple[str, str]:
    """
    args : moves - list of algebraic chess moves. eg. [e4 e6 d4 d5 Nd2 c5 exd5]
    returns : pair of moves sep by \n. eg. [White : e4, Black : e6 \n.....]
              len(pair)
    """
    target = moves.split(" ")
    cot_moves = [f"White: {target[i]}, Black: {target[i+1]}" if i + 1 < len(target) else f"White: {target[i]}" for i in range(0, len(target), 2)]
    return "\n".join(cot_moves), len(cot_moves)


def get_capture_details(current_move : str, all_moves : str, index : int) -> typing.Tuple[str, str]:
    """
    The function get called after 'x' is detected in the move no filtering added
    args : current_move - split the move_pair and pass each element. e.g White : e5
           all_moves - list of moves, same as the input of move_pair, used to look for last occurrence
           index - index of current move in all_moves
    return : explanation - x detected, in move m, piece_a captured piece_b at position p
             captured piece - which piece have been captured, provides the capturing piece and captured piece

    """
    exp = ""

    # colour
    capturing_colour = current_move.split(" ")[0][:-1] # getting rid of ':'
    captured_colour = "Black" if capturing_colour == "White" else "White"
    
    # split list string
    list_moves = all_moves.split(" ")
    
    # this is the current move like White: Kxg3, target is the move only Kxg3
    target = current_move.split(" ")[1]
    
    #add to explanation
    exp+=f"letter 'x' present in {target}\n"
    
    # capturing piece
    target_piece = CHESS_PIECE[target[0]]
    
    # position like a8
    target_position = target[-2:] if len(target) == 4 else target[-3:-1] # this is for cases like Nxa8+
    # print(f"[DEBUG] Target {target}")
    # print(f"[DEBUG] Target position {target_position}")
    
    # captured position
    capture_position = ""
    for i in range(index-1, -1, -1):
        if target_position in list_moves[i]:
            capture_position = list_moves[i]
            break

    # captured piece
    # print(f"[DEBUG] Capture position {capture_position}")
    try:
        captured_piece = CHESS_PIECE[INITIAL_POSITIONS[target_position]] if capture_position == "" else CHESS_PIECE[capture_position[0]]
    except KeyError:
        print("[DEBUG] En Passant | Special Chess Move detected assigning Pawn")
        captured_piece = "Pawn"
    
    exp+=f"{capturing_colour} {target_piece} captured {captured_colour} {captured_piece} at position {target_position}\n"
    return exp, f"{captured_colour} {captured_piece}, "


def get_explanation(move_pairs : str, all_moves : str) -> str:
    """
    args : move_pairs - all moves in pair format from get_pair
           all_moves - list of all moves, to call get_capture_details
    returns : explanation - if no capture - no x detected
                            if capture -    x detected
                                            which piece captured which piece
                            summary -       white captured n pieces which are ...
                                            black captured n pieces whixh are ...
                            total -         in the game total x pieces were captured
    """
    exp = ""
    total_capture = 0
    white_capture = 0 # black piece
    white_captured = ""
    
    black_capture = 0 # white piece
    black_captured = ""
    
    list_move_pairs = move_pairs.split("\n")
    
    for i, move_pair in enumerate(list_move_pairs):
        # add the move string
        exp+=f"\n{move_pair}\n"
        
        move = move_pair.split(",")
        if "x" in move_pair:
            if "x" in move[0]:
                white_capture+=1
                explanation, captured_piece = get_capture_details(move[0].strip(), all_moves, i*2)
                exp+=explanation
                white_captured+=captured_piece
            try:
                if "x" in move[1]:
                    black_capture+=1
                    explanation, captured_piece = get_capture_details(move[1].strip(), all_moves, i*2+1)
                    exp+=explanation
                    black_captured+=captured_piece
            except IndexError:
                print("[DEBUG] Black move not present, IndexError caught. -- Improved logic will be added")

        else:
            exp+=f"(no capture at {move[0]} since {move[0]} does not contain 'x')\n"
            try:
                exp+=f"(no capture at {move[1]} since {move[1]} does not contain 'x')\n"          
            except IndexError:
                print("[DEBUG] Black move not present, IndexError caught. -- Improved logic will be added")
        
        total_capture += move_pair.count("x")

    exp+="\nSummary\n"
    exp+=f"White captured {white_capture} pieces which are {white_captured[:-2]}\n"
    exp+=f"Black captured {black_capture} pieces which are {black_captured[:-2]}\n"
    exp+="\nTotal Capture\n"
    exp+=f"Total {total_capture} pieces were captured in the game"

    return exp