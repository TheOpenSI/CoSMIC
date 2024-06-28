from chess_engine.chess import ChessEngine


# =============================================================================================================

if __name__ == '__main__':
    # Construct the chess engine
    chess_engine = ChessEngine(enable_visualization=True)

    # Set three examples
    for case in [0, 1, 2]:
        if case == 0:
            # One move string from '../data/generated_data_last_500.csv'
            sample_list = [
                "e4 e6 d4 b6 e5 Bb7 Nf3 h6 Bd3 g5 O-O g4 Nfd2 h5 Ne4 Nc6 Be3 Qe7 Qd2 Bh6 Bxh6 Nxh6 Nf6+" \
                " Kd8 Bh7 Nf5 Bxf5 exf5 c3 h4 Qg5 g3 fxg3 hxg3 Qxg3 Qf8 Rxf5 Ne7 Rg5 Ng6 Nd2 Qh6 Rh5 Qg7" \
                " Qg4 Bc8 Rxh8+ Qxh8 Rf1 d6 Qg5 Qh4 Qe3 Bb7 e6"
            ]
            move_mode = 'algebric'
        elif case == 1:
            # Some move examples from '../data/generated_data_last_500.csv'
            sample_list = ['e4', 'e6', 'd4', 'b6']  # push all once
            move_mode = 'algebric'
        elif case == 2:
            # Estimating from scratch for maximum 200 moves, can be cut when game overs.
            num_moves = 200  # random number, if too large will be game over automatically
            sample_list = [''] * num_moves  # of course, this can be replaced by while()
            move_mode = 'coordinate'

        # Init the board for multiple games, this is to mimic multiple __next_move__ in .csv
        chess_engine.initialize_engine()

        # Run
        for idx, current_move in enumerate(sample_list):
            # Check if auto predict by chess engine or not
            if current_move == '':
                if idx == 0:
                    current_move = ''
                    next_move = ''
                else:
                    # Set the move loop
                    current_move = next_move

            # Check if the last move to enable animate
            is_last_move = idx == len(sample_list) - 1

            # Run for each move
            next_move = chess_engine.run(
                current_move=current_move,
                move_mode=move_mode,
                is_last_move=is_last_move
            )

            # Show next move, especially for prediction
            print(
                f"Index: {idx}, current move: {current_move}" \
                f", next move: {next_move}, game over: {chess_engine.check_game_over()}."
            )

            # Quit
            if is_last_move or chess_engine.check_game_over():
                break

        # Check engine status
        print(
            f"Is game over: {chess_engine.check_game_over()}" \
            f", number of moves: {chess_engine.get_move_count()}."
        )

    # Need to close the engine
    chess_engine.finish()