# Install visualization tool by pip install fenToBoardImage
from fentoboardimage import fenToImage, loadPiecesFolder

# Install chess engine by pip install python-chess
import chess, os, imageio
import chess.engine


# =============================================================================================================

if __name__ == '__main__':
    # Set engine path and build constructor, download from XAI Files
    binary_path = '../third_party/stockfish/stockfish-ubuntu-x86-64-avx2'
    engine = chess.engine.SimpleEngine.popen_uci(binary_path)

    # Set initial chess board
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    # Set config
    num_moves = 200
    num_valid_moves = 0
    image_list = []
    move_mode = 'coordinate'
    assert move_mode in ['algmove', 'coordinate'], f'!!!Unknown format {move_mode}.'
    os.makedirs(f'../viz/{move_mode}/images', exist_ok=True)

    if move_mode == 'algmove':
        # Some examples from '../data/generated_data_last_500.csv'
        loop_range = ['e4', 'e6', 'd4', 'b6']
    else:
        loop_range = range(num_moves)  # of course, this can be replaced by while()

    # Run
    for idx, algmove in enumerate(loop_range):
        # Analyse for next move
        info = engine.analyse(board, chess.engine.Limit(depth=1))

        # This is coordinate format
        if move_mode == 'algmove':
            # Convert from algebraic to coordinate
            next_move = str(board.parse_san(algmove))
        else:  # default coordinate format
            # Predict the next move by .uci() and set to the board
            next_move = info['pv'][0].uci()

        # Assign to execute the move
        board.push(chess.Move.from_uci(next_move))

        num_valid_moves += 1

        # Now check the updated fen
        print(
            f'Move mode: {move_mode}, ID: {idx + 1}' \
            f', current board: {board.fen()}, next move: {next_move}'
        )

        # Check if game over
        if board.is_game_over():
            break

        # For visualization.
        # Download piece images from https://github.com/ReedKrawiec/Fen-To-Board-Image.git
        # Alternatively, online FEN visualization at https://www.dailychess.com/chess/chess-fen-viewer.php
        boardImage = fenToImage(
            fen=board.fen(),
            squarelength=100,
            pieceSet=loadPiecesFolder("../third_party/Fen-To-Board-Image/examples/pieces"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
        )

        # Save for visualization
        boardImage.save(f'../viz/{move_mode}/images/{idx + 1}.png')
        image_list.append(boardImage)

    engine.quit()

    # Convert to animate
    imageio.mimsave(f'../viz/{move_mode}/{num_valid_moves}_moves.gif', image_list, 'GIF', fps=1)