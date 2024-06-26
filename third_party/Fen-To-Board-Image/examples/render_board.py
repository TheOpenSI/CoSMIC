from fentoboardimage import fenToImage, loadPiecesFolder

board = fenToImage(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    squarelength=200,
    pieceSet=loadPiecesFolder("pieces"),
    darkColor="#79a65d",
    lightColor="#daf2cb"
)

board.show()
