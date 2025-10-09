from fentoboardimage import fenToImage, loadPiecesFolder, loadArrowsFolder

board = fenToImage(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    squarelength=200,
    pieceSet=loadPiecesFolder("pieces"),
    ArrowSet=loadArrowsFolder("arrows"),
    darkColor="#79a65d",
    lightColor="#daf2cb",
    Arrows=[
        ["a1", "a8"],
        ["b1", "d2"]
    ]
)

board.show()
