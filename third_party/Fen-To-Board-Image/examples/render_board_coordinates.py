from fentoboardimage import fenToImage, loadPiecesFolder, loadFontFile, CoordinatePositionFn

image1 = fenToImage(
    fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
    squarelength=125,
    pieceSet=loadPiecesFolder(
        "./pieces"),
    darkColor="#79a65d",
    lightColor="#daf2cb",
    coordinates={
        "font": loadFontFile("./fonts/Roboto-Bold.ttf"),
        "size": 28,
        "darkColor": "#daf2cb",
        "lightColor": "#79a65d",
        "positionFn": CoordinatePositionFn["outerBorder"],
        "padding": 15,
        "outsideBoardColor": "#000000"
    }
)

image1.show()
