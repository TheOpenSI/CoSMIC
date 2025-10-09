import unittest
import os
from fentoboardimage import fenToImage, loadPiecesFolder, loadArrowsFolder, CoordinatePositionFn, loadFontFile
from PIL import ImageChops
from operator import itemgetter
from PIL import ImageDraw
from PIL import Image


class FenToBoardImageTest(unittest.TestCase):

    def test_outputs(self):
        image1 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=125,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
        )
        image2 = Image.open("./boards/board1.png")
        diff = ImageChops.difference(image1, image2)
        self.assertEqual(diff.getbbox(), None)

    def test_board_colors(self):
        image1 = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#D18B47",
            lightColor="#FFCE9E"
        )
        image2 = Image.open("./boards/board2.png")
        diff = ImageChops.difference(image1, image2)
        self.assertEqual(diff.getbbox(), None)

    def test_pieces(self):
        image1 = fenToImage(
            fen="rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2 ",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces2"),
            darkColor="#909090",
            lightColor="#fffefe"
        )
        image2 = Image.open("./boards/board3.png")
        diff = ImageChops.difference(image1, image2)
        self.assertEqual(diff.getbbox(), None)

    def test_flip(self):
        original = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces2"),
            darkColor="#909090",
            lightColor="#fffefe"
        )
        flipped = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces2"),
            darkColor="#909090",
            lightColor="#fffefe",
            flipped=True
        )
        original_image = Image.open("./boards/board4.png")
        flipped_image = Image.open("./boards/board5.png")
        self.assertEqual(ImageChops.difference(
            original, original_image).getbbox(), None)
        self.assertEqual(ImageChops.difference(
            flipped, flipped_image).getbbox(), None)

    def test_lastMove(self):
        lastMoveDrawn = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            lastMove={
                "before": "a1",
                "after": "g7",
                "darkColor": "#a9a238",
                "lightColor": "#cdd269"
            }
        )
        lastMoveDrawnFlipped = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            flipped=True,
            lastMove={
                "before": "a1",
                "after": "g7",
                "darkColor": "#a9a238",
                "lightColor": "#cdd269"
            }
        )
        i = Image.open("./boards/board6.png")
        flipped = Image.open("./boards/board7.png")
        self.assertEqual(ImageChops.difference(
            lastMoveDrawn, i).getbbox(), None)
        self.assertEqual(ImageChops.difference(
            lastMoveDrawnFlipped, flipped).getbbox(), None)

    def test_KnightArrows(self):
        arrows = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            Arrows=[
                ("d5", "b4"),
                ("d5", "c3"),
                ("d5", "e3"),
                ("d5", "f4"),
                ("d5", "f6"),
                ("d5", "e7"),
                ("d5", "c7"),
                ("d5", "b6"),
            ]
        )
        flipped = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            flipped=True,
            Arrows=[
                ("d5", "b4"),
                ("d5", "c3"),
                ("d5", "e3"),
                ("d5", "f4"),
                ("d5", "f6"),
                ("d5", "e7"),
                ("d5", "c7"),
                ("d5", "b6"),
            ]
        )
        i = Image.open("./boards/board8.png")
        i_2 = Image.open("./boards/board9.png")
        self.assertEqual(ImageChops.difference(arrows, i).getbbox(), None)
        self.assertEqual(ImageChops.difference(flipped, i_2).getbbox(), None)

    def test_HorizontalArrows(self):
        arrows = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            Arrows=[
                ("a8", "h8"),
                ("b7", "g7"),
                ("c6", "f6"),
                ("d5", "e5"),
                ("e4", "d4"),
                ("f3", "c3"),
                ("g2", "b2"),
                ("h1", "a1"),
            ]
        )
        arrows_flipped = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            flipped=True,
            Arrows=[
                ("a8", "h8"),
                ("b7", "g7"),
                ("c6", "f6"),
                ("d5", "e5"),
                ("e4", "d4"),
                ("f3", "c3"),
                ("g2", "b2"),
                ("h1", "a1"),
            ]
        )
        i = Image.open("./boards/board10.png")
        self.assertEqual(ImageChops.difference(arrows, i).getbbox(), None)
        i2 = Image.open("./boards/board11.png")
        self.assertEqual(ImageChops.difference(
            arrows_flipped, i2).getbbox(), None)

    def test_VerticalArrows(self):
        arrows = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            Arrows=[
                ("a8", "a1"),
                ("b7", "b2"),
                ("c6", "c3"),
                ("d5", "d4"),
                ("e4", "e5"),
                ("f3", "f6"),
                ("g2", "g7"),
                ("h1", "h8"),
            ]
        )
        arrows_flipped = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            flipped=True,
            Arrows=[
                ("a8", "a1"),
                ("b7", "b2"),
                ("c6", "c3"),
                ("d5", "d4"),
                ("e4", "e5"),
                ("f3", "f6"),
                ("g2", "g7"),
                ("h1", "h8"),
            ]
        )
        i = Image.open("./boards/board12.png")
        i2 = Image.open("./boards/board13.png")
        self.assertEqual(ImageChops.difference(arrows, i).getbbox(), None)
        self.assertEqual(ImageChops.difference(
            arrows_flipped, i2).getbbox(), None)

    def test_DiagArrows(self):
        arrows = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            ArrowSet=loadArrowsFolder("./arrows1"),
            Arrows=[
                ("a8", "h1"),
                ("b8", "h2"),
                ("c8", "h3"),
                ("d8", "h4"),
                ("e8", "h5"),
                ("f8", "h6"),
                ("g8", "h7"),
                ("g1", "a7"),
                ("a6", "f1"),
                ("a1", "c3"),
                ("h6", "c1"),
                ("a3", "c1"),
                ("a2", "b1"),
            ]
        )
        arrows_flipped = fenToImage(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            squarelength=100,
            pieceSet=loadPiecesFolder("./pieces"),
            darkColor="#909090",
            lightColor="#fffefe",
            flipped=True,
            ArrowSet=loadArrowsFolder("./arrows1"),
            Arrows=[
                ("a8", "h1"),
                ("b8", "h2"),
                ("c8", "h3"),
                ("d8", "h4"),
                ("e8", "h5"),
                ("f8", "h6"),
                ("g8", "h7"),
                ("g1", "a7"),
                ("a6", "f1"),
                ("a1", "c3"),
                ("h6", "c1"),
                ("a3", "c1"),
                ("a2", "b1"),
            ]
        )
        i = Image.open("./boards/board14.png")
        i2 = Image.open("./boards/board15.png")
        self.assertEqual(ImageChops.difference(arrows, i).getbbox(), None)
        self.assertEqual(ImageChops.difference(
            arrows_flipped, i2).getbbox(), None)

    def test_coordinates(self):
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
        image2 = Image.open("./boards/board16.png")
        image3 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=40,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            coordinates={
                "font": loadFontFile("./fonts/Roboto-Bold.ttf"),
                "size": 18,
                "darkColor": "#daf2cb",
                "lightColor": "#79a65d",
                "positionFn": CoordinatePositionFn["innerBorder"],
                "padding": 5,
                "outsideBoardColor": "#000000"
            }
        )
        image4 = Image.open("./boards/board17.png")
        image5 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=40,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            coordinates={
                "font": loadFontFile("./fonts/Roboto-Bold.ttf"),
                "size": 18,
                "darkColor": "#daf2cb",
                "lightColor": "#79a65d",
                "positionFn": CoordinatePositionFn["everySquare"],
                "padding": 5,
                "outsideBoardColor": "#000000"
            }
        )
        image6 = Image.open("./boards/board18.png")

        diff = ImageChops.difference(image1, image2)
        self.assertEqual(diff.getbbox(), None)

        diff = ImageChops.difference(image3, image4)
        self.assertEqual(diff.getbbox(), None)

        diff = ImageChops.difference(image5, image6)
        self.assertEqual(diff.getbbox(), None)

    def test_coordinates_flipped(self):
        image1 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=125,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            flipped=True,
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
        image2 = Image.open("./boards/board19.png")
        image3 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=40,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            flipped=True,
            coordinates={
                "font": loadFontFile("./fonts/Roboto-Bold.ttf"),
                "size": 18,
                "darkColor": "#daf2cb",
                "lightColor": "#79a65d",
                "positionFn": CoordinatePositionFn["innerBorder"],
                "padding": 5,
                "outsideBoardColor": "#000000"
            }
        )
        image4 = Image.open("./boards/board20.png")
        image5 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=40,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            flipped=True,
            ArrowSet=loadArrowsFolder("./arrows1"),
            Arrows=[
                ("d5", "b4"),
                ("d5", "c3"),
                ("d5", "e3"),
                ("d5", "f4"),
                ("d5", "f6"),
                ("d5", "e7"),
                ("d5", "c7"),
                ("d5", "b6"),
            ],
            coordinates={
                "font": loadFontFile("./fonts/Roboto-Bold.ttf"),
                "size": 18,
                "darkColor": "#daf2cb",
                "lightColor": "#79a65d",
                "positionFn": CoordinatePositionFn["everySquare"],
                "padding": 5,
                "outsideBoardColor": "#000000"
            }
        )
        image6 = Image.open("./boards/board21.png")

        diff = ImageChops.difference(image1, image2)
        self.assertEqual(diff.getbbox(), None)

        diff = ImageChops.difference(image3, image4)
        self.assertEqual(diff.getbbox(), None)

        diff = ImageChops.difference(image5, image6)
        self.assertEqual(diff.getbbox(), None)

    def test_highlighting_square(self):
        image1 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=125,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            highlighting={
                ("#ff0000", "#702963"): ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
                ("#00ff00", "#2e7d32"): ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
                "#0000ff": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
            }
        )
        image2 = Image.open("./boards/board22.png")
        image3 = fenToImage(
            fen="8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44",
            squarelength=125,
            pieceSet=loadPiecesFolder(
                "./pieces"),
            darkColor="#79a65d",
            lightColor="#daf2cb",
            flipped=True,
            highlighting={
                ("#ff0000", "#702963"): ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
                ("#00ff00", "#2e7d32"): ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
                "#0000ff": ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
            }
            )
        image4 = Image.open("./boards/board23.png")
        diff = ImageChops.difference(image1, image2)
        self.assertEqual(diff.getbbox(), None)
        diff = ImageChops.difference(image3, image4)
        self.assertEqual(diff.getbbox(), None)

if __name__ == '__main__':
    unittest.main()
