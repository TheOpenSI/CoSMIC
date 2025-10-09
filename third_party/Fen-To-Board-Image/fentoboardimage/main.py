#!/usr/bin/env python

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import math
from .Utils import indicesToSquare, squareToIndices, flipCoordTuple, flippedCheck
from .Coordinates import CoordinatePositionFn, paintCoordinateOverlay
from .FenParser import FenParser
from .Checkerboard import paintCheckerBoard
from .Pieces import paintAllPieces
from .Arrows import paintAllArrows

pieceCache = {}
resizedCache = {}


"""
Loads the sprites for a piece set, and prepares it for the fenToBoardImage function

Parameters
----------
path: str
    Loads piece set located at the path provided.

"""


def loadPiecesFolder(path, cache=True):
    if path in pieceCache:
        pieceImages = pieceCache[path]
    else:
        whitePath = os.path.join(path, "white")
        blackPath = os.path.join(path, "black")
        # Generates the path for the requested piece
        def wPath(piece): return os.path.join(whitePath, piece + ".png")
        def bPath(piece): return os.path.join(blackPath, piece + ".png")
        pieceImages = {
            "p": Image.open(bPath("Pawn")).convert("RGBA"),
            "P": Image.open(wPath("Pawn")).convert("RGBA"),
            "r": Image.open(bPath("Rook")).convert("RGBA"),
            "R": Image.open(wPath("Rook")).convert("RGBA"),
            "n": Image.open(bPath("Knight")).convert("RGBA"),
            "N": Image.open(wPath("Knight")).convert("RGBA"),
            "b": Image.open(bPath("Bishop")).convert("RGBA"),
            "B": Image.open(wPath("Bishop")).convert("RGBA"),
            "q": Image.open(bPath("Queen")).convert("RGBA"),
            "Q": Image.open(wPath("Queen")).convert("RGBA"),
            "k": Image.open(bPath("King")).convert("RGBA"),
            "K": Image.open(wPath("King")).convert("RGBA")
        }
        if cache:
            pieceCache[path] = pieceImages

    def load(board):
        if f'{path}-{board.size[0]}' in resizedCache:
            return resizedCache[f'{path}-{board.size[0]}']
        else:
            pieceSize = int(board.size[0]/8)
            resized = {}
            for piece in pieceImages:
                resized[piece] = pieceImages[piece].resize(
                    (pieceSize, pieceSize))
            if cache:
                resizedCache[f'{path}-{board.size[0]}'] = resized
            return resized
    return load


arrowsCache = {}
resizedArrowsCache = {}


def loadArrowsFolder(path, cache=True):
    if path in arrowsCache:
        arrows = arrowsCache[path]
    else:
        def arrowP(name): return os.path.join(path, name + ".png")
        arrows = {
            "one": Image.open(arrowP("Knight")).convert("RGBA"),
            "up": Image.open(arrowP("Up")).convert("RGBA")
        }

    def load(board):
        if f'{path}-{board.size[0]}' in resizedArrowsCache:
            return resizedArrowsCache[f'{path}-{board.size[0]}']
        else:
            squareSize = int(board.size[0]/8)
            resized = {}
            resized["one"] = arrows["one"].resize((squareSize*3, squareSize*2))
            resized["up"] = arrows["up"].resize((squareSize, squareSize*3))
            if cache:
                resizedArrowsCache[f'{path}-{board.size[0]}'] = resized
            return resized
    return load


"""
Takes given parameters and returns a PIL
image of the resulting chess position

Parameters
----------
fen: str
    Fen string representing a position
squarelength: int
    the length of one square on the board
    resulting board will be 8 * squarelength long
pieceSet: loadPiecesFolder
    the piece set, loaded using the loadPiecesFolder function
        if coordinate[0] == "a":
            size = font.getsize(coordinate[1])
            width, height = size
            collection.append({
                "coordinate": (squareOrigin[0] - padding - width, squareOrigin[1] + squarelength/2 - height/2),
                "text": coordinate[1]
            })
        if coordinate[0] == "h":
            size = font.getsize(coordinate[1])
            width, height = size
            collection.append({
                "coordinate": (squareOrigin[0] + padding + squarelength, squareOrigin[1] + squarelength/2 - height/2),
                "text": coordinate[1]
            })
darkColor: str
    dark square color on the board
lightColor: str
    light square color on the board
flipped: boolean
    default = False
    Whether to flip to board, and render it from black's perspective

"""


def loadFontFile(path):
    def loader(size):
        if ".ttf" in path:
            return ImageFont.truetype(path, size=size)
        return ImageFont.load(path)
    return loader


def fenToImage(fen, squarelength, pieceSet, darkColor, lightColor, ArrowSet=None, Arrows=None, flipped=False, lastMove=None, coordinates=None, highlighting=None):
    board = Image.new("RGB", (squarelength * 8, squarelength * 8), lightColor)
    parsedBoard = FenParser(fen).parse()
    # Flip the list to reverse the position, and
    # render from black's POV.
    if Arrows != None:
        for index, arrowTuple in enumerate(Arrows):
            arrow = list(arrowTuple)
            if type(arrow[0]) == str:
                arrow[0] = squareToIndices(arrow[0])
            if type(arrow[1]) == str:
                arrow[1] = squareToIndices(arrow[1])
            Arrows[index] = arrow
    if lastMove != None:
        if type(lastMove["before"]) == str:
            lastMove["before"] = squareToIndices(lastMove["before"])
        if type(lastMove["after"]) == str:
            lastMove["after"] = squareToIndices(lastMove["after"])
    if highlighting != None:
        for color_pair in highlighting:
            highlighting[color_pair] = map(squareToIndices, highlighting[color_pair])
    if flipped:
        parsedBoard.reverse()
        for row in parsedBoard:
            row.reverse()
        if lastMove != None:
            lastMove["before"] = flipCoordTuple(lastMove["before"])
            lastMove["after"] = flipCoordTuple(lastMove["after"])
        if Arrows != None:
            for index, arrow in enumerate(Arrows):
                Arrows[index] = (flipCoordTuple(arrow[0]),
                                 flipCoordTuple(arrow[1]))
        if highlighting != None:
            for color_pair in highlighting:
                highlighting[color_pair] = map(flipCoordTuple, highlighting[color_pair])


    board = paintCheckerBoard(board, darkColor, lastMove, highlighting)
    paintOffset = (0, 0)
    if coordinates != None:
        board, paintOffset = paintCoordinateOverlay(
            board, coordinates, squarelength, flipped)
    pieceAndArrowOverlay = Image.new(
        "RGBA", (squarelength * 8, squarelength * 8))
    pieceAndArrowOverlay = paintAllPieces(
        pieceAndArrowOverlay, parsedBoard, pieceSet(pieceAndArrowOverlay))
    if ArrowSet != None and Arrows != None:
        pieceAndArrowOverlay = paintAllArrows(
            pieceAndArrowOverlay, Arrows, ArrowSet(pieceAndArrowOverlay))
    board.paste(pieceAndArrowOverlay, paintOffset, mask=pieceAndArrowOverlay)
    return board
