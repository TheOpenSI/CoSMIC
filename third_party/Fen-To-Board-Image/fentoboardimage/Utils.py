def indicesToSquare(indices, flipped=False):
    if flipped:
        return f"{chr((7 - indices[0]) + 97)}{ indices[1] + 1}"
    return f"{chr(indices[0] + 97)}{ 7 - indices[1] + 1}"


def squareToIndices(square):
    """
    Converts a square string to a tuple of indices
    """
    return (ord(square[0]) - 97, 8 - int(square[1]))


def flipCoordTuple(coord):
    """
    Flips a tuple of coordinates
    """
    return (7 - coord[0], 7 - coord[1])


def flippedCheck(flipped, checkChar, c1, c2):
    if not flipped and checkChar == c1:
        return True
    if flipped and checkChar == c2:
        return True
    return False
