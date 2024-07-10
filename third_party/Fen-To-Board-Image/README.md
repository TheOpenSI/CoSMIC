

<div align="center">
  <img src="https://raw.githubusercontent.com/reedkrawiec/fenToBoardImage/main/documentation/logo.png" />
</div>

# About

fentoboardimage takes a Fen string representing a Chess position, and renders a PIL image of the resulting position.

# Examples

Examples can be found under the `examples` folder in this repository. 
Further examples can be found within `tests/test.py`

###  You can customize:
- the size and color of the board
- piece sprites
- black or white perspective
- Board highlighting for last move
- Arrows

# Installation

Install the package using pip
```
$ pip install fentoboardimage
```

Then import the fenToImage and loadPiecesFolder functions and use them as follows:
```
from fentoboardimage import fenToImage, loadPiecesFolder

boardImage = fenToImage(
	fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
	squarelength=100,
	pieceSet=loadPiecesFolder("./pieces"),
	darkColor="#D18B47",
	lightColor="#FFCE9E"
)
```

In order to load a piece set, the pieces must follow this file structure, and must be a .png:
```
-piece_set_name
  -white
    - Knight.png
    - Rook.png
    - Pawn.png
    - Bishop.png
    - Queen.png
    - King.png
  -black
    - Knight.png
    - Rook.png
    - Pawn.png
    - Bishop.png
    - Queen.png
    - King.png
```


In order to load an arrow set, the images must follow this file structure, and must be a .png:
```
- arrow_set_name
  - Knight.png
  - Up.png
```

- `Knight.png` is a 3:2 aspect ratio .png image containing an arrow pointing from the bottom right square to the top left square in the image. This image is internally rotated to correctly point at the target square from the origin
- `Up.png` is a 1:3 aspect ratio image depicting an upward facing arrow point from the bottom most square to the top most square. This image is internally segmented and used to construct arrows in the ordinate and diagonal direction.

# Usage

The `fenToImage` function has these parameters:

- fen: `str`

	  Fen string representing a position

- squarelength: `int`

	  The length of one square on the board. Resulting board will be 8 * squarelength long

- pieceSet: `loadPiecesFolder`

	  the piece set, loaded using the `loadPiecesFolder` function

	- The loadPiecesFolder function has two parameters:

       path: `str`

          Loads piece set located at the path provided.

      cache: `boolean`

          Whether to internally cache the piece pngs to avoid reloading

- darkColor: `str`

	  dark square color on the board

- lightColor: `str`

	  light square color on the board

- flipped: `boolean` default = False

	  Whether to flip to board, and render it from black's perspective

- ArrowSet: `loadArrowsFolder()`

	  the arrow set, loaded using the `loadArrowsFolder` function

	- The loadArrowsFolder function has two parameters:

      - path: `str`

	        Loads arrow set located at the path provided.

	  - cache: `boolean`

	        Whether to internally cache the arrow pngs to avoid reloading

- Arrows: `list[(str,str)]`

	  A list of tuples containing coordinates to place arrows. In the format
	  of (start, end) using standard chess notation for the squares.
- lastMove: `dict`
	
	  A dictionary containing the fields `before`, `after`, `darkColor` and `lightColor`. 
	  `before` and `after` using standard chess notation for the squares, and `darkColor`
	  and `lightColor` should be hex strings.

- coordinates: `dict`
     
      A dictionary containing the fields "font", "size", "darkColor", "lightColor", "positionFn",
      "padding", and "outsideBoardColor"
	   
	 - font: `loadFontFile()`
       
           The font file to use for coordinates, load using the loadFontFile function, with the
           file path to the ttf file or bitmap font file as its sole argument. 
   - size: int

         The font size for the coordinate
   - darkColor: `str`
    
         The color of the coordinate for dark squares
   -  lightColor: `str`
        
          The color of the coordinate for light squares
   -  posititionFn: `CoordinatePositionFn["outBorder"]` | `CoordinatePositionFn["innerBorder"]` | `CoordinatePositionFn["everySquare"]` | `function`
   
          The function used to determine what characters to draw and where for each square
          on the board. Can either be one of the three predefined functions from the
          CoordinatePositionFn dict, or a custom function. See the definitions for the
          builtin functions in "fentoboardimage/Coordinates.py" for examples to follow when
          writing a custom position fn.
   - padding: `int`
   
		 The padding between the square's edge and the resulting placement
		 of the coordinate character 
   - outsideBorderColor: `str` optional
   
         What color to paint the background if the coordinates end up outside the board
    

- highlighting: `dict`
  
  		A dictionary of highlighting color => squares to highlight. The keys of this object have
  		are either a hex color, or a tuple of (light_square_color, dark_square_color). The values
		of the dictionary must be a list of square coordinates. A hex color key will apply the same
  		color to every listed square, while a tuple key will conditionally apply either light_square_color
  		or dark_square_color depending on if the coordinate is a light or dark square.
  
  Example:
  
      highlighting={
         ("#ff0000", "#702963"): ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
         ("#00ff00", "#2e7d32"): ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
         "#0000ff": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
      }

  
  

# Dependencies
- [Pillow](https://pypi.org/project/Pillow/)

