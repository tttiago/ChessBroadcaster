# Broadcast chess games played in a real board to Lichess

Program that enables you to broadcast chess games played in a real chess board to Lichess.  
Using computer vision it will detect the moves made on chess board. It will also try to estimate the clock times.

Based on the work of Alper Karayaman and Frank Groeneveld. See https://github.com/karayaman/Play-online-chess-with-real-chess-board

## Setup

1. Place your camera near to your chessboard so that all of the squares and pieces can be clearly seen by it. Preferably, it should be above the chess board.

2. Remove all pieces from your chess board.

3. Run "board_calibration.py"

4. Check that corners of your chess board are correctly detected by "board_calibration.py" and press key "q" to save detected chess board corners. You don't need to manually select chess board corners, it should be automatically detected by the program. The square covered by points (0,0), (0,1),(1,0) and (1,1) should be a8. You can rotate the image by pressing key "r" to adjust that. Example chess board detection result:

   ![](./calibrated_board.jpg)

## Usage

1. Place pieces of chess board to their starting position.
2. Run "main.py"
3. Make the moves in the real board.
4. Enjoy!


## Required libraries

- opencv-python
- python-chess
- numpy
- scikit-image
- berserk
- pyinput