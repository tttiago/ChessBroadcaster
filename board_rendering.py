import chess
import chess.svg
from chessboard import display

game_board = display.start()


def render(board):
    display.update(board.fen(), game_board)
