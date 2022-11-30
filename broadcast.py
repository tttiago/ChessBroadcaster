import io
import re

import chess
import chess.pgn
import cv2
import numpy as np

from helper import detect_state, get_square_image
from lichess_broadcast import LichessBroadcast


#import board_rendering
import requests


class Broadcast:
    def __init__(
        self, board_basics, token, broadcast_id, pgn_games, roi_mask, game_id,
        round=None
    ):
        '''
        assert token
        self.internet_broadcast = LichessBroadcast(
            token, broadcast_id, pgn_games, game_id, round=round
        )
        '''
        self.board_basics = board_basics
        self.game_id = game_id
        self.executed_moves = []
        self.played_moves = []
        self.board = chess.Board()
        self.roi_mask = roi_mask
        self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        self.knn = cv2.ml.KNearest_create()
        self.features = None
        self.labels = None

    def initialize_hog(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pieces = []
        squares = []
        for row in range(8):
            for column in range(8):
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                piece = self.board.piece_at(square)
                square_image = get_square_image(row, column, frame)
                square_image = cv2.resize(square_image, (64, 64))
                if piece:
                    pieces.append(square_image)
                else:
                    squares.append(square_image)
        pieces_hog = [self.hog.compute(piece) for piece in pieces]
        squares_hog = [self.hog.compute(square) for square in squares]
        labels_pieces = np.ones((len(pieces_hog), 1), np.int32)
        labels_squares = np.zeros((len(squares_hog), 1), np.int32)
        pieces_hog = np.array(pieces_hog)
        squares_hog = np.array(squares_hog)
        features = np.float32(np.concatenate(
            (pieces_hog, squares_hog), axis=0))
        labels = np.concatenate((labels_pieces, labels_squares), axis=0)
        self.knn.train(features, cv2.ml.ROW_SAMPLE, labels)
        self.features = features
        self.labels = labels

    def detect_state_hog(self, chessboard_image):
        chessboard_image = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
        chessboard = [
            [get_square_image(row, column, chessboard_image)
             for column in range(8)]
            for row in range(8)
        ]

        board_hog = [
            [
                self.hog.compute(cv2.resize(chessboard[row][column], (64, 64)))
                for column in range(8)
            ]
            for row in range(8)
        ]
        knn_result = []
        for row in range(8):
            knn_row = []
            for column in range(8):
                ret, result, neighbours, dist = self.knn.findNearest(
                    np.array([board_hog[row][column]]), k=3
                )
                knn_row.append(result[0][0])
            knn_result.append(knn_row)
        board_state = [[knn_result[row][column] >
                        0.5 for column in range(8)] for row in range(8)]
        return board_state

    def get_valid_move_hog(self, fgmask, frame):
        print("Hog working")
        board = [
            [self.board_basics.get_square_image(
                row, column, fgmask).mean() for column in range(8)]
            for row in range(8)
        ]
        potential_squares = []
        square_scores = {}
        for row in range(8):
            for column in range(8):
                score = board[row][column]
                if score < 10.0:
                    continue
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                potential_squares.append(square)
                square_scores[square] = score

        potential_moves = []

        board_result = self.detect_state_hog(frame)
        for move in self.board.legal_moves:
            if (move.from_square in potential_squares) and (move.to_square in potential_squares):
                # Currently only allows to promote to queen.
                if move.promotion and move.promotion != chess.QUEEN:
                    continue
                self.board.push(move)
                if self.check_state_hog(board_result):
                    self.board.pop()
                    total_score = square_scores[move.from_square] + \
                        square_scores[move.to_square]
                    potential_moves.append((total_score, move.uci()))
                else:
                    self.board.pop()
        if potential_moves:
            print("Hog!")
            return True, max(potential_moves)[1]
        else:
            return False, ""

    def is_light_change(self, frame):
        result = detect_state(frame, self.board_basics.d[0], self.roi_mask)
        result_hog = self.detect_state_hog(frame)
        state = self.check_state_for_light(result, result_hog)
        if state:
            print("Light change")
            return True
        else:
            return False

    def check_state_hog(self, result):
        for row in range(8):
            for column in range(8):
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                piece = self.board.piece_at(square)
                if piece and (not result[row][column]):
                    print("Expected piece at " + square_name)
                    return False
                if (not piece) and (result[row][column]):
                    print("Expected empty at " + square_name)
                    return False
        return True

    def check_state_for_move(self, result):
        for row in range(8):
            for column in range(8):
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                piece = self.board.piece_at(square)
                if piece and (True not in result[row][column]):
                    print("Expected piece at " + square_name)
                    return False
                if (not piece) and (False not in result[row][column]):
                    print("Expected empty at " + square_name)
                    return False
        return True

    def check_state_for_light(self, result, result_hog):
        for row in range(8):
            for column in range(8):
                if len(result[row][column]) > 1:
                    result[row][column] = [result_hog[row][column]]
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                piece = self.board.piece_at(square)
                if piece and (False in result[row][column]):
                    print(square_name)
                    return False
                if (not piece) and (True in result[row][column]):
                    print(square_name)
                    return False
        return True

    def get_valid_move_canny(self, fgmask, frame):
        print("Canny working")
        board = [
            [self.board_basics.get_square_image(
                row, column, fgmask).mean() for column in range(8)]
            for row in range(8)
        ]
        potential_squares = []
        square_scores = {}
        for row in range(8):
            for column in range(8):
                score = board[row][column]
                if score < 10.0:
                    continue
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                potential_squares.append(square)
                square_scores[square] = score

        potential_moves = []

        board_result = detect_state(
            frame, self.board_basics.d[0], self.roi_mask)
        for move in self.board.legal_moves:
            if (move.from_square in potential_squares) and (move.to_square in potential_squares):
                if move.promotion and move.promotion != chess.QUEEN:
                    continue
                self.board.push(move)
                if self.check_state_for_move(board_result):
                    self.board.pop()
                    total_score = square_scores[move.from_square] + \
                        square_scores[move.to_square]
                    potential_moves.append((total_score, move.uci()))
                else:
                    self.board.pop()
        if potential_moves:
            print("Canny!")
            return True, max(potential_moves)[1]
        else:
            return False, ""

    def register_move(self, fgmask, previous_frame, next_frame):
        (
            potential_squares,
            potential_moves,
        ) = self.board_basics.get_potential_moves(fgmask, previous_frame, next_frame, self.board)
        success, valid_move_string = self.get_valid_move(
            potential_squares, potential_moves)
        print("Valid move string: " + valid_move_string)
        if not success:
            success, valid_move_string = self.get_valid_move_canny(
                fgmask, next_frame)
            print("Valid move string 2: " + valid_move_string)
            if not success:
                success, valid_move_string = self.get_valid_move_hog(
                    fgmask, next_frame)
                print("Valid move string 3: " + valid_move_string)
            if success:
                pass
            else:
                print(self.board.fen())
                return False

        valid_move_UCI = chess.Move.from_uci(valid_move_string)

        self.played_moves.append(valid_move_UCI)
        self.executed_moves.append(self.board.san(valid_move_UCI))

        # self.internet_broadcast.move(self.executed_moves[-1])
        self.board.push(valid_move_UCI)

        if not requests.post(
            "http://127.0.0.1:5000/updateBoard",
                data=valid_move_string):
            print("ERROR: post failed")

        # board_rendering.render(self.board)

        self.learn(next_frame)
        return True

    def learn(self, frame):
        result = self.detect_state_hog(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pieces = []
        new_squares = []

        for row in range(8):
            for column in range(8):
                square_name = self.board_basics.convert_row_column_to_square_name(
                    row, column)
                square = chess.parse_square(square_name)
                piece = self.board.piece_at(square)
                if piece and (not result[row][column]):
                    print("Learning piece at " + square_name)
                    piece_hog = self.hog.compute(
                        cv2.resize(get_square_image(
                            row, column, frame), (64, 64))
                    )
                    new_pieces.append(piece_hog)
                if (not piece) and (result[row][column]):
                    print("Learning empty at " + square_name)
                    square_hog = self.hog.compute(
                        cv2.resize(get_square_image(
                            row, column, frame), (64, 64))
                    )
                    new_squares.append(square_hog)
        labels_pieces = np.ones((len(new_pieces), 1), np.int32)
        labels_squares = np.zeros((len(new_squares), 1), np.int32)
        if new_pieces:
            new_pieces = np.array(new_pieces)
            self.features = np.float32(np.concatenate(
                (self.features, new_pieces), axis=0))
            self.labels = np.concatenate((self.labels, labels_pieces), axis=0)
        if new_squares:
            new_squares = np.array(new_squares)
            self.features = np.float32(np.concatenate(
                (self.features, new_squares), axis=0))
            self.labels = np.concatenate((self.labels, labels_squares), axis=0)

        self.features = self.features[:100]
        self.labels = self.labels[:100]
        # print(self.features.shape)
        # print(self.labels.shape)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.features, cv2.ml.ROW_SAMPLE, self.labels)

    def get_valid_move(self, potential_squares, potential_moves):
        print("Potential squares")
        print(potential_squares)
        print("Potential moves")
        print(potential_moves)

        valid_move_string = ""
        for score, start, arrival in potential_moves:
            if valid_move_string:
                break

            uci_move = start + arrival
            try:
                move = chess.Move.from_uci(uci_move)
            except Exception as e:
                print(e)
                continue

            if move in self.board.legal_moves:
                valid_move_string = uci_move
            else:
                uci_move_promoted = uci_move + "q"
                promoted_move = chess.Move.from_uci(uci_move_promoted)
                if promoted_move in self.board.legal_moves:
                    valid_move_string = uci_move_promoted
                    # print("There has been a promotion")

        potential_squares = [square[1] for square in potential_squares]
        print(potential_squares)
        # Detect castling king side with white
        if (
            ("e1" in potential_squares)
            and ("h1" in potential_squares)
            and ("f1" in potential_squares)
            and ("g1" in potential_squares)
            and (chess.Move.from_uci("e1g1") in self.board.legal_moves)
        ):
            valid_move_string = "e1g1"

        # Detect castling queen side with white
        if (
            ("e1" in potential_squares)
            and ("a1" in potential_squares)
            and ("c1" in potential_squares)
            and ("d1" in potential_squares)
            and (chess.Move.from_uci("e1c1") in self.board.legal_moves)
        ):
            valid_move_string = "e1c1"

        # Detect castling king side with black
        if (
            ("e8" in potential_squares)
            and ("h8" in potential_squares)
            and ("f8" in potential_squares)
            and ("g8" in potential_squares)
            and (chess.Move.from_uci("e8g8") in self.board.legal_moves)
        ):
            valid_move_string = "e8g8"

        # Detect castling queen side with black
        if (
            ("e8" in potential_squares)
            and ("a8" in potential_squares)
            and ("c8" in potential_squares)
            and ("d8" in potential_squares)
            and (chess.Move.from_uci("e8c8") in self.board.legal_moves)
        ):
            valid_move_string = "e8c8"

        if not valid_move_string and len(potential_squares) == 2:
            for move in (
                potential_squares[0] + potential_squares[1],
                potential_squares[1] + potential_squares[0],
            ):
                if chess.Move.from_uci(move) in self.board.legal_moves:
                    valid_move_string = move
                    print("Magical function worked!")
                    break

        if valid_move_string:
            print("ssim!")
            return True, valid_move_string
        else:
            return False, valid_move_string

    def correct_moves(self):
        with open(f"./ongoing_games/game{self.game_id}.pgn") as f:
            self.internet_broadcast.pgn_game = f.read()

        self.internet_broadcast.push_current_pgn()
        print("Done updating broadcast.")
        game = chess.pgn.read_game(io.StringIO(
            self.internet_broadcast.pgn_game.split("\n\n")[-1]))
        self.board = game.board()
        self.internet_broadcast.num_half_moves = 0
        for move in game.mainline_moves():
            self.board.push(move)
            self.internet_broadcast.num_half_moves += 1

        print("Done updating board.\n")

    def correct_clocks(self, response):
        # Find 'h:mm:ss' parts.
        times = re.findall("([0-9]:[0-5][0-9]:[0-5][0-9])", response)
        if len(times) == 2:
            # Assign each time string to the respective clock.
            self.internet_broadcast.clock_times[0] = self.get_sec(times[0])
            self.internet_broadcast.clock_times[1] = self.get_sec(times[1])
            print("Clocks successfully updated.")
        else:
            print("Clock times could not be updated.")

    @staticmethod
    def get_sec(time_str):
        """Get seconds from time."""
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
