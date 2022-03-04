import sys

import berserk
import chess
import chess.pgn


class LichessBroadcast:
    def __init__(self, token, broadcast_id, pgn_games):
        self.token = token
        self.broadcast_id = broadcast_id
        self.pgn_games = pgn_games

        session = berserk.TokenSession(self.token)
        self.client = berserk.Client(session)

        try:
            broadcast = self.client.broadcasts.get(self.broadcast_id)
            b_name = broadcast["tour"]["name"]
            print(f"Found broadcast: {b_name}")

        except berserk.exceptions.ResponseError:
            print("No broadcast found.")
            sys.exit(0)

        self.broadcast = broadcast["tour"]
        self.round_id = broadcast["rounds"][-1]["id"]

        self.round_setup()

    @property
    def pgn_list(self):
        return [str(game) for game in self.pgn_games]

    def round_setup(self):
        self.client.broadcasts.push_pgn_update(
            self.round_id, slug="round", pgn_games=self.pgn_list
        )

    def move(self, move, game_id=0):
        self.pgn_games[game_id].end().add_variation(move)

        try:
            self.client.broadcasts.push_pgn_update(
                self.round_id, slug="round", pgn_games=self.pgn_list
            )
        except:
            session = berserk.TokenSession(self.token)
            self.client = berserk.Client(session)
            self.client.broadcasts.push_pgn_update(
                self.round_id, slug="round", pgn_games=self.pgn_list
            )
            print("Reconnected to Lichess.")

        print("Done playing move " + str(move))


if __name__ == "__main__":
    token = "lip_v9e3Ie7aWgCmTaRoO9Q2"
    broadcast_id = "r9K4Vjgf"

    with open("example_game.pgn") as f:
        pgn = chess.pgn.read_game(f)
    pgn_games = [pgn]

    broadcast = LichessBroadcast(token, broadcast_id, pgn_games)

    broadcast.round_setup()

    broadcast.move(chess.Move.from_uci("e2e4"))
    broadcast.move(chess.Move.from_uci("e7e5"))
    broadcast.move(chess.Move.from_uci("g1f3"))
    broadcast.move(chess.Move.from_uci("b8c6"))
    broadcast.move(chess.Move.from_uci("f1b5"))
