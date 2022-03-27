import re
import sys

import berserk


class LichessBroadcast:
    def __init__(self, token, broadcast_id, pgn_games):
        self.token = token
        self.broadcast_id = broadcast_id
        self.pgn_games = [game + "\n\n" for game in pgn_games]

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

    def push_current_pgn(self):
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

    def move(self, move, game_id=0):
        self.pgn_games[game_id] += move + " "
        with open("ongoing_games.pgn", "w") as f:
            f.write("\n\n\n".join([pgn_game for pgn_game in self.pgn_games]))
        self.push_current_pgn()
        print("Done playing move " + str(move))


if __name__ == "__main__":
    import os

    token = os.environ.get("LICHESS_TOKEN")
    broadcast_id = "r9K4Vjgf"

    with open("initial_game.pgn") as f:
        pgn = f.read().split("\n\n\n")
    pgn_games = pgn

    broadcast = LichessBroadcast(token, broadcast_id, pgn_games)

    broadcast.round_setup()

    broadcast.move("e4")
    broadcast.move("e5")
    broadcast.move("Nf3")
    broadcast.move("Nc6")
    broadcast.move("Bb5")

    input("Press Enter after updating PGN")
    with open("ongoing_games.pgn") as f:
        broadcast.pgn_games = f.read().split("\n\n\n")
    broadcast.push_current_pgn()

    broadcast.move("Bc5")