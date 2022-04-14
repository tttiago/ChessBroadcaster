import datetime
import sys
import time

import berserk


class LichessBroadcast:
    def __init__(self, token, broadcast_id, pgn_games, game_id, time_control="90+30"):
        self.token = token
        self.broadcast_id = broadcast_id
        self.game_id = game_id
        self.all_games = pgn_games
        self.pgn_game = pgn_games[game_id] + "\n\n"
        init_minutes = int(time_control.split("+")[0])
        self.clock_times = [init_minutes * 60, init_minutes * 60]
        self.increment = int(time_control.split("+")[1])
        self.num_half_moves = 0
        self.last_move_time = time.time()
        self.cur_move_time = None

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

        if self.game_id == 0:
            self.round_setup()

    @property
    def pgn_list(self):
        pgn_list = [str(game) for game in self.all_games]
        pgn_list[self.game_id] = self.pgn_game + "*"
        return pgn_list

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

    def move(self, move):
        self.num_half_moves += 1
        # If it is a white move, add a newline and the move number.
        if self.num_half_moves % 2 == 1:
            self.pgn_game += f"\n{(self.num_half_moves + 1) // 2}. "

        self.pgn_game += move + self.get_clock_update() + " "
        with open(f"./ongoing_games/game{self.game_id}.pgn", "w") as f:
            f.write(self.pgn_game)
        self.push_current_pgn()
        print("Done playing move " + str(move))

    def get_clock_update(self):
        self.cur_move_time = time.time()
        elapsed_time = self.cur_move_time - self.last_move_time
        self.last_move_time = self.cur_move_time
        what_clock = (self.num_half_moves - 1) % 2
        self.clock_times[what_clock] -= elapsed_time
        self.clock_times[what_clock] += self.increment
        clock_str = str(datetime.timedelta(seconds=self.clock_times[what_clock])).split(".")[0]
        return f" {{[%clk {clock_str}]}}"


if __name__ == "__main__":
    import glob
    import os

    token = os.environ.get("LICHESS_TOKEN")
    broadcast_id = "r9K4Vjgf"

    # End transmission.
    pgn_games = []
    for game in glob.glob("./ongoing_games/*"):
        with open(game) as f:
            pgn_games.append(f.read())
    broadcast = LichessBroadcast(token, broadcast_id, pgn_games, 0)
    broadcast.round_setup()

    # with open("initial_games.pgn") as f:
    #     pgn_games = f.read().split("\n\n\n")

    # broadcast = LichessBroadcast(token, broadcast_id, pgn_games, 0)

    # broadcast.round_setup()

    # broadcast.move("e4")
    # time.sleep(4)
    # broadcast.move("e5")
    # time.sleep(2)
    # broadcast.move("Nf3")
    # time.sleep(1)
    # broadcast.move("Nc6")
    # time.sleep(3)
    # broadcast.move("Bb5")

    # input("Press Enter after updating PGN")
    # with open("ongoing_games.pgn") as f:
    #     broadcast.pgn_game = f.read().split("\n\n\n")
    # broadcast.push_current_pgn()

    # broadcast.move("Bc5")
