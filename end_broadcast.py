"""Upload the final PGNs to Lichess to mark the round as finished."""

import glob
import os

from lichess_broadcast import LichessBroadcast

token = os.environ.get("LICHESS_TOKEN")
broadcast_id = "DZIXQnPc"

pgn_games = []
for game in glob.glob("./ongoing_games/*"):
    with open(game) as f:
        pgn_games.append(f.read())
broadcast = LichessBroadcast(token, broadcast_id, pgn_games, 0)
