"""Upload the pairing info to Lichess before the round starts."""

import os

from broadcast_info import BroadcastInfo
from lichess_broadcast import LichessBroadcast

token = os.environ.get("LICHESS_TOKEN")
broadcast_info = BroadcastInfo()
broadcast_id = broadcast_info.broadcast_id

# Load the games metadata from a single PGN file.
with open("initial_games.pgn") as f:
    pgn_games = f.read().split("\n\n\n")

broadcast = LichessBroadcast(token, broadcast_id, pgn_games, 0)
print("Round set up successfully.")
