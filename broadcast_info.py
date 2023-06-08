"""Information relevant for the broadcast.
Includes camera IPs and password, and Lichess token and broadcast_id."""

import os
from dataclasses import dataclass


@dataclass
class BroadcastInfo:
    broadcast_id = "FcGvKmME/"  # Festival
    # broadcast_id = "TGQmC5iC" # test
    lichess_token = os.environ.get("LICHESS_TOKEN")

    IPs = [
        "192.168.1.144",
        "192.168.1.152",
        "192.168.1.151",
        "192.168.1.149",
        "192.168.1.145",
        "192.168.1.147",
        "192.168.1.148",
    ]
    camera_password = os.environ.get("CAMERA_GALITOS")
