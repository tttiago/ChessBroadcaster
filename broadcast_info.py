"""Information relevant for the broadcast.
Includes camera IPs and password, and Lichess token and broadcast_id."""

import os
from dataclasses import dataclass


@dataclass
class BroadcastInfo:
    broadcast_id = "TGQmC5iC"
    lichess_token = os.environ.get("LICHESS_TOKEN")

    IPs = [
        "192.168.1.78",
        "192.168.1.98",
        "192.168.1.77",
        "192.168.1.97",
        "192.168.1.107",
    ]
    camera_password = os.environ.get("CAMERA_GALITOS")
