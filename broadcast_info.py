"""Information relevant for the broadcast."""

import os
from dataclasses import dataclass


@dataclass
class BroadcastInfo:
    broadcast_id = "TNmpREIH"

    IPs = [
        "192.168.1.78",
        "192.168.1.98",
        "192.168.1.77",
        "192.168.1.97",
        "192.168.1.107",
    ]
    camera_password = os.environ.get("CAMERA_GALITOS")
