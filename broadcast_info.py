"""Information relevant for the broadcast."""

import os
from dataclasses import dataclass


@dataclass
class BroadcastInfo:
    broadcast_id = "DZIXQnPc"

    IPs = [
        "192.168.0.122",
        "192.168.0.123",
        "192.168.0.107",
        "192.168.0.108",
        "192.168.0.109",
    ]
    camera_password = os.environ.get("CAMERA_GALITOS")
