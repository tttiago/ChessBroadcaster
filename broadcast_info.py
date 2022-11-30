"""Information relevant for the broadcast."""

import os
from dataclasses import dataclass


@dataclass
class BroadcastInfo:
    broadcast_id = "a9i3pnQP"

    IPs = [
        "192.168.1.74",
        "192.168.1.98",
        "192.168.1.72",
        "192.168.1.97",
        "192.168.1.100",
    ]
    camera_password = os.environ.get("CAMERA_GALITOS")
