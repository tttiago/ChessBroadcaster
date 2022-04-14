import argparse
import os


def create_parser():
    parser = argparse.ArgumentParser(description="Calibrate camera using empty board.")
    parser.add_argument(
        "-c",
        "--camera-index",
        metavar="CI",
        type=int,
        nargs="?",
        default=1,
        help="Index of the camera to be used (starts at 1).",
    )
    parser.add_argument(
        "-g",
        "--game-id",
        metavar="GI",
        type=int,
        nargs="?",
        default=0,
        help="Index of the game to be used (starts at 0).",
    )
    parser.add_argument(
        "-s",
        "--stream",
        metavar="STREAM",
        type=int,
        nargs="?",
        default=2,
        help="Index of the stream to be used (1 is 720p, 2 is 360p).",
    )
    return parser


class CameraInfo:
    def __init__(self):
        self.IPs = ["192.168.0.122", "192.168.0.123"]
        self.password = os.environ.get("CAMERA_GALITOS")
