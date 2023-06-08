"""Allow to use command line arguments for the camera and board numbers."""

import argparse


def create_parser(task="broadcast"):
    if task == "broadcast":
        description = "Broadcast game from a camera to Lichess."
    elif task == "calibrate":
        description = "Calibrate camera using empty board."

    parser = argparse.ArgumentParser(description=description)
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
        default=None,
        help="Number of the table to be broadcasted (starts at 1).",
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
