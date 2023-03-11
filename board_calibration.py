"""Calibrate a single camera using an empty board."""

import os
import pickle
import sys
import tkinter as tk
from math import inf
from tkinter import messagebox

import cv2
import numpy as np

from broadcast_info import BroadcastInfo
from helper_functions import (
    edge_detection,
    euclidean_distance,
    mark_corners,
    perspective_transform,
    rotateMatrix,
)
from parser_helper import create_parser

DEBUG = False
SHOW_INFO = True

# Create needed folders if they don't exist.
if DEBUG:
    if not os.path.exists("images/"):
        os.makedirs("images/")
if not os.path.exists("constants/"):
    os.makedirs("constants/")

parser = create_parser(task="calibrate")
args = parser.parse_args()

broadcast_info = BroadcastInfo()
cam_id = args.camera_index
cam_ip = broadcast_info.IPs[cam_id - 1]
cam_pwd = broadcast_info.camera_password
stream = args.stream
RTSP_URL = f"rtsp://camera{cam_id}:{cam_pwd}@{cam_ip}:554/stream{stream}"
cap_api = cv2.CAP_FFMPEG

if SHOW_INFO:
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Board Calibration",
        "Board calibration will start. It should detect corners of the chess"
        + 'board almost immediately. If it does not, you should press key "q"'
        + "to stop board calibration and change webcam/board position.",
    )

cap = cv2.VideoCapture(RTSP_URL, cap_api)
# cap = cv2.VideoCapture(cap_index, cap_api)

if not cap.isOpened():
    print("Couldn't open your webcam. Please check your webcam connection.")
    sys.exit(0)
board_dimensions = (7, 7)

for _ in range(10):
    ret, frame = cap.read()
    if ret == False:
        print("Error reading frame. Please check your webcam connection.")
        continue

while True:
    ret, frame = cap.read()
    if ret == False:
        print("Error reading frame. Please check your webcam connection.")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray, patternSize=board_dimensions)

    if retval:
        if SHOW_INFO:
            messagebox.showinfo(
                "Chess Board Detected",
                "Please check that corners of your chess board are correctly"
                + " detected. The square covered by points (0,0), (0,1),(1,0) "
                + "and (1,1) should be a8. You can rotate the image by "
                + 'pressing key "r" to adjust that. Press key "q" to save '
                + "detected chess board corners and finish board calibration.",
            )
            root.destroy()
        if corners[0][0][0] > corners[-1][0][0]:  # corners returned in reverse order
            corners = corners[::-1]
        minX, maxX, minY, maxY = inf, -inf, inf, -inf
        augmented_corners = []
        row = []
        for i in range(6):
            corner1 = corners[i]
            corner2 = corners[i + 8]
            x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
            y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
            row.append((x, y))

        for i in range(4, 7):
            corner1 = corners[i]
            corner2 = corners[i + 6]
            x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
            y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
            row.append((x, y))

        augmented_corners.append(row)

        for i in range(7):
            row = []
            corner1 = corners[i * 7]
            corner2 = corners[i * 7 + 1]
            x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
            y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
            row.append((x, y))

            for corner in corners[i * 7 : (i + 1) * 7]:
                x = corner[0][0]
                y = corner[0][1]
                row.append((x, y))

            corner1 = corners[i * 7 + 6]
            corner2 = corners[i * 7 + 5]
            x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
            y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
            row.append((x, y))
            augmented_corners.append(row)

        row = []
        for i in range(6):
            corner1 = corners[42 + i]
            corner2 = corners[42 + i - 6]
            x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
            y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
            row.append((x, y))

        for i in range(4, 7):
            corner1 = corners[42 + i]
            corner2 = corners[42 + i - 8]
            x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
            y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
            row.append((x, y))

        augmented_corners.append(row)

        while (
            augmented_corners[0][0][0] > augmented_corners[8][8][0]
            or augmented_corners[0][0][1] > augmented_corners[8][8][1]
        ):
            rotateMatrix(augmented_corners)

        pts1 = np.float32(
            [
                list(augmented_corners[0][0]),
                list(augmented_corners[8][0]),
                list(augmented_corners[0][8]),
                list(augmented_corners[8][8]),
            ]
        )

        empty_board = perspective_transform(frame, pts1)
        edges = edge_detection(empty_board)
        if DEBUG:
            cv2.imshow("edge", edges)
            cv2.waitKey(0)
        kernel = np.ones((7, 7), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        roi_mask = cv2.bitwise_not(edges)
        if DEBUG:
            cv2.imshow("edge2", edges)
            cv2.waitKey(0)
            cv2.imshow("roi", roi_mask)
            cv2.waitKey(0)
            roi_mask[:7, :] = 0
        roi_mask[:, :7] = 0
        roi_mask[-7:, :] = 0
        roi_mask[:, -7:] = 0
        if DEBUG:
            cv2.imshow("roi2", roi_mask)
            cv2.waitKey(0)
            cv2.imwrite("images/empty_board.jpg", empty_board)

        rotation_count = 0
        while True:
            cv2.imshow(
                "frame",
                mark_corners(frame.copy(), augmented_corners, rotation_count),
            )
            response = cv2.waitKey(0)
            if response & 0xFF == ord("r"):
                rotation_count += 1
                rotation_count %= 4
            elif response & 0xFF == ord("q"):
                cv2.imwrite(
                    "images/calibrated_board.jpg",
                    mark_corners(frame.copy(), augmented_corners, rotation_count),
                )
                break
        break

    cv2.imshow("frame", frame)
    if cv2.waitKey(3) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

first_row = euclidean_distance(augmented_corners[1][1], augmented_corners[1][7])
last_row = euclidean_distance(augmented_corners[7][1], augmented_corners[7][7])
first_column = euclidean_distance(augmented_corners[1][1], augmented_corners[7][1])
last_column = euclidean_distance(augmented_corners[1][7], augmented_corners[7][7])

if abs(first_row - last_row) >= abs(first_column - last_column):
    if first_row >= last_row:
        side_view_compensation = (1, 0)
    else:
        side_view_compensation = (-1, 0)
else:
    if first_column >= last_column:
        side_view_compensation = (0, -1)
    else:
        side_view_compensation = (0, 1)

print("Side view compensation" + str(side_view_compensation))
print("Rotation count " + str(rotation_count))
filename = f"./constants/constants{cam_id}.bin"
with open(filename, "wb") as outfile:
    pickle.dump(
        [augmented_corners, side_view_compensation, rotation_count, roi_mask],
        outfile,
    )
