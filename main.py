import math
import os
import pickle
import sys
import time
from collections import deque

import cv2
import imutils
import numpy as np

from board_basics import BoardBasics
from broadcast import Broadcast
from broadcast_fixer import BroadcastFixer
from broadcast_info import BroadcastInfo
from helper_functions import perspective_transform
from parser_helper import create_parser
from video_capture import Video_capture_thread

################################################################################

DEBUG = False

# Create needed folders if they don't exist.
if DEBUG:
    if not os.path.exists("images/"):
        os.makedirs("images/")
if not os.path.exists("ongoing_games/"):
    os.makedirs("ongoing_games/")

parser = create_parser(task="broadcast")
args = parser.parse_args()
broadcast_info = BroadcastInfo()

# Lichess Token and Broadcast ID
token = broadcast_info.lichess_token
broadcast_id = broadcast_info.broadcast_id

# Camera setup
cam_id = args.camera_index
cam_ip = broadcast_info.IPs[cam_id - 1]
cam_pwd = broadcast_info.camera_password
stream = args.stream
RTSP_URL = f"rtsp://camera{cam_id}:{cam_pwd}@{cam_ip}:554/stream{stream}"
cap_api = cv2.CAP_FFMPEG
cap_index = RTSP_URL

# n_boards, from 1 to n ----> game_id from 0 to n-1
game_id = args.game_id - 1

# Load the games metadata from a single PGN file.
if not os.path.isfile("initial_games.pgn"):
    print(
        'Make sure you create a file named "initial_games.pgn" with the games metadata.'
        '\nYou can copy and rename the "initial_games_template.pgn" file.'
    )
with open("initial_games.pgn") as f:
    pgn_games = f.read().split("\n\n\n")


############# CV constants #########
MOTION_START_THRESHOLD = 1.0
HISTORY = 100
MAX_MOVE_MEAN = 50
COUNTER_MAX_VALUE = 3
MAX_QUEUE_LENGTH = 10
####################################

move_fgbg = cv2.createBackgroundSubtractorKNN()
motion_fgbg = cv2.createBackgroundSubtractorKNN(history=HISTORY)

filename = f"./constants/constants{cam_id}.bin"
with open(filename, "rb") as infile:
    corners, side_view_compensation, rotation_count, roi_mask = pickle.load(infile)

board_basics = BoardBasics(side_view_compensation, rotation_count)
broadcast = Broadcast(board_basics, token, broadcast_id, pgn_games, roi_mask, game_id)

video_capture_thread = Video_capture_thread()
video_capture_thread.daemon = True
video_capture_thread.capture = cv2.VideoCapture(cap_index, cap_api)
video_capture_thread.start()

# Detect keypresses for the correction of moves and clock times.
broadcast_fixer = BroadcastFixer(broadcast)
listener = broadcast_fixer.listener
listener.start()

# pts1 is the list of coordinates of the four chessboard corners (order: a1, h1, a8, h8; format: (x, y))
pts1 = np.float32(
    [
        list(corners[0][0]),
        list(corners[8][0]),
        list(corners[0][8]),
        list(corners[8][8]),
    ]
)
# print(pts1)


def waitUntilMotionCompletes():
    """Halt program until motion over the board is completed."""
    counter = 0
    while counter < COUNTER_MAX_VALUE:
        frame = video_capture_thread.get_frame()
        # frame = corcamera(frame)
        frame = perspective_transform(frame, pts1)
        fgmask = motion_fgbg.apply(frame)
        ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        mean = fgmask.mean()
        if mean < MOTION_START_THRESHOLD:
            counter += 1
        else:
            counter = 0


def stabilize_background_subtractors():
    best_mean = float("inf")
    counter = 0
    while counter < COUNTER_MAX_VALUE:
        frame = video_capture_thread.get_frame()
        # frame = corcamera(frame)
        frame = perspective_transform(frame, pts1)
        move_fgbg.apply(frame)
        fgmask = motion_fgbg.apply(frame, learningRate=0.1)
        ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        mean = fgmask.mean()
        if mean >= best_mean:
            counter += 1
        else:
            best_mean = mean
            counter = 0

    best_mean = float("inf")
    counter = 0
    while counter < COUNTER_MAX_VALUE:
        frame = video_capture_thread.get_frame()
        # frame = corcamera(frame)
        frame = perspective_transform(frame, pts1)
        fgmask = move_fgbg.apply(frame, learningRate=0.1)
        ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        motion_fgbg.apply(frame)
        mean = fgmask.mean()
        if mean >= best_mean:
            counter += 1
        else:
            best_mean = mean
            counter = 0

    return frame


################### NEW FUNCTIONS FROM DUARTE ###########################


def corcamera(img):
    mtx = np.matrix(
        [[786.098795, 0.0, 704.94428], [0.0, 1145.3459778, 415.1591336], [0.0, 0.0, 1.0]]
    )
    dist = np.matrix([-0.4021267, 0.1650988, -0.00039044, -0.0003982488, -0.0086298372])
    h, w = img.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    img = img[y : y + h, x : x + w]

    return dst


def getboardloc_complete(mask, prevxsquare):
    xyratio = 1.115
    if prevxsquare == 0:
        # xsquare = 21
        xsquare = 39
        minxsquare = 39
        maxxsquare = 43
        # minxsquare = 21
        # maxxsquare = 46
        # minangle = -40
        # maxangle = 40
        minangle = -20
        maxangle = 20

        div = 2
    else:
        minxsquare = prevxsquare - 4
        maxxsquare = prevxsquare + 4
        xsquare = minxsquare
        minangle = -20
        maxangle = 20
        div = 1
    ysquare = xsquare * xyratio
    xx = xsquare * 8
    yy = ysquare * 8
    template = gen_mask(xsquare, ysquare, 0)
    method = cv2.TM_CCOEFF_NORMED
    rotated = imutils.rotate_bound(mask, -20)
    res = cv2.matchTemplate(template, rotated, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    maxmax = max_val
    top_left = max_loc
    anglemax = -20
    xsquaremax = xsquare
    for xsquare in range(minxsquare, maxxsquare, 1):
        ysquare = xsquare * xyratio
        xx = xsquare * 8
        yy = ysquare * 8
        template = gen_mask(xsquare, ysquare, 0)

        for angle in range(minangle, maxangle, 1):
            angle = angle / div
            rotated = imutils.rotate_bound(mask, angle)
            res = cv2.matchTemplate(template, rotated, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > maxmax:
                maxmax = max_val
                # print(maxmax)
                anglemax = angle
                top_left = max_loc
                xsquaremax = xsquare

    return anglemax, top_left, xsquaremax, maxmax


def getboardloc_normal(mask, anglemax, xsquare):
    method = cv2.TM_CCOEFF_NORMED
    rotated = imutils.rotate_bound(mask, anglemax)
    xyratio = 1.115
    ysquare = xsquare * xyratio
    xx = xsquare * 8
    yy = ysquare * 8
    template = gen_mask(xsquare, ysquare, 0)
    res = cv2.matchTemplate(template, rotated, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    return top_left, max_val


def gen_mask(sizex, sizey, topleftcolour):
    # Use a breakpoint in the code line below to debug your script.
    width = round(sizex * 8)
    height = round(sizey * 8)
    mask = np.zeros((height, width), np.uint8)
    # print(mask)
    for n in range(8):
        for m in range(8):
            mask[
                round(n * sizey) : round((n + 1) * sizey),
                round(m * sizex) : round((m + 1) * sizex),
            ] = (topleftcolour != ((n + m) % 2)) * 255
    return mask


def rotatepoint(pt, alpha, img):
    h, w = img.shape[:2]
    alpha = -alpha * math.pi / 180
    yr = w * math.sin(alpha) + h * math.cos(alpha) - pt[1]
    xr = pt[0]

    x0 = math.sin(alpha) * h
    y2 = math.tan(alpha) * (xr - x0)
    y1 = yr - y2
    y = math.cos(alpha) * y1
    x2 = math.tan(alpha) * y
    x1 = y2 / math.sin(alpha)
    x = x1 + x2
    ptout = (x, h - y)
    return ptout


######################## DUARTE VAR INIT ###############################

xsquaremax = 0
trigger = True


#############################  Main code ################################

previous_frame = stabilize_background_subtractors()
board_basics.initialize_ssim(previous_frame)
broadcast.initialize_hog(previous_frame)
previous_frame_queue = deque(maxlen=MAX_QUEUE_LENGTH)
previous_frame_queue.append(previous_frame)

while not broadcast.board.is_game_over():
    sys.stdout.flush()
    frame = video_capture_thread.get_frame()
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)

    # frame = corcamera(frame)

    # cv2.imshow("corrected_frame", frame)
    # cv2.waitKey(0)
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgHSV, (7, 7), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2
    )
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    method = cv2.TM_CCOEFF_NORMED

    if trigger:
        anglemax, top_left, xsquaremax, max_val_hist = getboardloc_complete(mask, xsquaremax)
        print("Finish getboardloccomplete")
        xsquare = xsquaremax
        print(xsquaremax)
        ysquare = 1.115 * xsquare
        xx = xsquare * 8
        yy = ysquare * 8
        pts1 = np.float32(
            [
                list(rotatepoint(top_left, anglemax, frame)),
                list(rotatepoint((top_left[0], top_left[1] + yy), anglemax, frame)),
                list(rotatepoint((top_left[0] + xx, top_left[1]), anglemax, frame)),
                list(rotatepoint((top_left[0] + xx, top_left[1] + yy), anglemax, frame)),
            ]
        )
        trigger = False
    else:
        ...
        # top_left, max_val = getboardloc_normal(mask, anglemax, xsquaremax)
        # pts1 = np.float32(
        #     [
        #         list(rotatepoint(top_left, anglemax, frame)),
        #         list(rotatepoint((top_left[0], top_left[1] + yy), anglemax, frame)),
        #         list(rotatepoint((top_left[0] + xx, top_left[1]), anglemax, frame)),
        #         list(rotatepoint((top_left[0] + xx, top_left[1] + yy), anglemax, frame)),
        #     ]
        # )
        # max_val_hist = max_val_hist * 0.95 + max_val * 0.05
        # # print(max_val_hist)
        # if max_val_hist < 0.5 or max_val < 0.25:
        #     trigger = True

    # print(pts1)
    frame = perspective_transform(frame, pts1)
    cv2.imshow("frame_depois", frame)
    cv2.waitKey(1)

    fgmask = motion_fgbg.apply(frame)
    cv2.imshow("fgmask_antes", fgmask)
    cv2.waitKey(1)

    ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    cv2.imshow("fgmask", fgmask)
    cv2.waitKey(1)
    mean = fgmask.mean()
    if mean > MOTION_START_THRESHOLD:
        # cv2.imwrite("motion.jpg", fgmask)
        waitUntilMotionCompletes()
        frame = video_capture_thread.get_frame()
        frame = perspective_transform(frame, pts1)
        fgmask = move_fgbg.apply(frame, learningRate=0.0)
        if fgmask.mean() >= 10.0:
            ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        # print("Move mean " + str(fgmask.mean()))
        if fgmask.mean() >= MAX_MOVE_MEAN:
            fgmask = np.zeros(fgmask.shape, dtype=np.uint8)
        motion_fgbg.apply(frame)
        move_fgbg.apply(frame, learningRate=1.0)
        last_frame = stabilize_background_subtractors()
        previous_frame = previous_frame_queue[0]

        if not broadcast.is_light_change(last_frame):
            # Try to get two moves after each board movement, to detect quick replies.
            for _ in range(2):
                if not broadcast.register_move(fgmask, previous_frame, last_frame):
                    continue
                if DEBUG:
                    cv2.imwrite(
                        "images/" + broadcast.executed_moves[-1] + " frame.jpg",
                        last_frame,
                    )
                    cv2.imwrite(
                        "images/" + broadcast.executed_moves[-1] + " mask.jpg",
                        fgmask,
                    )
                    cv2.imwrite(
                        "images/" + broadcast.executed_moves[-1] + " background.jpg",
                        previous_frame,
                    )

        previous_frame_queue = deque(maxlen=MAX_QUEUE_LENGTH)
        previous_frame_queue.append(last_frame)
    else:
        move_fgbg.apply(frame)
        previous_frame_queue.append(frame)
cv2.destroyAllWindows()
time.sleep(2)
