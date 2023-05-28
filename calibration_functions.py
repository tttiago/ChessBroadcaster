import math

import cv2
import imutils
import numpy as np

XY_RATIO = 1.115


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


def getboardloc_complete(mask, prev_xsquare_size):
    """Find the angle and square size of the chess board.
    Also returns the location of the corner closest to the top-left
    and the highest convolution value."""
    if not prev_xsquare_size:
        x_square_size = 45
        min_xsquare_size = 45
        max_xsquare_size = 60
        max_angle = 20
        angle_step = 0.5
    else:
        min_xsquare_size = prev_xsquare_size - 4
        max_xsquare_size = prev_xsquare_size + 4
        x_square_size = min_xsquare_size
        max_angle = 20
        angle_step = 1
    ysquare_size = x_square_size * XY_RATIO

    template = gen_mask(x_square_size, ysquare_size, 0)
    method = cv2.TM_CCOEFF_NORMED
    rotated = imutils.rotate_bound(mask, -20)
    res = cv2.matchTemplate(template, rotated, method)
    # TODO: Ask Duarte what best_max represents.
    _, best_max, _, top_left = cv2.minMaxLoc(res)

    best_angle = -max_angle
    best_xsquare_size = x_square_size
    for x_square_size in range(min_xsquare_size, max_xsquare_size):
        ysquare_size = x_square_size * XY_RATIO
        template = gen_mask(x_square_size, ysquare_size, 0)

        for angle in np.linspace(-max_angle, max_angle, int(2 * max_angle / angle_step) + 1):
            rotated = imutils.rotate_bound(mask, angle)
            res = cv2.matchTemplate(template, rotated, method)
            _, cur_max_val, _, max_loc = cv2.minMaxLoc(res)
            if cur_max_val > best_max:
                best_max = cur_max_val
                best_angle = angle
                top_left = max_loc
                best_xsquare_size = x_square_size

    return best_angle, top_left, best_xsquare_size, best_max


def getboardloc_normal(mask, anglemax, xsquare_size):
    # TODO: Ask Duarte what's the point of this function.
    method = cv2.TM_CCOEFF_NORMED
    rotated = imutils.rotate_bound(mask, anglemax)
    ysquare_size = xsquare_size * XY_RATIO
    template = gen_mask(xsquare_size, ysquare_size, 0)
    res = cv2.matchTemplate(template, rotated, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    return top_left, max_val


def gen_mask(xquare_size, ysquare_size, topleft_colour):
    # TODO: Ask Duarte what this does and what is topleft_colour.
    width = round(xquare_size * 8)
    height = round(ysquare_size * 8)
    mask = np.zeros((height, width), np.uint8)
    # print(mask)
    for n in range(8):
        for m in range(8):
            mask[
                round(n * ysquare_size) : round((n + 1) * ysquare_size),
                round(m * xquare_size) : round((m + 1) * xquare_size),
            ] = (topleft_colour != ((n + m) % 2)) * 255
    return mask


def rotate_point(point, angle, img):
    """Returns the coordinates of a point rotated by `angle` in relation to the image center."""
    if angle == 0:
        return point

    h, w = img.shape[:2]

    if angle <= 0:
        angle = abs(angle * math.pi / 180)
        yr = w * math.sin(angle) + h * math.cos(angle) - point[1]
        xr = point[0]
        x0 = math.sin(angle) * h
        y2 = math.tan(angle) * (xr - x0)
        y1 = yr - y2
        y = math.cos(angle) * y1
        x2 = math.tan(angle) * y
        x1 = y2 / math.sin(angle)
        x = x1 + x2
    else:
        angle = abs(angle * math.pi / 180)
        yr = w * math.sin(angle) + h * math.cos(angle) - point[1]
        xr = point[0]
        y2 = w * math.sin(angle) - xr * math.tan(angle)
        y1 = yr - y2
        y = math.cos(angle) * y1
        x2 = y2 / math.sin(angle)
        x1 = y1 * math.sin(angle)
        x = w - x1 - x2

    rotated_point = (x, h - y)
    return rotated_point


def get_pts1(top_left, best_angle, xsquare_size, frame):
    """Create list of board corner coordinates
    using the coordinates of the top left corner, the angle of rotation and the square size."""
    ysquare_size = XY_RATIO * xsquare_size
    board_width = xsquare_size * 8
    board_height = ysquare_size * 8
    points = np.float32(
        [
            list(
                rotate_point(
                    (top_left[0] + board_width, top_left[1] + board_height), best_angle, frame
                )
            ),
            list(rotate_point((top_left[0] + board_width, top_left[1]), best_angle, frame)),
            list(rotate_point((top_left[0], top_left[1] + board_height), best_angle, frame)),
            list(rotate_point(top_left, best_angle, frame)),
        ]
    )

    return points
