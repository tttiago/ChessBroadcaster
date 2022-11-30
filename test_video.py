"""Use this file to test if you can connect to your camera."""

import cv2

from broadcast_info import BroadcastInfo

from camera_info import CameraInfo
video_capture = CameraInfo.video_capture

if not video_capture.isOpened():
    raise Exception("Could not open video device")

# Set properties. Each returns === True on success (i.e. correct resolution)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = video_capture.read()
    cv2.imshow("Stream", frame)

    if cv2.waitKey(1) == 27:
        break
