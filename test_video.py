"""Use this file to test if you can connect to your camera."""

import cv2

from broadcast_info import BroadcastInfo

CAMERA_ID = 2

# To use Tapo Cameras.
broadcast_info = BroadcastInfo()
pwd = broadcast_info.camera_password
camera_ip = broadcast_info.IPs[CAMERA_ID - 1]
RTSP_URL = f"rtsp://camera{CAMERA_ID}:{pwd}@{camera_ip}/stream1"
cap_index = RTSP_URL
cap_api = cv2.CAP_FFMPEG

# To use with an USB camera (or DroidCam).
# You may need to change the index to other (small) integer values if you have multiple cameras.
# cap_index = 0
# You may need to try different cap_api's. The default is CAP_ANY. Use CAP_V4L2 in Linux.
# cap_api = cv2.CAP_ANY
# cap_api = cv2.CAP_V4L2

video_capture = cv2.VideoCapture(cap_index, cap_api)

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
