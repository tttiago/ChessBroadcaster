from dataclasses import dataclass
import cv2

use_usb = True
use_v4l2 = True


@dataclass
class CameraInfo:
    if not use_usb:
        # To use Tapo Cameras.
        broadcast_info = BroadcastInfo()
        pwd = broadcast_info.camera_password
        camera_ip = broadcast_info.IPs[4]
        RTSP_URL = f"rtsp://camera5:{pwd}@{camera_ip}/stream1"
        cap_index = RTSP_URL
        cap_api = cv2.CAP_FFMPEG
    else:
        # To use with an USB camera (or DroidCam).
        # You may need to change the index to other (small) integer values if you have multiple cameras.
        cap_index = 2
        # You may need to try different cap_api's. The default is CAP_ANY. Use CAP_V4L2 in Linux.
        if not use_v4l2:
            cap_api = cv2.CAP_ANY
        else:
            cap_api = cv2.CAP_V4L2

    video_capture = cv2.VideoCapture(cap_index, cap_api)
