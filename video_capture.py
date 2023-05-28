"""Capture video stream in a different thread."""

from collections import deque
from threading import Thread


class Video_capture_thread(Thread):
    def __init__(self, *args, **kwargs):
        super(Video_capture_thread, self).__init__(*args, **kwargs)
        self.queue = deque(maxlen=20)
        self.capture = None

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if ret == False:
                continue
            self.queue.append(frame)

    def get_frame(self):
        while not self.queue:
            pass

        return self.queue.popleft()
