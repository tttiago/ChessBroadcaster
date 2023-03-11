"""Capture video stream in a different thread."""

from queue import Queue
from threading import Thread


class Video_capture_thread(Thread):
    def __init__(self, *args, **kwargs):
        super(Video_capture_thread, self).__init__(*args, **kwargs)
        self.queue = Queue()
        self.capture = None

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if ret == False:
                continue
            self.queue.put(frame)

    def get_frame(self):
        return self.queue.get()
