import time
from multiprocessing import Queue
from threading import Thread

import cv2

from conf import Tracking, resize_frame, draw_tr_ellipses, draw_tr_groups, swift, limit_fps, get_args
from count import get_statistics
from detecting import detect_bees
from tracking import Tracker_bees


class Frame_stream(Thread):

    def __init__(self):
        self.stop = False
        self._done = False
        self._extract_flow = Queue()
        self._classifierResultQueue = None
        self._frame_flow = None
        Thread.__init__(self)

    def get_position_flow(self):
        return self._extract_flow

    def frames_flow(self, queue):
        self._frame_flow = queue

    def run(self: Thread) -> None:
        _process_time = 0
        _process_cnt = 0

        tracker = Tracker_bees(50, 20)

        if type(self._frame_flow) == type(None):
            raise ("No frame flow provided!")

        while not self.stop:

            _start_t = time.time()

            if not self._frame_flow.empty():
                _process_cnt += 1

                fs = self._frame_flow.get()
                if resize_frame == "RES_150x300":
                    frame_1080, frame_540, frame_180 = fs
                elif resize_frame == "RES_75x150":
                    frame_540, frame_180 = fs

                detected_bee, detected_groups = detect_bees(frame_180, 3)

                if Tracking:
                    tracker.update(detected_bee, detected_groups)

                if not get_args().noPreview:

                    frame = frame_540.copy()
                    if draw_tr_ellipses:
                        for item in detected_bee:
                            cv2.ellipse(frame, item, (0, 0, 255), 2)
                    if draw_tr_groups:
                        for item in detected_groups:
                            cv2.ellipse(frame, item, (255, 0, 0), 2)

                    if Tracking:
                        tracker.draw_tr(frame)

                    skip_key = 1 if swift else 0

                    cv2.imshow("frame", frame)
                    if cv2.waitKey(skip_key) & 0xFF == ord('q'):
                        break

                _end_t = time.time() - _start_t
                limit_time = 1 / limit_fps
                if _end_t < limit_time:
                    time.sleep(limit_time - _end_t)

                _dh = get_statistics()
                _dh.frame_processed()

            else:
                time.sleep(0.1)

        self._done = True

    def is_done(self: Thread) -> bool:
        return self._done

    def stop(self: Thread) -> None:
        self.stop = True
