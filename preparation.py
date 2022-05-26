import cv2
import time
import multiprocessing
import signal

from conf import buffer, set_frame_on_camera, camera_resolution, get_frame_config


class Preparation(object):

    def __init__(self, video_source=None, video_file=None):

        self.frame_config = None
        self._video_flow = None
        self._stop = multiprocessing.Value('i', 0)
        self._start = multiprocessing.Value('i', 0)
        self._process = None

        frame_config = get_frame_config()

        self.frame_config = frame_config
        if video_file is not None:
            self._flow = multiprocessing.Queue(maxsize=buffer)

        else:
            self._flow = multiprocessing.Queue(maxsize=set_frame_on_camera)

        self._process = multiprocessing.Process(target=self._preparate,
                                                args=(
                                                    self._flow, frame_config, video_source, video_file, self._stop,
                                                    self._start))
        self._process.start()

    def get_flow(self):
        return self._flow

    def is_started(self):
        return self._start.value

    def is_done(self):
        return self._stop.value

    def stop(self) -> None:
        self._stop.value = 1
        while not self._flow.empty():
            self._flow.get()

    def join(self):
        self._process.terminate()
        self._process.join()

    @staticmethod
    def _preparate(flow_out, config, video_source, video_file, stop, start):

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if video_source == None:

            _video_flow = cv2.VideoCapture(video_file)
        else:

            _video_flow = cv2.VideoCapture(video_source)
            width, height, f = camera_resolution
            if f is not None:
                fourcc = cv2.VideoWriter_fourcc(*f)
                _video_flow.set(cv2.CAP_PROP_FOURCC, fourcc)
            if width is not None:
                _video_flow.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height is not None:
                _video_flow.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        _skipped_cnt = 0
        while stop.value == 0:

            if flow_out.full():
                time.sleep(0.1)
                _skipped_cnt += 1
            else:
                _ret, _frame = _video_flow.read()

                if start.value == 0:
                    start.value = 1

                if _ret:

                    frame_set = tuple()
                    for item in config:
                        width, height = _frame.shape[0:2]
                        if width != item[0] or height != item[1]:
                            _frame = cv2.resize(_frame, (item[1], item[0]))
                        if item[2] == cv2.IMREAD_GRAYSCALE:
                            tmp = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
                            frame_set += (tmp,)
                        else:
                            frame_set += (_frame,)

                    flow_out.put(frame_set)

                else:
                    stop.value = 1

