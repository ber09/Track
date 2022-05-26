from filterpy.common import kinematic_kf
from collections import deque
import numpy as np
import math
import cv2
import random

from conf import draw_rectangle, draw_tracking_line, draw_id, draw_stat, point_in_e
from count import get_statistics


class Track_bee():

    def __init__(self, id):
        super(Track_bee, self).__init__()
        self._last_detect = None
        self.trace = deque(maxlen=10000)
        self.skipped_frames = 0
        self.processed_frames = 0
        self.track_id = id
        self.first_position = None
        self.in_group = False
        self.dt = 1
        self.KF = kinematic_kf(dim=2, order=2, dt=self.dt, dim_z=1, order_by_dim=True)
        self.KF.R *= 2
        self.KF.Q = np.array(
            [[self.dt ** 4 / 4, self.dt ** 3 / 2, self.dt ** 4 / 2, 0, 0, 0],
             [self.dt ** 3 / 2, self.dt ** 2, self.dt ** 4, 0, 0, 0],
             [self.dt ** 3 / 1, self.dt ** 1, self.dt ** 1 / 2, 0, 0, 0],
             [0, 0, 0, self.dt ** 4 / 4, self.dt ** 3 / 2, self.dt ** 4 / 2],
             [0, 0, 0, self.dt ** 3 / 2, self.dt ** 2, self.dt ** 4],
             [0, 0, 0, self.dt ** 3 / 1, self.dt ** 1, self.dt ** 1 / 2]
             ])

    def set_position(self, position):
        self.KF.x[0] = position[0]
        self.KF.x[3] = position[1]

        if len(self.trace) == 0:
            self.first_position = position
        self.trace.append(position)

    def correct(self, position):
        self.trace.append(position)
        self.KF.update(position[0:2])

    def predict(self):
        self.KF.predict()
        self.last_predict = self.KF.x
        return self.KF.x


class Tracker_bees(object):

    def __init__(self, dist_threshold, max_frame_skipped, frame_size=(960, 540)):
        super(Tracker_bees, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.track_id = 0
        self.tracks = []
        self._frame_height = frame_size[1]
        self._frame_width = frame_size[0]
        self.track_colors = []
        for i in range(255):
            self.track_colors.append((random.randint(100, 255),
                                      random.randint(100, 255),
                                      random.randint(100, 255)))

    def get_track_id(self, id):
        for item in self.tracks:
            if item.track_id == id:
                return item
        return None

    def draw_tr(self, frame):

        for i in range(len(self.tracks)):

            if len(self.tracks[i].trace) > 1:

                t_c = self.track_colors[self.tracks[i].track_id % len(self.track_colors)]

                if draw_rectangle and self.tracks[i].in_group:
                    x = int(self.tracks[i].trace[-1][0])
                    y = int(self.tracks[i].trace[-1][1])
                    tl = (x - 30, y - 30)
                    br = (x + 30, y + 30)
                    cv2.rectangle(frame, tl, br, (0, 0, 0), 10)

                if draw_tracking_line:
                    for j in range(len(self.tracks[i].trace)):
                        x = int(self.tracks[i].trace[j][0])
                        y = int(self.tracks[i].trace[j][1])

                        if j > 0:
                            x2 = int(self.tracks[i].trace[j - 1][0])
                            y2 = int(self.tracks[i].trace[j - 1][1])
                            cv2.line(frame, (x, y), (x2, y2), t_c, 4)
                            cv2.line(frame, (x, y), (x2, y2), (0, 0, 0), 1)

                if draw_id:
                    cv2.putText(frame, str(self.tracks[i].track_id), (x, y - 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

        if draw_stat:
            _dh = get_statistics()
            bees_in, bees_out = _dh.get_count_over()
            cv2.putText(frame, "In: {0}, Out: {1}".format(bees_in, bees_out), (20, 510),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        return frame

    def get_last_position(self, frame_step):
        data = []
        for j in range(len(self.tracks)):
            track = self.tracks[j]
            if len(track.trace) and track.skipped_frames == 0 and \
                    track.processed_frames % frame_step == 0:
                data.append((self.tracks[j].track_id, self.tracks[j].trace[-1]))
        return data

    def isOutOfPane(self, pos):
        return pos[1] < 5 or pos[1] > (self._frame_height - 5)

    def _delete_tr(self, id, count=False):
        track = self.tracks[id]
        if count:
            _dh = get_statistics()
            f_y = track.first_position[1]
            l_y = track.trace[-1][1]
            pH = int(self._frame_height / 2)
            if f_y > pH and l_y <= pH:
                _dh.add_in()
            if f_y < pH and l_y >= pH:
                _dh.add_out()

        del self.tracks[id]

    def update(self, detections: list, groups: list):
        tmp = np.zeros((len(detections), 5))
        for i, item in enumerate(detections):
            tmp[i] = np.concatenate((item[0], item[1], [item[2]]), axis=0)
        detections = tmp

        def matched(item):
            t = item[1]
            d = item[2]
            used_tracks.append(t)
            used_detections.append(d)
            self.tracks[t]._last_dectect = detections[d]
            self.tracks[t].correct(detections[d])
            self.tracks[t].skipped_frames = 0
            self.tracks[t].processed_frames += 1

        dist_list = []
        for num_t, item_t in enumerate(self.tracks):

            item_t.in_group = False
            for g in groups:
                item_t.in_group |= point_in_e(item_t.trace[-1], g)

            item_t.skipped_frames += 1

            if item_t.in_group:
                item_t.KF.x[1] = item_t.KF.x[1] * 0.5
                item_t.KF.x[2] = item_t.KF.x[2] * 0.5
                item_t.KF.x[4] = item_t.KF.x[4] * 0.5
                item_t.KF.x[5] = item_t.KF.x[5] * 0.5
                item_t.skipped_frames -= 1

            predict_b = item_t.predict()
            for num_d, item_d in enumerate(detections):

                if item_t.in_group:
                    last_v_position = item_t._last_detect
                    p_diff = (np.array([last_v_position[0],
                                        last_v_position[1]]).reshape(-1, 2) -
                              np.array(item_d[0:2]).reshape(-1, 2))[0]
                    p_dist = math.sqrt(p_diff[0] * p_diff[0] + p_diff[1] * p_diff[1])
                    dist_list.append((p_dist, num_t, num_d))

                p_diff = (np.array([predict_b[0], predict_b[3]]).reshape(-1, 2) -
                          np.array(item_d[0:2]).reshape(-1, 2))[0]

                p_dist = math.sqrt(p_diff[0] * p_diff[0] + p_diff[1] * p_diff[1])

                dist_list.append((p_dist, num_t, num_d))

        dist_list = sorted(dist_list, key=lambda entry: entry[0])

        used_tracks = []
        used_detections = []

        for item in dist_list:
            dist, num_t, num_d = item

            if num_t in used_tracks or num_d in used_detections:
                continue

            per_track = list(filter(lambda x: x[1] == num_t, dist_list))

            per_track = list(filter(lambda x: x[2] not in used_detections, per_track))

            by_dist = list(filter(lambda x: x[0] < self.dist_threshold, per_track))

            if len(by_dist) == 1:
                matched(by_dist[0])
            elif len(by_dist) > 1:
                matched(by_dist[0])

        IN = 0
        OUT = 0

        for num_t in reversed(range(len(self.tracks))):
            item = self.tracks[num_t]

            if num_t not in used_tracks and item.skipped_frames > 0 and item.processed_frames == 0:
                self._delete_tr(num_t, count=False)

            elif num_t not in used_tracks and item.skipped_frames > item.processed_frames:
                self._delete_tr(num_t, count=False)

            elif self.tracks[num_t].skipped_frames > self.max_frame_skipped:
                self._delete_tr(num_t, count=False)

            elif self.isOutOfPane(self.tracks[num_t].trace[-1]):
                self._delete_tr(num_t, count=True)

            elif self.isOutOfPane([self.tracks[num_t].last_predict[0],
                                   self.tracks[num_t].last_predict[3]]):
                self._delete_tr(num_t, count=True)

        unmatched_detections = list(filter(lambda x: x not in used_detections,
                                           range(len(detections))))
        for item in unmatched_detections:

            if True:
                track = Track_bee(self.track_id)
                track._last_detect = detections[item]
                self.tracks.append(track)
                track.set_position(detections[item])
                self.track_id += 1

        return (IN, OUT)
