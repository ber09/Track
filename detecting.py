import cv2
import numpy as np
import math


def detect_bees(frame, scale):
    def near(point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

    def area(ellips):
        return np.pi * ellips[1][0] * ellips[1][1]

    blue, green, red = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    o = 255 - (green - v)

    o = cv2.GaussianBlur(o, (9, 9), 9)
    _, o = cv2.threshold(o, 150, 255, cv2.THRESH_BINARY)

    o = 255 - o

    contours, hierarchy = cv2.findContours(o, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    ellipses = []
    groups = []

    for i in range(len(contours)):

        if len(contours[i]) >= 5:

            ellips = cv2.fitEllipse(contours[i])

            if ellips[1][0] < 8 or ellips[1][1] < 8:
                continue

            area_el = area(ellips)
            if 100 < area_el < 2500:
                ellips = (
                    (ellips[0][0] * scale, ellips[0][1] * scale), (ellips[1][0] * scale, ellips[1][1] * scale),
                    ellips[2])
                ellipses.append(ellips)
            elif 3000 < area_el < 12500:
                ellips = (
                    (ellips[0][0] * scale, ellips[0][1] * scale), (ellips[1][0] * scale, ellips[1][1] * scale),
                    ellips[2])
                groups.append(ellips)

    done = []
    skip = []
    solved = []

    for i in ellipses:

        group = []
        for blue in ellipses:

            if (i, blue) in done or (blue, i) in done or i == blue:
                continue
            done.append((i, blue))

            dist = near(i[0], blue[0])
            if dist < 50:

                if i not in group:
                    group.append(i)
                if blue not in group:
                    group.append(blue)

                if not i in skip:
                    skip.append(i)
                if not blue in skip:
                    skip.append(blue)

        if len(group):
            solved.append(max(group, key=area))

    rest = list(filter(lambda x: x not in skip, ellipses))
    merged = rest + solved

    return merged, groups
