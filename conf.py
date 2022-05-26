# Размер очереди
import argparse
import math
import cv2 as cv
buffer = 20

# Длина буферизованных ищображений для входов
set_frame_on_camera = 3

# Разрешение камеры
camera_resolution = [1920, 1080, "MJPG"]

# Вкл/Выкл трекинг
Tracking = True

# Ресайз изображений
resize_frame = 'RES_75x150'

# Отрисовать детектировнные эллипсы
draw_tr_ellipses = True

# Отрисовать детектировнные группы
draw_tr_groups = False

# Переключатель
swift = True

# Предел FPS
limit_fps = 20

# Отрисовать прмоугольник
draw_rectangle = False

# Отрисовка линий трекинга
draw_tracking_line = True

# Отрисовка id
draw_id = True

# Отрисовка статистики
draw_stat = True


def get_args():
    """! Prepares and parses the command arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--noPreview", help="Run without producing any visual output", action="store_true")
    parser.add_argument("--video", help="Use video file")
    return parser.parse_args()


def point_in_e(point, ellipse):
    x_point = point[0]
    y_point = point[1]
    x_ellips = ellipse[0][0]
    y_ellips = ellipse[0][1]
    diametr_x = ellipse[1][0] / 2
    diametr_y = ellipse[1][1] / 2
    angle = ellipse[2] / 180 * math.pi  # ((e[2] * 180 / math.pi) + 180) % 180
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    t1 = cos_a * (x_point - x_ellips) + sin_a * (y_point - y_ellips)
    t2 = sin_a * (x_point - x_ellips) - cos_a * (y_point - y_ellips)
    res = ((t1 * t1) / (diametr_x * diametr_x)) + ((t2 * t2) / (diametr_y * diametr_y))

    return res <= 1


def get_frame_config():
    frame_config = None
    if resize_frame == "RES_75x150":
        frame_config = (
            (540, 960, cv.IMREAD_UNCHANGED),
            (180, 320, cv.IMREAD_UNCHANGED)
        )
    elif resize_frame == "RES_150x300":
        frame_config = (
            (1080, 1920, cv.IMREAD_UNCHANGED),
            (540, 960, cv.IMREAD_UNCHANGED),
            (180, 320, cv.IMREAD_UNCHANGED)
        )
    else:
        raise BaseException("Wrong image extraction setting")

    return frame_config
