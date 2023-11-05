from ultralytics import YOLO
from multiprocessing import Manager, Process, Queue
import numpy as np
import cv2
import collections


DIAGRAM_POINTS = {
    "left_corner": (145, 130),
    "right_corner": (1355, 130),
    "left_tick": (655, 130),
    "right_tick": (1345, 130),
    "middle": (1000, 130),
    "left_paint": {
        "t_left": (145, 440),
        "t_right": (490, 440),
        "b_left": (145, 730),
        "b_right": (490, 730),
    },
    "right_paint": {
        "t_left": (1510, 440),
        "t_right": (1855, 440),
        "b_left": (1510, 730),
        "b_right": (1855, 730),
    },
}

COURT_CLASSES = ["corner", "middle", "paint", "tick"]


def center(coords):
    x, y, w, h = coords
    return (int(x + w / 2), int(y + h / 2))


# Only for court
def parse_court(data):
    print("in parse")
    labels = collections.defaultdict(list)
    for label in data:
        coords = (label[0], label[1], label[2], label[3])
        conf = label[-2].item()
        label_class = COURT_CLASSES[int(label[-1])]
        id = None
        if len(label) == 7:
            id = int(label[-3])

        labels[label_class].append((center(coords), conf, id))

    return labels


def define_paint(corners):
    sorted_x_corners = sorted(corners, key=lambda x: x[0])
    left_corners = sorted_x_corners[:2]
    right_corners = sorted_x_corners[2:]

    sorted_left_y_corners = sorted(left_corners, key=lambda x: x[1])
    sorted_right_y_corners = sorted(right_corners, key=lambda x: x[1])
    return {
        "top_left": sorted_left_y_corners[1],
        "top_right": sorted_right_y_corners[1],
        "bottom_left": sorted_left_y_corners[0],
        "bottom_right": sorted_right_y_corners[0],
    }


def homography(labels, threshold=0):
    paint = define_paint(labels["paint"])
    court_points = np.array(
        [
            paint["top_left"],
            paint["top_right"],
            paint["bottom_right"],
            paint["bottom_left"],
        ],
        dtype="float32",
    )

    diagram_points = np.array(
        [
            court_points["right_paint"]["t_left"],
            court_points["right_paint"]["t_right"],
            court_points["right_paint"]["b_right"],
            court_points["right_paint"]["b_left"],
        ],
        dtype="float32",
    )

    homography_matrix, _ = cv2.findHomography(court_points, diagram_points)

    return homography_matrix


def detect(detection_list, model_path, source):
    model = YOLO(model_path)
    detections = model.track(
        source=source, project="./", conf=0.1, iou=0.5, stream=True, show=True
    )
    for detection in detections:
        print(detection.boxes)
        detection_list.append(detection.boxes.data)


def track():
    manager = Manager()
    player_detection_list = manager.list()
    court_detection_list = manager.list()

    player_thread = Process(
        target=detect,
        args=(player_detection_list, "models/yolo/yolov8t.pt", "data/yolo/video.mp4"),
    )
    court_thread = Process(
        target=detect,
        args=(court_detection_list, "models/yolo/yolov8c.pt", "data/yolo/video.mp4"),
    )

    court_thread.start()
    player_thread.start()

    while (
        player_thread.is_alive()
        or court_thread.is_alive()
        or player_detection_list
        or court_detection_list
    ):
        if not len(player_detection_list) or not len(court_detection_list):
            continue

        court_data = court_detection_list.pop(0)
        player_data = player_detection_list.pop(0)
        court_data = parse_court(court_data)
        # player_data = parse(player_data)

    player_thread.join()
    court_thread.join()


if __name__ == "__main__":
    track()
