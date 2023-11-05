from ultralytics import YOLO
from multiprocessing import Process, Queue
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

CLASSES = ["corner", "middle", "paint", "tick"]


def center(box):
    x, y, w, h, _, _, _ = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    return (x + int(w / 2), y + int(h / 2))


def parse(data):
    data = data.boxes.data
    labels = collections.defaultdict(list)
    for label in data:
        id = int(label[4])
        conf = label[5].item()
        label_class = CLASSES[int(label[6])]
        labels[label_class].append((center(label), conf, id))
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
        source=source, project="./", conf=0.1, stream=True, show=True
    )
    for detection in detections:
        detection_list.put(detection)


def track():
    player_detection_queue = Queue()
    court_detection_queue = Queue()

    player_thread = Process(
        target=detect,
        args=(player_detection_queue, "models/yolo/yolov8t.pt", "data/yolo/video.mp4"),
    )
    court_thread = Process(
        target=detect,
        args=(court_detection_queue, "models/yolo/yolov8c.pt", "data/yolo/video.mp4"),
    )

    player_thread.start()
    court_thread.start()

    while (
        player_thread.is_alive()
        or court_thread.is_alive()
        or not player_detection_queue.empty()
        or not court_detection_queue.empty()
    ):
        if player_detection_queue.empty() and not court_detection_queue.empty():
            continue

        player_data = player_detection_queue.get()
        court_data = court_detection_queue.get()
        print("player_data: ", player_data)
        print("court_data: ", court_data)

    player_thread.join()
    court_thread.join()


if __name__ == "__main__":
    track()
