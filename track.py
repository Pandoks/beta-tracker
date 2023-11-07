from ultralytics import YOLO
from multiprocessing import Manager, Process
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
PLAYER_CLASSES = [
    "Ball",
    "Hoop",
    "Period",
    "Player",
    "Ref",
    "Shot Clock",
    "Team Name",
    "Team Points",
    "Time Remaining",
]


def center_court(coords):
    x1, y1, x2, y2 = coords
    width, height = x2 - x1, y2 - y1
    return (int(x1 + width / 2), int(y1 + height / 2))


def center_player(coords):
    x1, y1, x2, y2 = coords
    width, height = x2 - x1, y2 - y1
    return (int(x1 + width / 2), int(y1 + height))


def parse_court(data):
    labels = collections.defaultdict(list)
    for label in data:
        coords = (label[0], label[1], label[2], label[3])
        conf = label[-2].item()
        label_class = COURT_CLASSES[int(label[-1])]
        id = None
        if len(label) == 7:
            id = int(label[-3])

        labels[label_class].append((center_court(coords), conf, id))

    return labels


def parse_player(data):
    labels = collections.defaultdict(list)
    for label in data:
        coords = (label[0], label[1], label[2], label[3])
        conf = label[-2].item()
        label_class = PLAYER_CLASSES[int(label[-1])]
        id = None
        if len(label) == 7:
            id = int(label[-3])
        labels[label_class].append((center_player(coords), conf, id))
    return labels


def define_paint(corners):
    if len(corners) != 4:
        return None

    sorted_x_corners = sorted(corners, key=lambda x: x[0])
    left_corners = sorted_x_corners[:2]
    right_corners = sorted_x_corners[2:]

    sorted_left_y_corners = sorted(left_corners, key=lambda x: x[1])
    sorted_right_y_corners = sorted(right_corners, key=lambda x: x[1])
    return {
        "top_left": sorted_left_y_corners[0],
        "top_right": sorted_right_y_corners[0],
        "bottom_left": sorted_left_y_corners[1],
        "bottom_right": sorted_right_y_corners[1],
    }


def homography(labels, threshold=0):
    paint = define_paint([label[0] for label in labels["paint"]])
    if not paint:
        return None

    court_points = np.array(
        [
            paint["top_left"],
            paint["top_right"],
            paint["bottom_right"],
            paint["bottom_left"],
        ],
        dtype=np.float32,
    )

    diagram_points = np.array(
        [
            DIAGRAM_POINTS["right_paint"]["t_left"],
            DIAGRAM_POINTS["right_paint"]["t_right"],
            DIAGRAM_POINTS["right_paint"]["b_right"],
            DIAGRAM_POINTS["right_paint"]["b_left"],
        ],
        dtype=np.float32,
    )

    homography_matrix, _ = cv2.findHomography(court_points, diagram_points)

    return homography_matrix


def transform_points(points, homography):
    points = [point[0] for point in points]
    points = np.array(points, dtype="float32").reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, homography)
    transformed_points = [
        (int(point[0][0]), int(point[0][1])) for point in transformed_points
    ]
    return transformed_points


def detect(detection_list, model_path, source):
    model = YOLO(model_path)
    detections = model.track(
        source=source, project="./", conf=0.1, iou=0.5, stream=True
    )
    for detection in detections:
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

    video = cv2.VideoCapture("data/yolo/video.mp4")
    print(video.read())

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
        player_data = parse_player(player_data)
        homography_matrix = homography(court_data)
        _, frame = video.read()
        if homography_matrix is None:
            continue

        transformed_points = transform_points(player_data["Player"], homography_matrix)
        image = cv2.imread("data/court.png")
        frame_width = frame.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x, y in transformed_points:
            cv2.circle(image, (x, y), radius=5, color=(0, 0, 0), thickness=-1)
            cv2.putText(image, f"({x},{y})", (x, y), font, 0.5, (255, 0, 0), 2)

        for point in player_data["Player"]:
            point = point[0]
            x, y = point
            cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(frame, f"({x},{y})", (x, y), font, 0.4, (0, 0, 0), 1)
        for point in court_data["paint"]:
            point = point[0]
            x, y = point
            cv2.circle(frame, point, radius=5, color=(0, 0, 0), thickness=-1)
            cv2.putText(frame, f"({x},{y})", (x, y), font, 0.4, (0, 0, 0), 1)

        image = cv2.resize(
            image, (frame_width, int(image.shape[0] * (frame_width / image.shape[1])))
        )

        combined = cv2.vconcat([frame, image])
        cv2.imshow("Track", combined)
        cv2.waitKey(1)

    player_thread.join()
    court_thread.join()


if __name__ == "__main__":
    track()
