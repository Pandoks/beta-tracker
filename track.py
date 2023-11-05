from ultralytics import YOLO
from multiprocessing import Process, Queue


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

    index = 0
    while (
        player_thread.is_alive()
        or court_thread.is_alive()
        or not player_detection_queue.empty()
        or not court_detection_queue.empty()
    ):
        if player_detection_queue.empty() and not court_detection_queue.empty():
            continue

        # print(player_detection_queue.qsize())

        player_data = player_detection_queue.get()
        court_data = court_detection_queue.get()
        print("player_data: ", player_data)
        print("court_data: ", court_data)
        index += 1

    player_thread.join()
    court_thread.join()


if __name__ == "__main__":
    track()
