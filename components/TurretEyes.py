import cv2
import numpy as np
import logging
import configparser
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Read configuration
config = configparser.ConfigParser()
config.read("config.ini")

MODEL_PROTO = config["Model"]["Proto"]
MODEL_WEIGHTS = config["Model"]["Weights"]

# Threshold and image size constants
THRESHOLD_VALUE = 30
MIN_CONTOUR_AREA = 1000
BLUR_SIZE = (21, 21)
DILATE_ITERATIONS = 2
IMAGE_SIZE = (300, 300)
TARGET_LOCK_DURATION = (
    5  # Duration to lock onto the target after motion detection (in seconds)
)


class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", src)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        self.cap.release()


class HumanDetector:
    def __init__(self, confidence_threshold=0.2):
        self.net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, IMAGE_SIZE), 0.007843, IMAGE_SIZE, 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        humans = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # Class label for 'person' in COCO dataset
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    humans.append((startX, startY, endX, endY))
        return humans


class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def is_moving(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.prev_frame = gray

        # Filter out small contours
        motion_detected = any(
            cv2.contourArea(contour) > MIN_CONTOUR_AREA for contour in contours
        )
        return motion_detected


class HumanMarker:
    @staticmethod
    def mark(frame, humans, target_index):
        if target_index is not None and target_index < len(humans):
            (startX, startY, endX, endY) = humans[target_index]
            centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
            radius = 10  # Make the circle smaller
            color = (0, 0, 255)  # Red color in BGR format
            cv2.circle(frame, (centerX, centerY), radius, color, 2)
        return frame


class TargetSelector:
    def __init__(self):
        self.target_index = None
        self.target_centroid = None
        self.last_motion_time = 0

    def select_target(self, humans):
        if not humans:
            self.target_index = None
            return None

        current_time = time.time()

        if self.target_centroid is None or (
            current_time - self.last_motion_time > TARGET_LOCK_DURATION
        ):
            # Select the first human detected if no target is currently selected or lock duration expired
            self.target_index = 0
            self.target_centroid = self.get_centroid(humans[self.target_index])
            return self.target_index

        # Find the human closest to the current target's centroid
        min_distance = float("inf")
        closest_index = None
        for i, (startX, startY, endX, endY) in enumerate(humans):
            centroid = self.get_centroid((startX, startY, endX, endY))
            distance = self.compute_distance(self.target_centroid, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        self.target_index = closest_index
        self.target_centroid = self.get_centroid(humans[self.target_index])
        return self.target_index

    def update_target(self, humans, is_moving):
        if is_moving:
            self.last_motion_time = time.time()
            self.select_target(humans)
        elif self.target_centroid is None or (
            time.time() - self.last_motion_time > TARGET_LOCK_DURATION
        ):
            self.target_index = None

    @staticmethod
    def get_centroid(box):
        (startX, startY, endX, endY) = box
        centerX = (startX + endX) // 2
        centerY = (startY + endY) // 2
        return (centerX, centerY)

    @staticmethod
    def compute_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    camera = Camera()
    detector = HumanDetector()
    motion_detector = MotionDetector()
    target_selector = TargetSelector()

    try:
        while True:
            frame = camera.get_frame()
            humans = detector.detect(frame)
            is_moving = motion_detector.is_moving(frame)
            target_selector.update_target(humans, is_moving)
            frame = HumanMarker.mark(frame, humans, target_selector.target_index)
            if is_moving:
                logging.info(
                    f"Motion detected. Tracking target index: {target_selector.target_index}"
                )
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
