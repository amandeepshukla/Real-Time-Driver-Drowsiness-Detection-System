import time

import cv2
import dlib
import imutils

from config import (
    SHAPE_PREDICTOR_PATH,
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    CAMERA_INDEX,
    FRAME_WIDTH,
    LEFT_EYE_POINTS,
    RIGHT_EYE_POINTS,
)
from ear_utils import shape_to_np, average_ear
from alarm import Alarm


def draw_eye_contour(frame, eye_points):
    hull = cv2.convexHull(eye_points)
    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)


def main():
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    alarm = Alarm()

    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(1)
    time.sleep(1.0)  # let camera warm up

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {1}")

    frame_counter = 0  # consecutive frames below EAR threshold

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame, exiting.")
                break

            # Resize only for display/processing speed; detection still
            # runs on this same unaltered-content frame (no filters applied).
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray, 0)

            for face in faces:
                shape = predictor(gray, face)
                landmarks = shape_to_np(shape)

                ear, left_eye, right_eye = average_ear(
                    landmarks, LEFT_EYE_POINTS, RIGHT_EYE_POINTS
                )

                draw_eye_contour(frame, left_eye)
                draw_eye_contour(frame, right_eye)

                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= EAR_CONSEC_FRAMES:
                        alarm.trigger()
                        cv2.putText(
                            frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )
                else:
                    frame_counter = 0

                cv2.putText(
                    frame, f"EAR: {ear:.2f}", (FRAME_WIDTH - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
