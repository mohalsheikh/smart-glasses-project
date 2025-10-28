import cv2
import torch
from ultralytics import YOLO
import os

def main():
    # resolve path to weights relative to this file
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    weights_path = os.path.join(PROJECT_ROOT, "models", "yolo_glasses_v2.pt")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    print("Loading model:", weights_path)

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera 0, trying 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Still couldn't open camera. Exiting.")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        results = model.predict(
            source=frame,
            device=device,
            conf=0.3,
            verbose=False
        )

        r = results[0]

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for (x1, y1, x2, y2), score, cid in zip(boxes_xyxy, scores, class_ids):
            label = f"{names[cid]} {score:.2f}"

            # draw box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # draw text bg
            cv2.rectangle(
                frame,
                (int(x1), int(y1) - 20),
                (int(x1) + 200, int(y1)),
                (0, 255, 0),
                -1
            )

            # draw text
            cv2.putText(
                frame,
                label,
                (int(x1) + 5, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imshow("Smart Glasses Detector (raw YOLO)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
