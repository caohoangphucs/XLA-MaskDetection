import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained mask model
model = load_model("mask_mobilenet.h5")

# Load OpenCV DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better FPS
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # Face detection blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        det_conf = detections[0, 0, i, 2]

        if det_conf > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            # Clamp box to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess face for MobileNet
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face.astype("float32"))
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0][0]

            # Decode prediction
            if pred < 0.5:
                label = "With Mask"
                confidence = (1 - pred) * 100
                color = (0, 255, 0)
            else:
                label = "No Mask"
                confidence = pred * 100
                color = (0, 0, 255)

            # Optional: uncertain prediction
            if confidence < 70:
                label = "Uncertain"
                color = (0, 255, 255)

            text = f"{label}: {confidence:.1f}%"

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Mask Detection - Realtime (DNN)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
