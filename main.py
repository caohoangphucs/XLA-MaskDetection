import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model("mask_mobilenet.h5")

# Load face detector
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# GET / -> index.html
@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()

# POST /detect
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    results = []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < 0.3:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)  # thường cho Y lớn hơn (cằm + khẩu trang)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)


        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face.astype("float32"))
        face = np.expand_dims(face, axis=0)

        pred = float(model.predict(face, verbose=0)[0][0])

        if pred < 0.5:
            label = "With Mask"
            confidence = (1 - pred) * 100
        else:
            label = "No Mask"
            confidence = pred * 100

        results.append({
        "box": [int(x1), int(y1), int(x2), int(y2)],
        "label": label,
        "confidence": float(round(confidence, 1))
    })


    return JSONResponse({
        "faces": len(results),
        "results": results
    })
