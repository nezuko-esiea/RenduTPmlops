import os
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Geometrie Inference API")

REQS = Counter("http_requests_total", "Total HTTP requests", ["endpoint", "status"])
LAT = Histogram("http_request_duration_seconds", "Request latency", ["endpoint"])

MODEL_NAME = os.environ.get("MODEL_NAME", "geometrie-model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "None")  # ex: "Production" si tu utilises stages
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

model = None

def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MODEL_STAGE != "None":
        uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    else:
        # dernière version
        uri = f"models:/{MODEL_NAME}/latest"

    model = mlflow.tensorflow.load_model(uri)

@app.on_event("startup")
def startup():
    load_model()

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((28, 28))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    start = time.time()
    endpoint = "/predict"
    try:
        img = Image.open(file.file)
        x = preprocess(img)
        preds = model.predict(x)
        cls = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        REQS.labels(endpoint=endpoint, status="200").inc()
        LAT.labels(endpoint=endpoint).observe(time.time() - start)
        return {"class_id": cls, "confidence": conf, "probs": preds[0].tolist()}
    except Exception:
        REQS.labels(endpoint=endpoint, status="500").inc()
        LAT.labels(endpoint=endpoint).observe(time.time() - start)
        raise
