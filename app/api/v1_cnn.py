# app/api/v1_cnn.py
import io
from typing import Dict

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.core.settings import get_settings
from app.services.security import verify_api_key
# (Grad-CAM can be re-added later with tf-keras-vis)

router = APIRouter()
settings = get_settings()

# ── model & constants ───────────────────────────────────
model = tf.keras.models.load_model(settings.MODEL_PATH)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMG_SIZE = 28  # model was trained on 28×28 RGB

# ── pydantic response schema ────────────────────────────
class ClassifyOut(BaseModel):
    label: str
    prob: float

# ── helpers ─────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    """Resize & scale image; return tensor shaped (1, IMG_SIZE, IMG_SIZE, 3)."""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32") / 255.0
    return arr[np.newaxis, ...]


# ── routes ──────────────────────────────────────────────
@router.post(
    "/v1/classify",
    response_model=ClassifyOut,
    dependencies=[Depends(verify_api_key)],
)
async def classify(file: UploadFile = File(...)) -> Dict[str, float]:
    # 1. read & preprocess
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = preprocess(img)

    # 2. inference
    probs = model.predict(tensor, verbose=0)[0]  # shape: (num_classes,)
    idx = int(np.argmax(probs))
    prob = round(float(probs[idx]), 4)
    label = CLASS_NAMES[idx]

    return {"label": label, "prob": prob}


@router.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    """Liveness probe for load-balancers."""
    return "alive"
