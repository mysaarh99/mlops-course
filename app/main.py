from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os, io, time, logging
from PIL import Image
import wandb

logging.basicConfig(level=logging.INFO)

# ========= إعدادات قابلة للتهيئة =========
# اختر أي نموذج نص/صورة عبر المتغيرات البيئية
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "mlflow")   # baseline|mlflow|wandb
IMG_MODEL_NAME  = os.getenv("IMG_MODEL_NAME",  "wandb")   # baseline|mlflow|wandb

TEXT_MODEL_PATH = f"models/text_{TEXT_MODEL_NAME}_model.pkl"
VECT_PATH       = f"models/text_{TEXT_MODEL_NAME}_vectorizer.pkl"
IMG_MODEL_PATH  = f"models/cifar10_model_{IMG_MODEL_NAME}.keras"

WANDB_PROJECT   = os.getenv("WANDB_PROJECT", "mlops-monitoring")
WANDB_ENTITY    = os.getenv("WANDB_ENTITY")  # اختياري
WANDB_MODE      = os.getenv("WANDB_MODE")    # offline/online (اتركه للبيئة)

# ========= تحميل النماذج =========
def safe_load_text_models(model_path, vect_path):
    if not (os.path.exists(model_path) and os.path.exists(vect_path)):
        logging.warning("Text model/vectorizer not found.")
        return None, None
    return joblib.load(model_path), joblib.load(vect_path)

def safe_load_image_model(model_path):
    if not os.path.exists(model_path):
        logging.warning("Image model not found.")
        return None
    return tf.keras.models.load_model(model_path)

text_model, vectorizer = safe_load_text_models(TEXT_MODEL_PATH, VECT_PATH)
image_model = safe_load_image_model(IMG_MODEL_PATH)

# ========= واندبي: Run واحد للخدمة =========
# استخدم API key في البيئة (لا تستدعِ wandb.login() هنا)
wandb_run = None
try:
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="inference_service",
        name=f"api-inference-{int(time.time())}",
        reinit=False,
        settings=wandb.Settings(start_method="thread")
    )
    wandb.config.update({
        "text_model_name": TEXT_MODEL_NAME,
        "image_model_name": IMG_MODEL_NAME
    }, allow_val_change=True)
except Exception as e:
    logging.warning(f"W&B init failed or disabled: {e}")

# ========= بيانات CIFAR-10 =========
cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# ========= FastAPI =========
app = FastAPI(title="Unified ML Models API", version="1.1")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logging.info(f"{request.method} {request.url.path} finished in {duration:.3f}s")
    return response

@app.get("/health")
async def health_check():
    classes = None
    try:
        classes = text_model.classes_.tolist() if text_model is not None else None
    except Exception:
        classes = None
    return {
        "status": "ok",
        "text_model_loaded": text_model is not None,
        "image_model_loaded": image_model is not None,
        "text_model_path": TEXT_MODEL_PATH,
        "image_model_path": IMG_MODEL_PATH
    }

# ======== معالجة الصورة ========
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 5_000_000))  # 5MB

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# ======== طلب النص ========
class PredictTextRequest(BaseModel):
    text: str

@app.post("/predict-text")
async def predict_text(req: PredictTextRequest):
    if text_model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Text model not available.")
    try:
        t0 = time.time()
        X = vectorizer.transform([req.text])
        # التنبؤ والاحتمال
        pred = int(text_model.predict(X)[0])
        prob = None
        if hasattr(text_model, "predict_proba"):
            prob = float(np.max(text_model.predict_proba(X)))
        duration = time.time() - t0

        # تسجيل في W&B (إن كان مفعلًا)
        if wandb_run is not None:
            wandb.log({
                "endpoint": "predict-text",
                "input_len": len(req.text or ""),
                "predicted_class": pred,
                "confidence": prob,
                "inference_time_sec": duration
            })

        return {
            "model": TEXT_MODEL_NAME,
            "predicted_class": pred,
            "confidence": prob,
            "inference_time_sec": duration
        }
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not available.")
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported image type.")
    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    try:
        t0 = time.time()
        X = preprocess_image(raw)
        preds = image_model.predict(X, verbose=0)
        idx = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
        duration = time.time() - t0

        if wandb_run is not None:
            wandb.log({
                "endpoint": "predict-image",
                "predicted_class": cifar_classes[idx],
                "confidence": conf,
                "inference_time_sec": duration
            })

        return {
            "model": IMG_MODEL_NAME,
            "predicted_class": cifar_classes[idx],
            "confidence": conf,
            "inference_time_sec": duration
        }
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
