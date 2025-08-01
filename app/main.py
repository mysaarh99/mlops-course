from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import joblib
import os
from PIL import Image
import io
import time
import logging
import wandb
import time


# 🔧 إعداد السجل
logging.basicConfig(level=logging.INFO)

# إعداد تسجيل W&B
wandb_project = os.getenv("WANDB_PROJECT", "mlops-monitoring")
wandb_entity = os.getenv("WANDB_ENTITY")  # يمكن تركه فارغًا لو لا تستخدم Teams
wandb.login()

def log_to_wandb(endpoint: str, input_type: str, input_sample, prediction, confidence=None, duration=None):
    with wandb.init(project=wandb_project, entity=wandb_entity, job_type="inference", reinit=True) as run:
        run.log({
            "endpoint": endpoint,
            "input_type": input_type,
            "input_sample": input_sample,
            "prediction": prediction,
            "confidence": confidence,
            "inference_time": duration
        })

# 📦 تحميل النماذج
text_model_path = "models/logistic_model_best.pkl"
vectorizer_path = "models/vectorizer.pkl"
text_model = joblib.load(text_model_path) if os.path.exists(text_model_path) else None
vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None

# تحميل نموذج الصور (CIFAR-10 CNN)
image_model_path = "models/cifar10_model_wandb_visual.keras"
image_model = tf.keras.models.load_model(image_model_path) if os.path.exists(image_model_path) else None

# أسماء فئات CIFAR-10
cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# 🚀 إنشاء التطبيق
app = FastAPI(title="Unified ML Models API", version="1.0")

# 📍 Middleware لتسجيل زمن الطلبات
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logging.info(f"{request.method} {request.url.path} finished in {duration:.3f}s")
    return response

# 🛡️ Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )

# ✅ Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "text_model_loaded": text_model is not None,
        "image_model_loaded": image_model is not None
    }

# 🖼️ إعداد الصورة
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# 🔤 واجهة التنبؤ للنصوص
# @app.post("/predict-text")
# async def predict_text(text: str = Form(...)):
#     if text_model is None or vectorizer is None:
#         return JSONResponse({"error": "نموذج النصوص غير متاح."}, status_code=500)
#     text_vec = vectorizer.transform([text])
#     prediction = text_model.predict(text_vec)[0]
#     return {"input_text": text, "predicted_class": int(prediction)}

# الدالة الجديدة تساعد في عملية المراقبة ما بعد النشر باستخدام wandb مما يسمح لنا برفع المعلومات وتخزينها هنا
@app.post("/predict-text")
async def predict_text(text: str = Form(...)):
    if text_model is None or vectorizer is None:
        return JSONResponse({"error": "Text model not available."}, status_code=500)
    try:
        start = time.time()
        text_vec = vectorizer.transform([text])
        prediction = text_model.predict(text_vec)[0]
        end = time.time()

        with wandb.init(project=wandb_project, entity=wandb_entity, job_type="inference", reinit=True) as run:
            run.log({
                "endpoint": "predict-text",
                "input_text": text,
                "predicted_class": int(prediction),
                "inference_time_sec": end - start
            })

        return {"input_text": text, "predicted_class": int(prediction)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 🖼️ واجهة التنبؤ للصور
# @app.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):
#     if image_model is None:
#         return JSONResponse({"error": "نموذج الصور غير متاح."}, status_code=500)
#     image_bytes = await file.read()
#     input_data = preprocess_image(image_bytes)
#     predictions = image_model.predict(input_data)
#     pred_index = np.argmax(predictions, axis=1)[0]
#     confidence = float(np.max(predictions))
#     return {
#         "predicted_class": cifar_classes[pred_index],
#         "confidence": confidence
#     }

@app.post("/predict-text")
async def predict_text(text: str = Form(...)):
    if text_model is None or vectorizer is None:
        return JSONResponse({"error": "Text model not available."}, status_code=500)
    try:
        start = time.time()
        text_vec = vectorizer.transform([text])
        prediction = text_model.predict(text_vec)[0]
        end = time.time()

        with wandb.init(project=wandb_project, entity=wandb_entity, job_type="inference", reinit=True) as run:
            run.log({
                "endpoint": "predict-text",
                "input_text": text,
                "predicted_class": int(prediction),
                "inference_time_sec": end - start
            })

        return {"input_text": text, "predicted_class": int(prediction)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
