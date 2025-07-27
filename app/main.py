from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import joblib
import os
from PIL import Image
import io

# تحميل نموذج النصوص (Logistic Regression + Vectorizer)
text_model_path = "models/logistic_model_best.pkl"
vectorizer_path = "models/vectorizer_trained.pkl"
text_model = joblib.load(text_model_path) if os.path.exists(text_model_path) else None
vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None

# تحميل نموذج الصور (CIFAR-10 CNN)
image_model_path = "models/cifar10_model_wandb_visual.keras"
image_model = tf.keras.models.load_model(image_model_path) if os.path.exists(image_model_path) else None

# أسماء فئات CIFAR-10
cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# إنشاء التطبيق
app = FastAPI(title="Unified ML Models API", description="تصنيف نصوص وصور في واجهة واحدة", version="1.0")

# دالة مساعدة لتحضير الصورة
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))  # CIFAR-10 حجم الصور
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# واجهة لتصنيف النصوص
@app.post("/predict-text")
async def predict_text(text: str = Form(...)):
    if text_model is None or vectorizer is None:
        return JSONResponse({"error": "نموذج النصوص غير متاح."}, status_code=500)
    try:
        text_vec = vectorizer.transform([text])
        prediction = text_model.predict(text_vec)[0]
        return {"input_text": text, "predicted_class": int(prediction)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# واجهة لتصنيف الصور
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if image_model is None:
        return JSONResponse({"error": "نموذج الصور غير متاح."}, status_code=500)
    try:
        image_bytes = await file.read()
        input_data = preprocess_image(image_bytes)
        predictions = image_model.predict(input_data)
        pred_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        return {
            "predicted_class": cifar_classes[pred_index],
            "confidence": confidence
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
