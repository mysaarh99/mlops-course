# mlflow_cnn.py
# -*- coding: utf-8 -*-
import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras

# ثباتية (اختياري)
tf.random.set_seed(42)
np.random.seed(42)

# إعدادات موحّدة
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

mlflow.set_experiment("CIFAR10_CNN_MLflow_FairCompare")

# ⏱️ بداية الزمن الكلي
overall_start = time.time()

# تحميل البيانات
train_data = np.load("data/cifar10/train.npz")
test_data  = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test,  y_test  = test_data["x"],  test_data["y"]

def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_model()

with mlflow.start_run():
    # تسجيل المعلمات الأساسية فقط (لتقليل الأعباء والحفاظ على عدالة الزمن)
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    })

    # تدريب
    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test), verbose=1
    )
    train_time = time.time() - t0

    # تقييم
    t1 = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    predict_time = time.time() - t1

    # ⏱️ نهاية الزمن الكلي
    overall_end = time.time()
    total_runtime = overall_end - overall_start

    # تسجيل المقاييس
    mlflow.log_metrics({
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "total_runtime_sec": total_runtime,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    })

    # حفظ النموذج وأرشفته
    os.makedirs("models", exist_ok=True)
    model_path = "models/cifar10_model_mlflow.keras"
    model.save(model_path)
    mlflow.keras.log_model(model, artifact_path="cifar10_cnn_model")

    print("✅ MLflow run complete")
    print(f"Train time (s):    {train_time:.4f}")
    print(f"Predict time (s):  {predict_time:.4f}")
    print(f"Total runtime (s): {total_runtime:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"Model saved at:    {model_path}")
