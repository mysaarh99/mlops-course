# baseline_cnn.py
# -*- coding: utf-8 -*-
import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ثباتية (اختياري)
tf.random.set_seed(42)
np.random.seed(42)

# إعدادات موحّدة
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# تحميل البيانات (npz)
train_data = np.load("data/cifar10/train.npz")
test_data  = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test,  y_test  = test_data["x"],  test_data["y"]

# بناء نفس المعمارية
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

overall_start = time.time()
t0 = time.time()
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test), verbose=1
)
train_time = time.time() - t0

t1 = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
predict_time = time.time() - t1

overall_end = time.time()
total_runtime = overall_end - overall_start

os.makedirs("models", exist_ok=True)
model_path = "models/cifar10_model_baseline.keras"
model.save(model_path)

print("✅ Baseline (no tracking)")
print(f"Train time (s):   {train_time:.4f}")
print(f"Predict time (s): {predict_time:.4f}")
print(f"Total runtime (s):{total_runtime:.4f}")
print(f"Test Accuracy:    {test_acc:.4f}")
print(f"Test Loss:        {test_loss:.4f}")
print(f"Model saved at:   {model_path}")
