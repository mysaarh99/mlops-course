import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# إعداد المعلمات
epochs = 5
batch_size = 64
learning_rate = 0.001

# تحميل بيانات CIFAR-10 من الملفات المحضرة
train_data = np.load("data/cifar10/train.npz")
test_data = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test, y_test = test_data["x"], test_data["y"]

# بناء نموذج CNN بسيط
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# تدريب النموذج
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(x_test, y_test), verbose=1)

# التقييم النهائي
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# التأكد من وجود مجلد models
os.makedirs("models", exist_ok=True)

# حفظ النموذج
model_path = "models/cifar10_model.keras"
model.save(model_path)

print(f"✅ CIFAR-10 model trained and saved at {model_path}")
print(f"Test Accuracy: {test_acc:.4f}")
