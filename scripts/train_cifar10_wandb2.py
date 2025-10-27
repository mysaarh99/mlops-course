# wandb_cnn.py
# -*- coding: utf-8 -*-
import os, time, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import wandb

# ثباتية (اختياري)
tf.random.set_seed(42)
np.random.seed(42)

# === إعدادات قابلة للتعديل عبر المتغيرات البيئية (لا نغيّر المعمارية الأساسية) ===
EPOCHS        = int(os.getenv("EPOCHS", 5))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 64))
LEARNING_RATE = float(os.getenv("LR", 0.001))
DROPOUT       = float(os.getenv("DROPOUT", 0.0))      # 0.0 لتعطيل الإسقاط
FILTER1       = int(os.getenv("FILTER1", 32))
FILTER2       = int(os.getenv("FILTER2", 64))
DENSE_UNITS   = int(os.getenv("DENSE_UNITS", 64))
OPTIMIZER     = os.getenv("OPTIMIZER", "adam")        # adam | sgd | rmsprop
AUGMENT       = os.getenv("AUGMENT", "0") == "1"      # 1 لتفعيل Augmentation

# تسجيل الدخول قبل التشغيل (مرة واحدة): wandb login
run = wandb.init(
    project="mlops_cifar10_fair_compare",
    name=f"cnn_cifar10_lr{LEARNING_RATE}_bs{BATCH_SIZE}_do{DROPOUT}_opt{OPTIMIZER}_aug{int(AUGMENT)}",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "filter1": FILTER1,
        "filter2": FILTER2,
        "dense_units": DENSE_UNITS,
        "optimizer": OPTIMIZER,
        "augment": AUGMENT
    }
)

# --- تحميل البيانات ---
train_data = np.load("data/cifar10/train.npz")
test_data  = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test,  y_test  = test_data["x"],  test_data["y"]

# تطبيع بسيط
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Augmentation اختياري (لا ندخله إن كان مطفأ لتجنّب طبقة فارغة)
def get_augmentation_layer():
    if not AUGMENT:
        return None
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ], name="augmentation")

# --- بناء النموذج (نفس البنية العامة مع مرونة طفيفة في الفلاتر و Dropout) ---
def build_model():
    aug = get_augmentation_layer()

    layers_list = [layers.Input(shape=(32, 32, 3))]
    if aug is not None:
        layers_list.append(aug)  # نضيف الأوغمنتيشن فقط عند تفعيله

    layers_list += [
        layers.Conv2D(FILTER1, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(FILTER2, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(DENSE_UNITS, activation='relu'),
        layers.Dropout(DROPOUT),
        layers.Dense(10, activation='softmax')
    ]

    model = models.Sequential(layers_list)

    if OPTIMIZER.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER.lower() == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model()

overall_start = time.time()

# --- تدريب ---
t0 = time.time()
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    verbose=1
)
train_time = time.time() - t0

# --- تقييم ---
t1 = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
predict_time = time.time() - t1

overall_end = time.time()
total_runtime = overall_end - overall_start

# --- تسجيل المقاييس في W&B ---
wandb.log({
    "train_time_sec": float(train_time),
    "predict_time_sec": float(predict_time),
    "total_runtime_sec": float(total_runtime),
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss)
})

# --- حفظ النموذج ورفعه كأرتيفاكت ---
os.makedirs("models", exist_ok=True)
model_path = "models/cifar10_model_wandb.keras"
model.save(model_path)

# بسيط وآمن: استخدم wandb.save مباشرة على المسار المحلّي
try:
    wandb.save(model_path)
except Exception as e:
    print(f"W&B save warning: {e}")
    # بديل احتياطي
    try:
        artifact_path = os.path.join(wandb.run.dir, os.path.basename(model_path))
        shutil.copy(model_path, artifact_path)
        wandb.save(artifact_path)
    except Exception as e2:
        print(f"Artifact copy warning: {e2}")

print("✅ W&B run complete")
print(f"Train time (s):   {train_time:.4f}")
print(f"Predict time (s): {predict_time:.4f}")
print(f"Total runtime (s):{total_runtime:.4f}")
print(f"Test Accuracy:    {test_acc:.4f}")
print(f"Test Loss:        {test_loss:.4f}")
print(f"Model saved at:   {model_path}")

wandb.finish()
