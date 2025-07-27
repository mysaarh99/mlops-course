# ========================
# 📦 استيراد المكتبات
# ========================
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import wandb
import shutil
import random  # لاختيار عينات عشوائية

# ========================
# 🚀 بدء جلسة W&B لتتبع التدريب بصريًا
# ========================
wandb.init(
    project="mlops_cifar10",                      # اسم المشروع في W&B
    name="cnn_cifar10_wandb_visual"               # اسم التجربة
)

# إعداد معلمات التجربة
config = wandb.config
config.epochs = 5                                 # عدد Epochs
config.batch_size = 64                            # حجم كل Batch
config.learning_rate = 0.001                      # معدل التعلم
config.num_visual_samples = 10                    # عدد الصور التي سنعرضها بصريًا في كل Epoch

# ========================
# 📥 تحميل بيانات CIFAR-10 المخزنة محليًا
# ========================
train_data = np.load("data/cifar10/train.npz")    # تحميل بيانات التدريب
test_data = np.load("data/cifar10/test.npz")      # تحميل بيانات الاختبار
x_train, y_train = train_data["x"], train_data["y"]
x_test, y_test = test_data["x"], test_data["y"]

# ========================
# 🏷️ أسماء الفئات في CIFAR-10 (للإظهار البصري)
# ========================
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ========================
# 🧠 تعريف نموذج CNN باستخدام Keras
# ========================
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),                  # صورة 32x32x3
    layers.Conv2D(32, (3, 3), activation='relu'),     # طبقة Convolution بـ 32 فلتر
    layers.MaxPooling2D((2, 2)),                      # طبقة Pooling
    layers.Conv2D(64, (3, 3), activation='relu'),     # طبقة Convolution ثانية
    layers.MaxPooling2D((2, 2)),                      # طبقة Pooling ثانية
    layers.Flatten(),                                 # تحويل الناتج إلى متجه
    layers.Dense(64, activation='relu'),              # طبقة مخفية
    layers.Dense(10, activation='softmax')            # إخراج بـ 10 فئات
])

# ========================
# ⚙️ تهيئة النموذج (خسارة - أمثلية - المقاييس)
# ========================
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ========================
# 🏋️‍♂️ التدريب مع تتبع الأداء والبصرية باستخدام W&B
# ========================
for epoch in range(config.epochs):
    # تدريب على Epoch واحد فقط
    history = model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=config.batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # استخراج الأداء
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    # تسجيل المقاييس في W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    # ========================
    # 👁️ تتبع بصري - عينات من الاختبار
    # ========================
    sample_indices = random.sample(range(len(x_test)), config.num_visual_samples)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices].flatten()

    predictions = model.predict(sample_images)
    pred_labels = np.argmax(predictions, axis=1)

    wandb_images = []
    for img, true_label, pred_label in zip(sample_images, sample_labels, pred_labels):
        caption = f"True: {classes[true_label]}, Pred: {classes[pred_label]}"
        wandb_images.append(wandb.Image(img, caption=caption))

    # تسجيل الصور إلى W&B لكل Epoch
    wandb.log({f"visual_samples_epoch_{epoch+1}": wandb_images})

# ========================
# ✅ التقييم النهائي للنموذج
# ========================
test_loss, test_acc = model.evaluate(x_test, y_test)
wandb.log({
    "test_accuracy": test_acc,
    "test_loss": test_loss
})

# ========================
# 💾 حفظ النموذج النهائي
# ========================
os.makedirs("models", exist_ok=True)
model_path = "models/cifar10_model_wandb_visual.keras"
model.save(model_path)

# ========================
# 🗃️ حفظ نسخة من النموذج داخل مجلد W&B
# ========================
wandb_artifact_path = os.path.join(wandb.run.dir, "cifar10_model_wandb_visual.keras")
shutil.copy(model_path, wandb_artifact_path)

# ✅ انتهاء التدريب
print(f"✅ Visual training complete (W&B). Test Accuracy: {test_acc:.4f}")
