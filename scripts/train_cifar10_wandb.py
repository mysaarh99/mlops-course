# ========================
# 📦 استيراد المكتبات اللازمة
# ========================
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import wandb
import shutil

# ========================
# 🚀 بدء جلسة W&B لتتبع التجربة
# ========================
wandb.init(
    project="mlops_cifar10",                 # اسم المشروع في منصة W&B
    name="cnn_cifar10_wandb_final"           # اسم التجربة الحالية (Run)
)

# إعداد معلمات التجربة (يمكن تعديلها لاحقًا أو استخدامها في sweeps)
config = wandb.config
config.epochs = 5                            # عدد التكرارات (epochs)
config.batch_size = 64                       # حجم الدفعة (batch size)
config.learning_rate = 0.001                 # معدل التعلم

# ========================
# 📥 تحميل بيانات CIFAR-10 المحفوظة مسبقًا
# ========================
# البيانات يجب أن تكون محفوظة كـ train.npz و test.npz داخل مجلد data/cifar10
train_data = np.load("data/cifar10/train.npz")
test_data = np.load("data/cifar10/test.npz")

# استخراج الصور والتسميات
x_train, y_train = train_data["x"], train_data["y"]
x_test, y_test = test_data["x"], test_data["y"]

# ========================
# 🧠 تعريف نموذج CNN باستخدام Keras
# ========================
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),             # إدخال صورة بحجم 32x32x3
    layers.Conv2D(32, (3, 3), activation='relu'),# طبقة Conv بـ 32 فلتر
    layers.MaxPooling2D((2, 2)),                 # Max Pooling لتقليل الأبعاد
    layers.Conv2D(64, (3, 3), activation='relu'),# طبقة Conv ثانية بـ 64 فلتر
    layers.MaxPooling2D((2, 2)),                 # Max Pooling أخرى
    layers.Flatten(),                            # تحويل 3D إلى 1D
    layers.Dense(64, activation='relu'),         # طبقة مخفية Fully Connected
    layers.Dense(10, activation='softmax')       # إخراج بـ 10 فئات (أصناف الصور)
])

# ========================
# ⚙️ تهيئة النموذج (الضياع + الأمثلية + المقاييس)
# ========================
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",  # خسارة التصنيف لفئات مفردة
    metrics=["accuracy"]                     # تتبع دقة النموذج
)

# ========================
# 🏋️‍♂️ تدريب النموذج مع تسجيل الأداء في كل Epoch
# ========================
for epoch in range(config.epochs):
    history = model.fit(
        x_train, y_train,
        epochs=1,                            # تدريب على Epoch واحد كل مرة
        batch_size=config.batch_size,       # حجم الدفعة
        validation_data=(x_test, y_test),   # بيانات التحقق أثناء التدريب
        verbose=1                           # عرض تقدم التدريب
    )

    # استخراج نتائج هذه الـ Epoch
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    # تسجيل النتائج في لوحة W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

# ========================
# ✅ تقييم النموذج على مجموعة الاختبار بعد التدريب
# ========================
test_loss, test_acc = model.evaluate(x_test, y_test)
wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})

# ========================
# 💾 حفظ النموذج المدرب
# ========================
os.makedirs("models", exist_ok=True)
model_path = "models/cifar10_model_wandb.keras"
model.save(model_path)

# ========================
# 📦 رفع النموذج إلى W&B كـ Artifact (نسخة)
# ========================
# نقوم بنسخ النموذج إلى مجلد W&B الحالي لتجنّب مشاكل symlink في Windows
wandb_artifact_path = os.path.join(wandb.run.dir, "cifar10_model_wandb.keras")
shutil.copy(model_path, wandb_artifact_path)

# تأكيد انتهاء التدريب
print(f"✅ Training complete (W&B). Test Accuracy: {test_acc:.4f}")
