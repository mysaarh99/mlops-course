import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import mlflow
import mlflow.keras

# 🧪 إعداد تجربة MLflow
# سيتم إنشاء تجربة باسم "CIFAR10_CNN_MLflow" (أو استخدامها إن كانت موجودة)
mlflow.set_experiment("CIFAR10_CNN_MLflow")

# ⚙️ تعريف إعدادات التدريب
epochs = 5
batch_size = 64
learning_rate = 0.001

# 📦 تحميل بيانات CIFAR-10 من ملفات محفوظة مسبقًا
# يجب أن تكون ملفات train.npz و test.npz موجودة داخل مجلد data/cifar10
train_data = np.load("data/cifar10/train.npz")
test_data = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test, y_test = test_data["x"], test_data["y"]

# 🧠 بناء نموذج CNN بسيط باستخدام Keras
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),                # حجم الصورة (32x32 RGB)
    layers.Conv2D(32, (3, 3), activation='relu'),   # طبقة التفاف (Conv)
    layers.MaxPooling2D((2, 2)),                    # طبقة تجميع (Pooling)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                               # تحويل الأبعاد إلى خطية
    layers.Dense(64, activation='relu'),            # طبقة مخفية Fully Connected
    layers.Dense(10, activation='softmax')          # طبقة إخراج بـ 10 فئات
])

# ✅ تجميع النموذج
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 🚀 بدء تسجيل التجربة داخل MLflow
with mlflow.start_run():
    # 📝 تسجيل المعلمات الرئيسية للتجربة
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    # 📁 تسجيل بيانات التدريب كـ Artifact (تُربط بالتجربة)
    mlflow.log_artifact("data/cifar10/train.npz", artifact_path="dataset")
    mlflow.log_artifact("data/cifar10/test.npz", artifact_path="dataset")

    # 📊 إنشاء قوائم لتخزين الأداء بهدف تحليل لاحق (اختياري)
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = []

    # 🔁 تدريب النموذج وتسجيل الأداء لكل Epoch
    for epoch in range(epochs):
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # استخراج القيم في نهاية الـ Epoch
        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        # حفظ الأداء في القوائم (اختياري)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # تسجيل الأداء داخل MLflow باستخدام step
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch + 1)

    # 🧪 التقييم النهائي بعد التدريب
    test_loss, test_acc = model.evaluate(x_test, y_test)
    mlflow.log_metrics({
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })

    # 💾 حفظ النموذج محليًا داخل مجلد models
    os.makedirs("models", exist_ok=True)
    model_path = "models/cifar10_model_mlflow.keras"
    model.save(model_path)

    # 🗃️ رفع النموذج إلى MLflow كـ Artifact
    mlflow.keras.log_model(model, "cifar10_cnn_model")

    print(f"✅ Training complete (MLflow). Test Accuracy: {test_acc:.4f}")
