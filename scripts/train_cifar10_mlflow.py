import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import mlflow
import mlflow.keras

# إعداد MLflow
mlflow.set_experiment("CIFAR10_CNN_MLflow")

epochs = 5
batch_size = 64
learning_rate = 0.001

# تحميل البيانات
train_data = np.load("data/cifar10/train.npz")
test_data = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test, y_test = test_data["x"], test_data["y"]

# تعريف نموذج CNN
model = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
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

# تشغيل التجربة وتسجيل يدوي لكل Epoch
with mlflow.start_run():
    mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate})

    # رفع البيانات كـ Artifact للربط مع النموذج
    mlflow.log_artifact("data/cifar10/train.npz", artifact_path="dataset")
    mlflow.log_artifact("data/cifar10/test.npz", artifact_path="dataset")

    # قوائم لتخزين القيم بهدف تصديرها لاحقًا
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(epochs):
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=1
        )

        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch+1)

    # تقييم نهائي وحفظ النموذج
    test_loss, test_acc = model.evaluate(x_test, y_test)
    mlflow.log_metrics({"test_accuracy": test_acc, "test_loss": test_loss})

    os.makedirs("models", exist_ok=True)
    model_path = "models/cifar10_model_mlflow.keras"
    model.save(model_path)
    mlflow.keras.log_model(model, "cifar10_cnn_model")

    print(f"✅ Training complete (MLflow). Test Accuracy: {test_acc:.4f}")
