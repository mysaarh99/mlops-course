import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import wandb
import shutil

# إعداد W&B
wandb.init(project="mlops_cifar10", name="cnn_cifar10_wandb_final")
config = wandb.config
config.epochs = 5
config.batch_size = 64
config.learning_rate = 0.001

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

optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# تدريب مع تسجيل يدوي لكل Epoch
for epoch in range(config.epochs):
    history = model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=config.batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    wandb.log({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

# تقييم نهائي وحفظ النموذج
test_loss, test_acc = model.evaluate(x_test, y_test)
wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})

os.makedirs("models", exist_ok=True)
model_path = "models/cifar10_model_wandb.keras"
model.save(model_path)

# نسخ النموذج لمجلد W&B بدل symlink (لتفادي مشكلة Windows)
wandb_artifact_path = os.path.join(wandb.run.dir, "cifar10_model_wandb.keras")
shutil.copy(model_path, wandb_artifact_path)

print(f"✅ Training complete (W&B). Test Accuracy: {test_acc:.4f}")
