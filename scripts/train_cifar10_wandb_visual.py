import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import wandb
import shutil
import random

# إعداد W&B مع اسم جديد لتوضيح أنه إصدار بصري
wandb.init(project="mlops_cifar10", name="cnn_cifar10_wandb_visual")
config = wandb.config
config.epochs = 5
config.batch_size = 64
config.learning_rate = 0.001
config.num_visual_samples = 10  # عدد الصور التي سنرفعها في كل Epoch

# تحميل البيانات
train_data = np.load("data/cifar10/train.npz")
test_data = np.load("data/cifar10/test.npz")
x_train, y_train = train_data["x"], train_data["y"]
x_test, y_test = test_data["x"], test_data["y"]

# أسماء الفئات في CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# نموذج CNN
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

# تدريب مع التتبع البصري
for epoch in range(config.epochs):
    # تدريب Epoch واحد
    history = model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=config.batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # استخراج القيم
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    # رفع المقاييس
    wandb.log({
        "epoch": epoch + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    # --- تتبع بصري: اختيار عينات من الاختبار ---
    sample_indices = random.sample(range(len(x_test)), config.num_visual_samples)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices].flatten()

    predictions = model.predict(sample_images)
    pred_labels = np.argmax(predictions, axis=1)

    # رفع الصور إلى W&B مع التسمية الحقيقية والتنبؤ
    wandb_images = []
    for img, true_label, pred_label in zip(sample_images, sample_labels, pred_labels):
        caption = f"True: {classes[true_label]}, Pred: {classes[pred_label]}"
        wandb_images.append(wandb.Image(img, caption=caption))

    wandb.log({f"visual_samples_epoch_{epoch+1}": wandb_images})

# تقييم وحفظ النموذج
test_loss, test_acc = model.evaluate(x_test, y_test)
wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})

os.makedirs("models", exist_ok=True)
model_path = "models/cifar10_model_wandb_visual.keras"
model.save(model_path)

# نسخ النموذج لمجلد W&B (بدون symlink)
wandb_artifact_path = os.path.join(wandb.run.dir, "cifar10_model_wandb_visual.keras")
shutil.copy(model_path, wandb_artifact_path)

print(f"✅ Visual training complete (W&B). Test Accuracy: {test_acc:.4f}")
