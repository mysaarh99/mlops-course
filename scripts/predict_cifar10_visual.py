import tensorflow as tf
import numpy as np
import random
import wandb
import os

# إعداد W&B لجلسة التقرير
wandb.init(project="mlops_cifar10", name="cifar10_final_visual_report")

# تحميل النموذج النهائي
model_path = "models/cifar10_model_wandb_visual.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ لم يتم العثور على النموذج: {model_path}. تأكد أنك شغّلت مرحلة التدريب أولاً.")

model = tf.keras.models.load_model(model_path)

# تحميل بيانات الاختبار
test_data = np.load("data/cifar10/test.npz")
x_test, y_test = test_data["x"], test_data["y"]

# أسماء الفئات
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# اختيار 20 صورة عشوائية
sample_indices = random.sample(range(len(x_test)), 20)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices].flatten()

# تنبؤات النموذج
predictions = model.predict(sample_images)
pred_labels = np.argmax(predictions, axis=1)

# رفع الصور إلى W&B مع العناوين
wandb_images = []
for img, true_label, pred_label in zip(sample_images, sample_labels, pred_labels):
    caption = f"True: {classes[true_label]}, Pred: {classes[pred_label]}"
    wandb_images.append(wandb.Image(img, caption=caption))

wandb.log({"final_visual_report": wandb_images})

print("✅ تم إنشاء تقرير بصري للنموذج النهائي (20 صورة) ورفعه إلى W&B.")
