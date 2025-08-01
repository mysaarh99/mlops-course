import tensorflow as tf
import numpy as np
import random
import wandb
import os

# إعداد جلسة Weights & Biases لتتبع التجربة
wandb.init(project="mlops_cifar10", name="cifar10_final_visual_report")

# تحديد مسار النموذج المدرب مسبقًا
model_path = "models/cifar10_model_wandb_visual.keras"

# التحقق من وجود النموذج، إذا لم يكن موجودًا، إظهار خطأ وإيقاف التنفيذ
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ لم يتم العثور على النموذج: {model_path}. تأكد أنك شغّلت مرحلة التدريب أولاً.")

# تحميل النموذج باستخدام TensorFlow
model = tf.keras.models.load_model(model_path)

# تحميل بيانات الاختبار (صور + تسميات)
test_data = np.load("data/cifar10/test.npz")
x_test, y_test = test_data["x"], test_data["y"]

# أسماء الفئات المستخدمة في CIFAR-10 لتسهيل التفسير البصري
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# اختيار 20 صورة عشوائية من بيانات الاختبار لعرضها في التقرير البصري
sample_indices = random.sample(range(len(x_test)), 3)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices].flatten()  # جعل التسميات أحادية الأبعاد

# استخدام النموذج لتحليل الصور المختارة والحصول على التنبؤات (احتمالات الفئات)
predictions = model.predict(sample_images)
# اختيار الفئة ذات الاحتمال الأعلى لكل صورة (التنبؤ الفعلي)
pred_labels = np.argmax(predictions, axis=1)

# إعداد قائمة لتخزين الصور مع عناوين توضيحية حقيقية ومتوقعة
wandb_images = []
for img, true_label, pred_label in zip(sample_images, sample_labels, pred_labels):
    # تكوين التسمية النصية لكل صورة (الحقيقة مقابل التنبؤ)
    caption = f"True: {classes[true_label]}, Pred: {classes[pred_label]}"
    # إضافة الصورة مع التعليق إلى قائمة صور W&B
    wandb_images.append(wandb.Image(img, caption=caption))

# رفع الصور والتسميات إلى Weights & Biases كسجل بصري للتقرير النهائي
wandb.log({
    "final_visual_report": wandb_images,
    "endpoint": "predict_cifar10_visual",
    "sample_size": len(sample_images)
})
# طباعة رسالة توضح أن التقرير البصري تم إنشاؤه ورفعه بنجاح
print("✅ Final visual report (3 sample images) successfully logged to Weights & Biases.")