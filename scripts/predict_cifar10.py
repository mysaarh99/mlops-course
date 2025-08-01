# سكربت التنبؤ باستخدام النموذج الأفضل المحفوظ لـ CIFAR-10
import tensorflow as tf
import numpy as np

# تحميل نموذج الـ CNN الذي تم تدريبه مسبقًا وحفظه
model = tf.keras.models.load_model("models/cifar10_model.keras")

# تحميل بيانات الاختبار المحفوظة مسبقًا
test_data = np.load("data/cifar10/test.npz")
x_test, y_test = test_data["x"], test_data["y"]

# اختيار 5 صور عشوائية من بيانات الاختبار (بدون تكرار)
idxs = np.random.choice(len(x_test), 2, replace=False)
images, labels = x_test[idxs], y_test[idxs]

# التنبؤ بالفئات لكل صورة مختارة
preds = model.predict(images)

# الحصول على الفئة ذات الاحتمال الأعلى لكل صورة (التصنيف المتوقع)
pred_labels = np.argmax(preds, axis=1)

# أسماء الفئات في مجموعة بيانات CIFAR-10 (مرتبة حسب الفئة الرقمية)
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# طباعة التصنيف الحقيقي والمتوقع لكل صورة
for i, idx in enumerate(idxs):
    print("-----------")
    print(f"التصنيف الحقيقي: {classes[labels[i][0]]}")  # الفئة الصحيحة من البيانات
    print(f"التصنيف المتوقع: {classes[pred_labels[i]]}")  # نتيجة النموذج
