# سكربت التنبؤ بالنموذج الافضل
import tensorflow as tf
import numpy as np

# تحميل النموذج
model = tf.keras.models.load_model("models/cifar10_model.keras")

# تحميل بيانات الاختبار
test_data = np.load("data/cifar10/test.npz")
x_test, y_test = test_data["x"], test_data["y"]

# اختيار 5 صور عشوائية للتنبؤ
idxs = np.random.choice(len(x_test), 5, replace=False)
images, labels = x_test[idxs], y_test[idxs]

# إجراء التنبؤ
preds = model.predict(images)
pred_labels = np.argmax(preds, axis=1)

# أسماء الفئات في CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# عرض النتائج
for i, idx in enumerate(idxs):
    print("-----------")
    print(f"التصنيف الحقيقي: {classes[labels[i][0]]}")
    print(f"التصنيف المتوقع: {classes[pred_labels[i]]}")
