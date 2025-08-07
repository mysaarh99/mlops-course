import tensorflow as tf
import numpy as np
import os

# دالة لتحضير وتحميل مجموعة بيانات CIFAR-10
def prepare_cifar10():

    train_path = "data/cifar10/train.npz"
    test_path = "data/cifar10/test.npz"

    # إذا كانت البيانات موجودة مسبقًا، لا تقم بإعادة التحميل
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("✅ CIFAR-10 dataset already exists. Skipping download.")
        return

    print("📥 Downloading and preparing CIFAR-10 dataset...")

    # تحميل بيانات CIFAR-10 (صور تدريب + تسميات، صور اختبار + تسميات)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # تطبيع قيم البيكسل إلى مجال [0, 1] بدلًا من [0, 255]
    # هذا يسهل تدريب النموذج ويزيد من كفاءته
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # إنشاء مجلد 'data/cifar10' إذا لم يكن موجودًا مسبقًا
    os.makedirs("data/cifar10", exist_ok=True)

    # حفظ بيانات التدريب مضغوطة في ملف train.npz داخل المجلد
    np.savez_compressed("data/cifar10/train.npz", x=x_train, y=y_train)

    # حفظ بيانات الاختبار مضغوطة في ملف test.npz داخل المجلد
    np.savez_compressed("data/cifar10/test.npz", x=x_test, y=y_test)

    # طباعة رسالة تأكيد الانتهاء بنجاح
    print("✅ CIFAR-10 dataset prepared and saved to data/cifar10/")

# تشغيل دالة التحضير فقط إذا كان الملف هو البرنامج الرئيسي الجاري تشغيله
if __name__ == "__main__":
    prepare_cifar10()
