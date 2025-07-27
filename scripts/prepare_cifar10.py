import tensorflow as tf
import numpy as np
import os
# سكربت تحضير البيانات
def prepare_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # تطبيع القيم [0,1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # حفظ البيانات في مجلد data/cifar10
    os.makedirs("data/cifar10", exist_ok=True)
    np.savez_compressed("data/cifar10/train.npz", x=x_train, y=y_train)
    np.savez_compressed("data/cifar10/test.npz", x=x_test, y=y_test)
    print("✅ CIFAR-10 dataset prepared and saved to data/cifar10/")

if __name__ == "__main__":
    prepare_cifar10()
