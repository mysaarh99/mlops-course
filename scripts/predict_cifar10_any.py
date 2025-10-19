# scripts/predict_cifar10_any.py
# -*- coding: utf-8 -*-
import os, time, json, argparse, numpy as np, tensorflow as tf
from datetime import datetime

CAND_IMG_MODELS = [
    ("models/cifar10_model_baseline.keras", "baseline"),
    ("models/cifar10_model_mlflow.keras",   "mlflow"),
    ("models/cifar10_model_wandb.keras",    "wandb"),
]

def pick_available_image_model():
    for m, name in CAND_IMG_MODELS:
        if os.path.exists(m):
            return m, name
    return None, None

def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ لم أعثر على {path} — شغّل مرحلة التحضير أولًا.")
    d = np.load(path)
    x, y = d["x"], d["y"].reshape(-1)
    x = x.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="in_npz",   default="data/cifar10/test.npz")
    ap.add_argument("--out",  dest="out_npy",  default="predictions/cifar10_any_preds.npy")
    ap.add_argument("--meta", dest="meta_json", default="predictions/cifar10_any_meta.json")
    ap.add_argument("--sample", type=int, default=0,
                    help="0 لاستخدام كل بيانات الاختبار، >0 لاختيار N عيّنات عشوائيًا")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)

    model_path, tag = pick_available_image_model()
    if not model_path:
        raise FileNotFoundError(
            "❌ لم أعثر على أي نموذج صور داخل models/ "
            "(baseline/mlflow/wandb). درّب أحدها أولًا."
        )

    x_test, y_test = load_npz(args.in_npz)
    if args.sample and args.sample > 0:
        idxs = np.random.choice(len(x_test), args.sample, replace=False)
        x, y = x_test[idxs], y_test[idxs]
    else:
        x, y = x_test, y_test

    model = tf.keras.models.load_model(model_path)

    t0 = time.time()
    probs = model.predict(x, verbose=0)
    duration = time.time() - t0

    pred_labels = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # احفظ التنبؤات كـ .npy (يمكنك تغييره لـ CSV إن رغبت)
    np.save(args.out_npy, pred_labels)

    acc = float((pred_labels == y).mean()) if len(y) else None
    meta = {
        "chosen_model_tag": tag,
        "model_path": model_path,
        "n_images": int(len(y)),
        "accuracy": acc,
        "avg_confidence": float(confidences.mean()) if len(confidences) else None,
        "duration_sec": float(duration),
        "avg_sec_per_image": float(duration/len(y)) if len(y) else None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(args.meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ استخدمت نموذج: {tag} → {model_path}")
    print(f"📄 التنبؤات: {args.out_npy}")
    print(f"🧾 الميتاداتا: {args.meta_json}")

if __name__ == "__main__":
    main()
