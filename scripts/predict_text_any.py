# scripts/predict_text_any.py
# -*- coding: utf-8 -*-
import os, time, json, csv, joblib, argparse
from datetime import datetime

CAND_MODELS = [
    ("models/text_baseline_model.pkl",  "models/text_baseline_vectorizer.pkl",  "baseline"),
    ("models/text_mlflow_model.pkl",    "models/text_mlflow_vectorizer.pkl",    "mlflow"),
    ("models/text_wandb_model.pkl",     "models/text_wandb_vectorizer.pkl",     "wandb"),
]

def pick_available_model():
    for m, v, name in CAND_MODELS:
        if os.path.exists(m) and os.path.exists(v):
            return m, v, name
    return None, None, None

def read_input_texts(path: str | None):
    if path and os.path.exists(path):
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)
        return texts
    # fallback افتراضي (غير مفضل علميًا، لكن عمليًا لو ما وفّرت ملف)
    return [
        "NASA is preparing a new mission to Mars.",
        "The local hockey team won their championship game.",
        "Discussions about atheism often become controversial."
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", help="مسار ملف نصوص (سطر لكل نص)", default=None)
    ap.add_argument("--out", "-o", help="مسار CSV للإخراج", default="predictions/text_any_output.csv")
    ap.add_argument("--meta", "-m", help="مسار JSON للميتاداتا", default="predictions/text_any_meta.json")
    args = ap.parse_args()

    model_path, vect_path, tag = pick_available_model()
    if not model_path:
        raise FileNotFoundError(
            "❌ لم أعثر على أي نموذج نصي أو vectorizer.\n"
            "تأكد من تشغيل مراحل التدريب (baseline/mlflow/wandb) وأن الملفات موجودة داخل models/."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    texts = read_input_texts(args.input)

    t0 = time.time()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    probs = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        probs = p.max(axis=1)

    duration = time.time() - t0

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "predicted_label", "confidence"])
        for i, txt in enumerate(texts):
            conf = float(probs[i]) if probs is not None else ""
            w.writerow([txt, int(preds[i]), conf])

    meta = {
        "chosen_model_tag": tag,
        "model_path": model_path,
        "vectorizer_path": vect_path,
        "n_texts": len(texts),
        "duration_sec": duration,
        "avg_sec_per_sample": (duration / len(texts)) if texts else None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ استخدمت نموذج: {tag} → {model_path}")
    print(f"📄 حُفِظت التنبؤات في: {args.out}")
    print(f"🧾 الميتاداتا/المقاييس في: {args.meta}")

if __name__ == "__main__":
    main()
