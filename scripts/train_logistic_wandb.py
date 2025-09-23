# -*- coding: utf-8 -*-
# Baseline TF-IDF + LogisticRegression with W&B tracking (same as first model)

import os, time, json, platform, socket, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

import sklearn
import wandb

import time
overall_start = time.time()  # ⏱️ بداية الزمن الكلي

# ====== إعدادات مطابقة للنموذج الأول ======
DATA_PATH = "data/dataset.csv"          # غيّرها إذا لزم
PROJECT   = "text-baseline-wandb"       # اسم المشروع في W&B
RUN_NAME  = "tfidf-logreg-baseline"     # اسم الـ run
RANDOM_STATE = 42

# ====== تحميل وتجهيز البيانات ======
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("")
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ====== نفس النموذج تمامًا ======
model = make_pipeline(
    TfidfVectorizer(),                 # الإعدادات الافتراضية (unigram)
    LogisticRegression(max_iter=200)   # نفس max_iter
)

# ====== بدء جلسة W&B وتسجيل الإعدادات ======
wandb.init(
    project=PROJECT,
    name=RUN_NAME,
    config={
        "vectorizer": "TfidfVectorizer(defaults)",
        "classifier": "LogisticRegression",
        "max_iter": 200,
        "random_state": RANDOM_STATE,
        "dataset": os.path.basename(DATA_PATH),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
    },
)

# ====== تدريب وقياس الوقت ======
t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0

# ====== تنبؤ وقياس الوقت ======
t1 = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - t1

# ====== المقاييس ======
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
report_text = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y)

print(f"start runtime (s): {overall_start:.4f}")

# ====== تسجيل المقاييس إلى W&B ======
wandb.log({
    "accuracy": acc,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
    "train_time_sec": train_time,
    "predict_time_sec": predict_time,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "n_classes": int(len(classes)),
})

# (اختياري) رفع مصفوفة الالتباس كـ صورة + مخطط تفاعلي
# 1) صورة ثابتة
os.makedirs("artifacts", exist_ok=True)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Baseline (TF-IDF + LR)")
plt.colorbar()
ticks = np.arange(len(classes))
plt.xticks(ticks, classes, rotation=45)
plt.yticks(ticks, classes)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
cm_path = "artifacts/confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
wandb.log({"confusion_matrix_img": wandb.Image(cm_path)})

# 2) مخطط تفاعلي داخل W&B
# ====== مصفوفة الالتباس ======
# مهم: نحول y_test و y_pred لقوائم عشان ما يصير KeyError
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test.tolist(),
        preds=y_pred.tolist(),
        class_names=[str(c) for c in classes]
    )
})

# ====== حفظ تقرير التصنيف ======
report_path = "artifacts/classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

report_artifact = wandb.Artifact(
    name="classification_report",
    type="report",
    description="Classification report for baseline model"
)

report_artifact.add_file(report_path)
wandb.log_artifact(report_artifact)

# ====== حفظ النموذج ======
model_path = "artifacts/baseline_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
artifact = wandb.Artifact(
    name="baseline_tfidf_logreg",
    type="model",
    description="Baseline TF-IDF + LogisticRegression (max_iter=200)",
    metadata={"framework": "sklearn", "max_iter": 200}
)
artifact.add_file(model_path)
wandb.log_artifact(artifact)

# ====== حساب الزمن الكلي ======
overall_end = time.time()
total_runtime = overall_end - overall_start
print(f"Total runtime (s): {total_runtime:.4f}")

# ====== تسجيل المقاييس النهائية في summary ======
wandb.summary["accuracy"] = float(acc)
wandb.summary["f1_macro"] = float(f1_macro)
wandb.summary["f1_weighted"] = float(f1_weighted)
wandb.summary["train_time_sec"] = float(train_time)
wandb.summary["predict_time_sec"] = float(predict_time)
wandb.summary["train_size"] = int(len(X_train))
wandb.summary["test_size"] = int(len(X_test))
wandb.summary["n_classes"] = int(len(classes))
wandb.summary["total_runtime_sec"] = float(total_runtime)

wandb.finish()
