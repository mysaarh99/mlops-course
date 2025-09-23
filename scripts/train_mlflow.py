# -*- coding: utf-8 -*-
import os, time, json, platform, socket
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

import mlflow
import mlflow.sklearn

overall_start = time.time()
# ======= إعدادات مطابقة للنموذج الأول =======
DATA_PATH = "data/dataset.csv"        # نفس الملف المستخدم أولاً
EXPERIMENT_NAME = "Text_MLflow_Baseline_Same_As_First"
RANDOM_STATE = 42

# (اختياري) لو عندك خادم MLflow:
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(EXPERIMENT_NAME)

# ======= تحميل وتجهيز البيانات (نفس الخطوات) =======
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("")
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ======= نفس النموذج تمامًا =======
model = make_pipeline(
    TfidfVectorizer(),                 # الإعدادات الافتراضية (unigram)
    LogisticRegression(max_iter=200)   # نفس max_iter
)

# ======= إضافة قياس الزمن الكلي =======


# ======= تشغيل جلسة MLflow وتسجيل كل شيء =======
with mlflow.start_run() as run:
    # لا نستخدم autolog لتجنّب أي اختلافات تلقائية
    # mlflow.sklearn.autolog(log_models=False)

    # قياس وقت التدريب (fit واحد فقط)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # التنبؤ وقياس الوقت
    t1 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t1

    # المقاييس (نفس ما كنت تحسبه وأكثر قليلًا)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # تسجيل المعلمات الأساسية فقط (مطابقة للنموذج الأول)
    tfidf = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    mlflow.log_params({
        "vectorizer": "TfidfVectorizer(defaults)",
        "clf": "LogisticRegression",
        "max_iter": clf.max_iter,
        "random_state": RANDOM_STATE,
        "dataset": os.path.basename(DATA_PATH),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
    })

    # تسجيل المقاييس
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_classes": int(len(np.unique(y))),
    })

    # حفظ الأرتيفاكتس (تقرير + مصفوفة الالتباس + ملخّص)
    os.makedirs("artifacts", exist_ok=True)

    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    mlflow.log_artifact(report_path)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Baseline (TF-IDF + LR)")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks); plt.yticks(ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)

    summary = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test))
    }
    summary_path = "artifacts/summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact(summary_path)

    # حفظ النموذج نفسه كأرتيفاكت
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="baseline_model"   # لا Model Registry هنا؛ مجرد أرتيفاكت
    )

    # ======= حساب الزمن الكلي =======
    overall_end = time.time()
    total_runtime = overall_end - overall_start
    mlflow.log_metric("total_runtime_sec", total_runtime)

    # مخرجات مفيدة في الطرفية
    print("========== RUN INFO ==========")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Run ID     : {run.info.run_id}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1-macro   : {f1_macro:.4f}")
    print(f"Train time : {train_time:.4f} sec")
    print(f"Predict    : {predict_time:.4f} sec")
    print(f"Total run  : {total_runtime:.4f} sec")
    print("==============================")
