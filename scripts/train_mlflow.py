# -*- coding: utf-8 -*-
import os, time, json, platform, socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import sklearn

import mlflow
import mlflow.sklearn
import joblib

overall_start = time.time()
DATA_PATH = "data/dataset.csv"
EXPERIMENT_NAME = "Text_MLflow_Baseline_Same_As_First"
RANDOM_STATE = 42

mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("")
X, y = df["text"], df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=200)
)

with mlflow.start_run() as run:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # ⭐ احفظ المخرجات التي يتوقعها dvc.yaml
    os.makedirs("models", exist_ok=True)
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    joblib.dump(clf, "models/text_mlflow_model.pkl")
    joblib.dump(vec, "models/text_mlflow_vectorizer.pkl")
    print("💾 Saved models/text_mlflow_model.pkl")
    print("💾 Saved models/text_mlflow_vectorizer.pkl")

    # التنبؤ والتقييم
    t1 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t1

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

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

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    mlflow.log_artifact("artifacts/classification_report.txt")

    import matplotlib
    matplotlib.use("Agg")  # لتفادي مشاكل واجهات العرض في الدوكر/CI
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Baseline (TF-IDF + LR)")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks); plt.yticks(ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=150)
    plt.close()
    mlflow.log_artifact("artifacts/confusion_matrix.png")

    with open("artifacts/summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "train_time_sec": train_time,
            "predict_time_sec": predict_time,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test))
        }, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact("artifacts/summary.json")

    # لا يزال تسجيل البايبلاين في MLflow مفيدًا
    mlflow.sklearn.log_model(sk_model=model, artifact_path="baseline_model")

    overall_end = time.time()
    mlflow.log_metric("total_runtime_sec", overall_end - overall_start)

    print("========== RUN INFO ==========")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Run ID     : {run.info.run_id}")
    print(f"Accuracy   : {acc:.4f}")
    print("==============================")
