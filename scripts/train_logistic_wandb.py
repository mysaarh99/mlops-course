# -*- coding: utf-8 -*-
"""
Baseline TF-IDF + LogisticRegression with Weights & Biases tracking.
Produces DVC outs:
  - models/text_wandb_model.pkl
  - models/text_wandb_vectorizer.pkl
"""

import os
import io
import time
import json
import socket
import platform
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # مهم داخل Docker/CI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
import sklearn
import joblib
import wandb


# ================== إعدادات عامة ==================
DATA_PATH     = "data/dataset.csv"
PROJECT       = os.getenv("WANDB_PROJECT", "text-baseline-wandb")
RUN_NAME      = "tfidf-logreg-wandb"
RANDOM_STATE  = 42

MODEL_DIR     = "models"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = os.path.join(MODEL_DIR, "text_wandb_model.pkl")
VECT_PATH     = os.path.join(MODEL_DIR, "text_wandb_vectorizer.pkl")


def main():
    overall_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1) Load & prepare data
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("")
    X, y = df["text"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 2) Build model
    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=200)  # lbfgs افتراضيًا
    )

    # 3) W&B run (لو ما فيه API key، سيطبع تحذير ويتخطّى)
    try:
        wandb.login()  # يستخدم WANDB_API_KEY إن كان متوفرًا
        wandb_enabled = True
    except Exception:
        wandb_enabled = False

    with wandb.init(
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
        reinit=True,
        mode=("online" if wandb_enabled else "disabled"),
    ) as run:

        # 4) Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # 5) Predict & metrics
        t1 = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - t1

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report_text = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y)

        # 6) Save EXACT outs for DVC
        vec = model.named_steps["tfidfvectorizer"]
        clf = model.named_steps["logisticregression"]
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(vec, VECT_PATH)
        print(f"💾 Saved {MODEL_PATH}")
        print(f"💾 Saved {VECT_PATH}")

        # 7) Log metrics to W&B
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

        # 8) Confusion matrix image
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix - TFIDF + LR")
        plt.colorbar()
        ticks = np.arange(len(classes))
        plt.xticks(ticks, classes, rotation=45)
        plt.yticks(ticks, classes)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        if wandb_enabled:
            wandb.log({"confusion_matrix_img": wandb.Image(cm_path)})

        # 9) Save & log report
        report_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 10) Log artifacts (الطريقة الصحيحة)
        if wandb_enabled:
            art = wandb.Artifact(
                name="baseline_tfidf_logreg_wandb",
                type="model",
                description="Baseline TF-IDF + LogisticRegression (max_iter=200)",
                metadata={"framework": "sklearn", "max_iter": 200},
            )
            # حفظ البايبلاين الكامل كمرجع (اختياري)
            pipe_path = os.path.join(ARTIFACTS_DIR, "pipeline.pkl")
            with open(pipe_path, "wb") as f:
                pickle.dump(model, f)

            art.add_file(pipe_path)
            art.add_file(report_path)
            art.add_file(cm_path)
            wandb.log_artifact(art)

        total_runtime = time.time() - overall_start
        wandb.summary["total_runtime_sec"] = float(total_runtime)
        print(f"⏱️ Total runtime: {total_runtime:.4f}s")


if __name__ == "__main__":
    main()
