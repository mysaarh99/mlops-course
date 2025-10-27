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

# ========= مؤثرات قوية مع افتراضات معقولة =========
# ملاحظات:
# - لم نستخدم stop_words='english' لأن لديك محتوى متعدد اللغات غالباً.
# - class_weight='balanced' مفيد لعدم توازن البيانات (يرفع F1_macro غالباً).
NGRAMS = tuple(map(int, os.getenv("NGRAMS", "1,2").split(",")))        # CHANGED: (1,2)
MIN_DF = int(os.getenv("MIN_DF", "2"))                                 # CHANGED: 2
MAX_DF = float(os.getenv("MAX_DF", "0.9"))                              # CHANGED: 0.9
SUBLINEAR = os.getenv("SUBLINEAR", "1") == "1"                          # CHANGED: True
MAX_FEAT = int(os.getenv("MAX_FEATURES", "50000"))                      # CHANGED: 50k (اختياري)
STRIP_ACCENTS = os.getenv("STRIP_ACCENTS", "unicode")                   # CHANGED: unicode/None
LR_C = float(os.getenv("LR_C", "2.0"))                                  # CHANGED: 2.0
LR_MAX_ITER = int(os.getenv("LR_MAX_ITER", "1000"))                     # CHANGED: 1000
LR_CLASS_WEIGHT = os.getenv("LR_CLASS_WEIGHT", "balanced")              # CHANGED: balanced/None
LR_SOLVER = os.getenv("LR_SOLVER", "lbfgs")                             # lbfgs مناسب لـ l2 والكثير من المزايا
LR_PENALTY = os.getenv("LR_PENALTY", "l2")                              # l2

df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("")
X, y = df["text"], df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

vectorizer = TfidfVectorizer(
    ngram_range=NGRAMS,            # CHANGED
    min_df=MIN_DF,                 # CHANGED
    max_df=MAX_DF,                 # CHANGED
    sublinear_tf=SUBLINEAR,        # CHANGED
    max_features=MAX_FEAT,         # CHANGED (يمكن وضع None لإزالة الحد)
    strip_accents=None if STRIP_ACCENTS.lower() == "none" else STRIP_ACCENTS
)

logreg = LogisticRegression(
    C=LR_C,                        # CHANGED
    max_iter=LR_MAX_ITER,          # CHANGED
    class_weight=None if LR_CLASS_WEIGHT.lower()=="none" else LR_CLASS_WEIGHT,  # CHANGED
    solver=LR_SOLVER,
    penalty=LR_PENALTY,
    n_jobs=None
)

model = make_pipeline(vectorizer, logreg)

run_name = (
    f"tfidf_ng{NGRAMS}_min{MIN_DF}_max{MAX_DF}_sub{int(SUBLINEAR)}_mf{MAX_FEAT}_"
    f"LR_C{LR_C}_w{LR_CLASS_WEIGHT}_it{LR_MAX_ITER}"
)

with mlflow.start_run(run_name=run_name) as run:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # مخرجات DVC (نفس الأسماء)
    os.makedirs("models", exist_ok=True)
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    joblib.dump(clf, "models/text_mlflow_model.pkl")
    joblib.dump(vec, "models/text_mlflow_vectorizer.pkl")

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
        "vectorizer": "TfidfVectorizer",
        "ngram_range": str(NGRAMS),          # CHANGED
        "min_df": MIN_DF,                    # CHANGED
        "max_df": MAX_DF,                    # CHANGED
        "sublinear_tf": SUBLINEAR,           # CHANGED
        "max_features": MAX_FEAT,            # CHANGED
        "strip_accents": STRIP_ACCENTS,      # CHANGED
        "clf": "LogisticRegression",
        "C": clf.C,                          # CHANGED
        "max_iter": clf.max_iter,            # CHANGED
        "class_weight": str(clf.class_weight),   # CHANGED
        "solver": clf.solver,
        "penalty": clf.penalty,
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
    matplotlib.use("Agg")
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - TFIDF+LR (tuned)")
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

    mlflow.sklearn.log_model(sk_model=model, artifact_path="baseline_model")

    overall_end = time.time()
    mlflow.log_metric("total_runtime_sec", overall_end - overall_start)

    print("========== RUN INFO ==========")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Run name   : {run_name}")
    print(f"Run ID     : {run.info.run_id}")
    print(f"Accuracy   : {acc:.4f} | F1_macro: {f1_macro:.4f}")
    print("==============================")
