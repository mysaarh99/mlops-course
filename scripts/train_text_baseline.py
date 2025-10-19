# scripts/train_text_baseline.py
import os, time, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "text_baseline_model.pkl")
VECT_PATH  = os.path.join(MODEL_DIR, "text_baseline_vectorizer.pkl")

def main():
    overall_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) تحميل وتجهيز البيانات
    df = pd.read_csv(DATA_PATH)
    X = df["text"].fillna("")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) تدريب: Vectorizer + Model
    vec = TfidfVectorizer()
    t0 = time.time()
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(Xtr, y_train)
    train_time = time.time() - t0
    print(f"⏱️ Training time: {train_time:.4f} s")

    # 3) تقييم سريع
    t1 = time.time()
    Xte = vec.transform(X_test)
    y_pred = clf.predict(Xte)
    pred_time = time.time() - t1
    print(f"⏱️ Predict time: {pred_time:.4f} s")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 4) حفظ المخرجات كما يطلب dvc.yaml
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VECT_PATH)
    print(f"💾 Saved model -> {MODEL_PATH}")
    print(f"💾 Saved vectorizer -> {VECT_PATH}")

    print(f"⏱️ Total time: {time.time() - overall_start:.4f} s")

if __name__ == "__main__":
    main()
