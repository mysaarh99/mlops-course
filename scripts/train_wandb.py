import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import wandb

# --- إعداد مشروع W&B ---
wandb.init(project="mlops_text_classification", name="naive_bayes_wandb")

# معلمات التجربة (يمكن تعديلها لاحقًا أو عبر Sweep)
params = {
    "test_size": 0.2,
    "ngram_range": (1, 2),
    "alpha": 1.0,
    "max_features": 5000
}
wandb.config.update(params)

# --- تحميل البيانات ---
df = pd.read_csv("data/dataset.csv")
# معالجة أي قيم فارغة
df["text"] = df["text"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=params["test_size"], random_state=42
)

# --- تحويل النصوص إلى ميزات ---
vectorizer = TfidfVectorizer(
    ngram_range=params["ngram_range"], stop_words="english", max_features=params["max_features"]
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- تدريب نموذج Naive Bayes ---
model = MultinomialNB(alpha=params["alpha"])
model.fit(X_train_tfidf, y_train)

# --- التقييم ---
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

# تسجيل النتائج في W&B
wandb.log({"accuracy": acc})

# تقرير التصنيف (Precision, Recall, F1)
report = classification_report(y_test, y_pred, output_dict=True)
wandb.log({"classification_report": report})

print(f"Accuracy: {acc:.4f}")

# ---

