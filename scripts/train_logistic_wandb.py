import os
import pandas as pd
import joblib
import wandb
import shutil

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================================
# إعداد المعلمات لـ WandB
# ================================
params = {
    "test_size": 0.2,
    "ngram_range": (1, 2),
    "C": 1.0,  # معلمة Logistic Regression
    "max_features": 5000
}

wandb.init(project="mlops_text_classification", name="logistic_regression_sweep", config=params)
config = wandb.config

# ⚠️ تحويل ngram_range من list إلى tuple إذا لزم الأمر
ngram_range = tuple(config.ngram_range)

# ================================
# تحميل البيانات
# ================================
df = pd.read_csv("data/dataset.csv")
df["text"] = df["text"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=config.test_size, random_state=42
)

# ================================
# استخراج الميزات باستخدام TF-IDF
# ================================
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    stop_words="english",
    max_features=config.max_features
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ================================
# تدريب نموذج Logistic Regression
# ================================
model = LogisticRegression(C=config.C, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ================================
# تقييم النموذج
# ================================
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

wandb.log({"accuracy": acc})
print(f"✅ Params: {dict(config)}")
print(f"✅ Accuracy: {acc:.4f}")

# ================================
# حفظ النموذج والـ Vectorizer
# ================================
# ================================
# حفظ النموذج والـ Vectorizer (بعد التأكد أنهما مدربان)
# ================================
assert hasattr(vectorizer, "idf_"), "Vectorizer is not fitted before saving!"

os.makedirs("models", exist_ok=True)

model_path = f"models/logistic_model_C{config.C}.pkl"
best_model_path = "models/logistic_model_best.pkl"
vectorizer_path = "models/vectorizer_trained.pkl"

joblib.dump(model, model_path)
joblib.dump(model, best_model_path)
joblib.dump(vectorizer, vectorizer_path)

artifact = wandb.Artifact("logistic_regression_model", type="model")
artifact.add_file(model_path)
artifact.add_file(best_model_path)
artifact.add_file(vectorizer_path)

wandb.save(best_model_path)
wandb.log_artifact(artifact)

print(f"✅ Model saved at {best_model_path}")
print(f"✅ Vectorizer saved at {vectorizer_path}")

# os.makedirs("models", exist_ok=True)
# model_path = f"models/logistic_model_C{config.C}.pkl"
# best_model_path = "models/logistic_model_best.pkl"
# vectorizer_path = "models/vectorizer.pkl"

# # حفظ النموذج والـ Vectorizer
# joblib.dump(model, model_path)
# joblib.dump(model, best_model_path)  # نسخة موحدة لسهولة الوصول
# joblib.dump(vectorizer, vectorizer_path)

# # تسجيل Artifact مع WandB
# artifact = wandb.Artifact("logistic_regression_model", type="model")
# artifact.add_file(model_path)
# artifact.add_file(best_model_path)
# artifact.add_file(vectorizer_path)

# wandb.save(best_model_path)
# wandb.log_artifact(artifact)

# print(f"✅ Model saved at {best_model_path}")
# print(f"✅ Vectorizer saved at {vectorizer_path}")



