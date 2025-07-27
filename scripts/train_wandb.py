# استيراد المكتبات الأساسية
import pandas as pd  # لتحميل ومعالجة البيانات
from sklearn.model_selection import train_test_split  # لتقسيم البيانات إلى تدريب واختبار
from sklearn.feature_extraction.text import TfidfVectorizer  # لتحويل النصوص إلى تمثيل رقمي
from sklearn.naive_bayes import MultinomialNB  # خوارزمية Naive Bayes متعددة الحدود
from sklearn.metrics import accuracy_score, classification_report  # لحساب الدقة والتقارير التفصيلية
import joblib  # لحفظ النموذج (غير مستخدم هنا فعلياً، لكنه جاهز)
import os  # للتعامل مع الملفات
import wandb  # لتتبع التجارب باستخدام Weights & Biases

# --- تهيئة مشروع W&B ---
wandb.init(project="mlops_text_classification", name="naive_bayes_wandb")

# إعداد معلمات التجربة (قابلة للتغيير لاحقًا عبر Sweeps أو يدويًا)
params = {
    "test_size": 0.2,  # نسبة البيانات للاختبار
    "ngram_range": (1, 2),  # استخدام الكلمات المفردة والثنائية
    "alpha": 1.0,  # معامل التنعيم في Naive Bayes
    "max_features": 5000  # عدد السمات القصوى في TF-IDF
}
wandb.config.update(params)  # تسجيل المعلمات في لوحة W&B

# --- تحميل البيانات ---
# df = pd.read_csv("data/dataset.csv")
df = pd.read_csv("data/dataset_small.csv")

# التعامل مع النصوص الفارغة
df["text"] = df["text"].fillna("")

# --- تقسيم البيانات إلى تدريب واختبار ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=params["test_size"], random_state=42
)

# --- تحويل النصوص إلى تمثيل عددي باستخدام TF-IDF ---
vectorizer = TfidfVectorizer(
    ngram_range=params["ngram_range"], 
    stop_words="english", 
    max_features=params["max_features"]
)

# التدريب والتحويل
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- تدريب نموذج Naive Bayes ---
model = MultinomialNB(alpha=params["alpha"])
model.fit(X_train_tfidf, y_train)

# --- التنبؤ والتقييم ---
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

# --- تسجيل الأداء في Weights & Biases ---
wandb.log({"accuracy": acc})

# تسجيل تقرير مفصل للتصنيف (الدقة - الاستدعاء - F1)
report = classification_report(y_test, y_pred, output_dict=True)
wandb.log({"classification_report": report})

# طباعة الدقة في الطرفية
print(f"Accuracy: {acc:.4f}")
