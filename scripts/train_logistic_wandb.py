# ========================
# 📦 استيراد المكتبات اللازمة
# ========================
import os
import pandas as pd
import joblib
import wandb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================================
# ⚙️ إعداد معلمات التجربة (Hyperparameters)
# ================================
params = {
    "test_size": 0.2,               # نسبة البيانات المستخدمة للاختبار
    "ngram_range": (1, 2),          # استخدام كلمات مفردة وثنائية في TF-IDF
    "C": 1.0,                       # معلمة انتظام لنموذج Logistic Regression
    "max_features": 5000            # الحد الأقصى لعدد الميزات النصية (كلمات) في TF-IDF
}

# ================================
# 🧪 تهيئة جلسة تتبع في Weights & Biases (W&B)
# ================================
wandb.init(
    project="mlops_text_classification",  # اسم المشروع في W&B
    name="logistic_regression_run",       # اسم التجربة الحالية
    config=params                          # تسجيل المعلمات مع W&B
)

# 🧩 استرجاع المعلمات من config (للتوافق مع sweeps لاحقًا)
config = wandb.config
ngram_range = tuple(config.ngram_range)  # تحويل ngram_range إلى tuple إذا لم يكن كذلك

# ================================
# 📁 تحميل البيانات النصية من ملف CSV
# ================================
df = pd.read_csv("data/dataset.csv")     # تأكد أن الملف موجود في هذا المسار
# df = pd.read_csv("data/dataset_small.csv")     # تأكد أن الملف موجود في هذا المسار

df["text"] = df["text"].fillna("")       # معالجة النصوص الفارغة بسلسلة فارغة

# ================================
# 🔀 تقسيم البيانات إلى تدريب واختبار
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=config.test_size, random_state=42
)

# ================================
# 🔤 تحويل النصوص إلى ميزات رقمية باستخدام TF-IDF
# ================================
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,             # مدى تركيب الجمل (1, 2) = unigram + bigram
    stop_words="english",                # إزالة الكلمات الشائعة مثل "the", "and"
    max_features=config.max_features     # تحديد عدد الميزات الأكثر شيوعًا
)

# تدريب الناقل على بيانات التدريب وتحويلها
X_train_tfidf = vectorizer.fit_transform(X_train)
# تحويل بيانات الاختبار بنفس ناقل الميزات المدرب
X_test_tfidf = vectorizer.transform(X_test)

# ================================
# 🧠 تدريب نموذج Logistic Regression
# ================================
model = LogisticRegression(C=config.C, max_iter=1000)
model.fit(X_train_tfidf, y_train)   # تدريب النموذج على بيانات TF-IDF

# ================================
# 🧮 التنبؤ على بيانات الاختبار وتقييم الأداء
# ================================
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)     # حساب الدقة على مجموعة الاختبار

# تسجيل نتيجة الدقة في W&B
wandb.log({"accuracy": acc})

# طباعة المعلمات والدقة في الطرفية
print(f"✅ Params: {dict(config)}")
print(f"✅ Accuracy: {acc:.4f}")

# ================================
# 💾 حفظ النموذج وناقل الميزات (TF-IDF)
# ================================
# التأكد من أن vectorizer قد تم تدريبه (fit) قبل الحفظ
assert hasattr(vectorizer, "idf_"), "❌ Vectorizer غير مدرب!"

# إنشاء مجلد لحفظ النماذج إن لم يكن موجودًا
os.makedirs("models", exist_ok=True)

# مسارات الملفات لحفظ النموذج والناقل
model_path = f"models/logistic_model_C{config.C}.pkl"
best_model_path = "models/logistic_model_best.pkl"
vectorizer_path = "models/vectorizer.pkl"

# حفظ النموذج المدرب وناقل الميزات باستخدام joblib
joblib.dump(model, model_path)
joblib.dump(model, best_model_path)         # نسخة موحدة للنشر أو التقييم
joblib.dump(vectorizer, vectorizer_path)

# ================================
# 📦 تسجيل النموذج كـ Artifact في W&B
# ================================
artifact = wandb.Artifact("logistic_regression_model", type="model")
artifact.add_file(model_path)
artifact.add_file(best_model_path)
artifact.add_file(vectorizer_path)

# تسجيل الملفات وربطها بالتجربة
# wandb.save(best_model_path)
wandb.log_artifact(artifact)

# تأكيد الحفظ
print(f"✅ Model saved at {best_model_path}")
print(f"✅ Vectorizer saved at {vectorizer_path}")
