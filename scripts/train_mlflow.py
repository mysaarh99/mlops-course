# استيراد المكتبات الضرورية
import pandas as pd  # لمعالجة البيانات
from sklearn.model_selection import train_test_split, GridSearchCV  # لتقسيم البيانات وضبط المعلمات
from sklearn.feature_extraction.text import TfidfVectorizer  # لتحويل النص إلى متجهات رقمية
from sklearn.linear_model import LogisticRegression  # خوارزمية الانحدار اللوجستي
from sklearn.pipeline import Pipeline  # لإنشاء سلسلة من المعالجات
from sklearn.metrics import accuracy_score  # لحساب الدقة

# مكتبات MLOps للتتبع والتسجيل
import mlflow
import mlflow.sklearn
import os  # للتعامل مع الملفات والمجلدات

# إعداد تجربة باسم "Advanced_Text_Classification"
mlflow.set_experiment("Advanced_Text_Classification")

# تحميل مجموعة البيانات من ملف CSV
# df = pd.read_csv("data/dataset.csv")
df = pd.read_csv("data/dataset_small.csv")


# معالجة القيم المفقودة في العمود النصي "text"
df["text"] = df["text"].fillna("")

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# تعريف الـ Pipeline الذي يتكون من:
# 1. تحويل النصوص إلى متجهات باستخدام TF-IDF
# 2. تصنيف باستخدام Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# إعداد شبكة المعلمات لتجربتها باستخدام Grid Search
param_grid = {
    'clf__C': [0.1, 1, 10],  # معامل الانتظام Regularization strength
    'clf__penalty': ['l2'],  # نوع العقوبة
    'clf__solver': ['lbfgs']  # الخوارزمية المستخدمة في التدريب
}

# تنفيذ البحث الشبكي مع 3-تقاطع (3-fold cross-validation)
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)

# بدء جلسة تسجيل باستخدام MLflow
with mlflow.start_run() as run:
    # تدريب النموذج باستخدام GridSearchCV
    grid.fit(X_train, y_train)
    
    # التنبؤ على بيانات الاختبار
    y_pred = grid.predict(X_test)
    
    # حساب دقة النموذج
    acc = accuracy_score(y_test, y_pred)

    # تسجيل أفضل المعلمات التي تم العثور عليها
    mlflow.log_param("best_C", grid.best_params_['clf__C'])
    mlflow.log_param("penalty", grid.best_params_['clf__penalty'])

    # تسجيل مقياس الأداء (الدقة)
    mlflow.log_metric("accuracy", acc)

    # إنشاء مجلد لحفظ النموذج إن لم يكن موجودًا
    os.makedirs("models", exist_ok=True)

    # تسجيل النموذج المدرب داخل MLflow
    mlflow.sklearn.log_model(grid.best_estimator_, "best_model")

    # طباعة معلومات للتتبع اليدوي
    print(f"Best params: {grid.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Run ID: {run.info.run_id}")
    print(f"URL: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

# إنهاء جلسة MLflow (اختياري في هذا السياق، لأن with block ينفذها تلقائيًا)
mlflow.end_run()
