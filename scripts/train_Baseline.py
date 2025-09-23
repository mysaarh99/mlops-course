import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import time

overall_start = time.time()
# تحميل البيانات
df = pd.read_csv("data/dataset.csv")  # غيّر المسار حسب مكان ملفك

# تجهيز البيانات
df['text'] = df['text'].fillna('')
X = df['text']
y = df['label']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# إنشاء البايبلاين
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=200)
)

# حساب وقت التدريب

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"⏱️ Training time: {training_time:.4f} seconds")

# التنبؤ والتقييم

t1 = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - t1
print(f"⏱️ Predict time: {predict_time:.4f} seconds")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

overall_end = time.time()
total_runtime = overall_end - overall_start
print(f"⏱️ Total time: {total_runtime:.4f} seconds")
