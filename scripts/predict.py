import joblib

# تحميل النموذج الأفضل والـ Vectorizer
try:
    model = joblib.load("models/logistic_model_best.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
except FileNotFoundError:
    print("❌ لم يتم العثور على النموذج أو الـ vectorizer. تأكد من تشغيل التدريب أولاً.")
    exit()

# نصوص تجريبية
texts = [
    "NASA is preparing a new mission to Mars.",
    "The local hockey team won their championship game.",
    "Discussions about atheism often become controversial."
]

# تحويل النصوص
X = vectorizer.transform(texts)
predictions = model.predict(X)

# طباعة النتائج
for text, label in zip(texts, predictions):
    print("-----------")
    print(f"النص: {text}")
    print(f"التصنيف المتوقع: {label}")
