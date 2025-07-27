import joblib

# محاولة تحميل النموذج الأفضل (المدرب) و الـ Vectorizer (لتحويل النصوص لأرقام)
try:
    # تحميل نموذج التصنيف المخزن سابقًا
    model = joblib.load("models/logistic_model_best.pkl")
    # تحميل الـ vectorizer الذي يحول النصوص إلى تمثيل رقمي (مثل TF-IDF أو CountVectorizer)
    vectorizer = joblib.load("models/vectorizer.pkl")
except FileNotFoundError:
    # في حال عدم وجود الملفات، طباعة رسالة خطأ وإنهاء البرنامج
    print("❌ لم يتم العثور على النموذج أو الـ vectorizer. تأكد من تشغيل التدريب أولاً.")
    exit()

# مجموعة نصوص تجريبية لاختبار النموذج عليها
texts = [
    "NASA is preparing a new mission to Mars.",  # نص عن الفضاء
    "The local hockey team won their championship game.",  # نص عن رياضة الهوكي
    "Discussions about atheism often become controversial."  # نص عن الإلحاد
]

# تحويل النصوص إلى تمثيل رقمي باستخدام الـ vectorizer
X = vectorizer.transform(texts)

# استخدام النموذج لتوقع تصنيفات النصوص المحولة
predictions = model.predict(X)

# طباعة كل نص مع التصنيف المتوقع له
for text, label in zip(texts, predictions):
    print("-----------")
    print(f"النص: {text}")
    print(f"التصنيف المتوقع: {label}")
