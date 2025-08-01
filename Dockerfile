# استخدم صورة رسمية خفيفة
FROM python:3.10-slim

# تعيين مجلد العمل
WORKDIR /app

# نسخ فقط ملف المتطلبات (يضمن cache أفضل)
COPY requirements.txt .

# تثبيت الحزم أولًا، حتى لو تغيّر الكود لاحقًا ما يعيد التثبيت
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# نسخ الملفات بعد تثبيت المتطلبات (يضمن إعادة البناء فقط إذا تغيّرت هذه الملفات)
COPY app/ ./app/
COPY models/ ./models/

# فتح المنفذ (اختياري للتوثيق فقط)
EXPOSE 8000

# الأمر الأساسي
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
