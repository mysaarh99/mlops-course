FROM python:3.10-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app


# نسخ المتطلبات
COPY requirements.txt .

# تثبيت المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ النماذج والملفات
COPY models ./models
COPY app ./app

# فتح المنفذ
EXPOSE 8000

# تشغيل الخدمة عند تشغيل الحاوية
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
