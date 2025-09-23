# scripts/prepare_data.py
import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

SMALL_PATH = Path("data/dataset_small.csv")
OUT_PATH = Path("data/dataset.csv")

def save_dataset():
    # تأكد من وجود مجلد data/
    os.makedirs("data", exist_ok=True)

    if not SMALL_PATH.exists():
        # داخل CI لا نحمّل أبداً
        if os.getenv("CI") == "true":
            print("❌ CI detected and data/dataset_small.csv is missing. Offline mode only.")
            print("▶️ أضف الملف للمستودع (git) أو تتبّعه بـ DVC ثم نفّذ `dvc pull` قبل `dvc repro`.")
            raise SystemExit(1)

        # محليًا: أنشئ الملف من المصدر
        print("⚠️ dataset_small.csv غير موجود. يتم إنشاؤه محليًا من 20 Newsgroups...")
        # نجلب عند الحاجة فقط لتجنّب 403 في CI

        # تحديد الفئات
        categories = ['alt.atheism', 'sci.space', 'rec.sport.hockey']

        # تحميل البيانات (بدون headers/footers/quotes)
        data = fetch_20newsgroups(
            subset='all',
            categories=categories,
            remove=('headers', 'footers', 'quotes')
        )

        # تحويل إلى DataFrame
        df = pd.DataFrame({'text': data.data, 'label': data.target})

        # أخذ عينة ثابتة 1000 صف
        df = df.sample(n=1000, random_state=42)

        # حفظ النسخة المصغرة
        df.to_csv(SMALL_PATH, index=False)
        print("✅ Created data/dataset_small.csv (1000 rows)")

    else:
        print("ℹ️ Using existing dataset_small.csv")

    # إنشاء dataset.csv للاستخدام في بقية السكربتات
    df_small = pd.read_csv(SMALL_PATH)
    df_small.to_csv(OUT_PATH, index=False)
    print(f"✅ Ready {OUT_PATH} for pipeline (from dataset_small.csv)")

if __name__ == "__main__":
    save_dataset()
