# import pandas as pd
# from sklearn.datasets import fetch_20newsgroups
# import os

# def save_dataset():
#     # تحديد الفئات المراد تحميلها من مجموعة بيانات 20 Newsgroups
#     categories = ['alt.atheism', 'sci.space', 'rec.sport.hockey']

#     # تحميل البيانات (النصوص والتسميات) بدون رؤوس الرسائل، تذييلات، والاقتباسات
#     data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

#     # تحويل البيانات إلى DataFrame من pandas مع عمود النصوص وعمود التسميات
#     df = pd.DataFrame({'text': data.data, 'label': data.target})

#     # معالجة القيم الفارغة (NaN) في عمود النصوص باستبدالها بنص فارغ ""
#     df['text'] = df['text'].fillna("")

#     # التأكد من وجود مجلد 'data'، وإن لم يكن موجودًا يتم إنشاؤه
#     os.makedirs("data", exist_ok=True)

#     # حفظ DataFrame إلى ملف CSV داخل مجلد 'data' بدون إضافة رقم الصفوف (index)
#     df.to_csv("data/dataset.csv", index=False)

#     # طباعة رسالة تأكيد نجاح العملية
#     print("✅ Dataset saved to data/dataset.csv (NaNs handled)")

# # تنفيذ الدالة فقط إذا كان هذا الملف هو الملف الرئيسي المشغل
# if __name__ == "__main__":
#     save_dataset()
# ----------------
# small data for DVC and storage
import pandas as pd
import os
from sklearn.datasets import fetch_20newsgroups

def save_dataset():
    # التأكد من وجود data/ والمصدر
    os.makedirs("data", exist_ok=True)

    small_path = "data/dataset_small.csv"

    # إذا لم تكن النسخة المصغرة موجودة، أنشئها من المصدر الأصلي
    if not os.path.exists(small_path):
        print("⚠️ dataset_small.csv غير موجود. يتم إنشاؤه من المصدر...")
        categories = ['alt.atheism', 'sci.space', 'rec.sport.hockey']
        data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
        df = pd.DataFrame({'text': data.data, 'label': data.target})
        df = df.sample(n=1000, random_state=42)
        df.to_csv(small_path, index=False)
        print("✅ Created data/dataset_small.csv (1000 rows)")
    else:
        print("ℹ️ Using existing dataset_small.csv")

    # انسخها إلى dataset.csv لاستخدامها في بقية السكربتات
    df_small = pd.read_csv(small_path)
    df_small.to_csv("data/dataset.csv", index=False)
    print("✅ Ready dataset.csv for pipeline (from dataset_small.csv)")

if __name__ == "__main__":
    save_dataset()

