import wandb
import joblib
import os

# هذا السكربت يقوم بالاتصال بمنصة Weights & Biases (W&B) باستخدام API للحصول على بيانات تجربة معينة (Sweep).

# إعداد متغيرات W&B (مشروع، كيان، Sweep ID)
PROJECT = "mlops_text_classification"  # اسم مشروع W&B
ENTITY = "mysaascholar98-mlops"  # اسم الحساب أو الفريق في W&B
SWEEP_ID = "t39ykh89"  # معرف الـ Sweep (تجارب متعددة ضمن نفس الإعدادات)

def download_best_model():
    api = wandb.Api()  # إنشاء كائن API للتواصل مع W&B

    # تكوين المسار الكامل للـ Sweep في W&B (ENTITY/PROJECT/SWEEP_ID)
    sweep_path = f"{ENTITY}/{PROJECT}/{SWEEP_ID}"

    try:
        sweep = api.sweep(sweep_path)  # تحميل بيانات الـ Sweep
    except wandb.errors.CommError as e:
        # في حال حدوث خطأ بالاتصال أو عدم وجود الـ Sweep
        print(f"❌ لم يتم العثور على Sweep بالمسار: {sweep_path}")
        print(str(e))
        return  # إيقاف التنفيذ

    runs = sweep.runs  # جلب كل التجارب (Runs) ضمن هذا الـ Sweep
    if not runs:
        print("❌ لا توجد تجارب ضمن هذا الـ Sweep")
        return  # إذا لم توجد تجارب، إيقاف التنفيذ

    # إيجاد أفضل تجربة حسب قيمة دقة النموذج (accuracy) في ملخص التجربة
    best_run = max(runs, key=lambda r: r.summary.get("accuracy", 0))

    print(f"🔹 أفضل تجربة: {best_run.name} بدقة {best_run.summary.get('accuracy'):.4f}")

    # التأكد من وجود مجلد models لحفظ الملفات
    os.makedirs("models", exist_ok=True)

    # تنزيل جميع الـ Artifacts (الملفات الناتجة) من أفضل تجربة ضمن المجلد "models"
    for artifact in best_run.logged_artifacts():
        print(f"⬇️ تنزيل Artifact: {artifact.name}")
        artifact.download(root="models")

    # إعادة تسمية النموذج الأفضل إلى اسم ثابت لتسهيل الاستخدام لاحقًا
    for file in os.listdir("models"):
        if file.endswith(".pkl") and "logistic_model" in file:
            target_path = "models/logistic_model_best.pkl"
            source_path = os.path.join("models", file)

            # إذا كان الملف الهدف موجود مسبقًا، احذفه قبل إعادة التسمية
            if os.path.exists(target_path):
                os.remove(target_path)

            os.rename(source_path, target_path)  # إعادة التسمية
            print("✅ النموذج الأفضل محفوظ باسم models/logistic_model_best.pkl")
            break  # إيقاف البحث بعد إيجاد أول ملف مطابق

# نقطة الدخول للسكربت
if __name__ == "__main__":
    download_best_model()
