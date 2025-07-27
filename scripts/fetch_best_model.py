import wandb
import joblib
import os

# إعداد W&B
PROJECT = "mlops_text_classification"
ENTITY = "mysaarh99-academic-journals"
SWEEP_ID = "t39ykh89"

def download_best_model():
    api = wandb.Api()

    # Sweep path الكامل
    sweep_path = f"{ENTITY}/{PROJECT}/{SWEEP_ID}"

    try:
        sweep = api.sweep(sweep_path)
    except wandb.errors.CommError as e:
        print(f"❌ لم يتم العثور على Sweep بالمسار: {sweep_path}")
        print(str(e))
        return

    runs = sweep.runs
    if not runs:
        print("❌ لا توجد تجارب ضمن هذا الـ Sweep")
        return

    # إيجاد التجربة الأعلى دقة
    best_run = max(runs, key=lambda r: r.summary.get("accuracy", 0))

    print(f"🔹 أفضل تجربة: {best_run.name} بدقة {best_run.summary.get('accuracy'):.4f}")

    # تأكد من وجود مجلد models
    os.makedirs("models", exist_ok=True)

    # تحميل جميع الـ artifacts من أفضل تجربة
    for artifact in best_run.logged_artifacts():
        print(f"⬇️ تنزيل Artifact: {artifact.name}")
        artifact.download(root="models")

    # إعادة التسمية إلى اسم ثابت
    for file in os.listdir("models"):
        if file.endswith(".pkl") and "logistic_model" in file:
            target_path = "models/logistic_model_best.pkl"
            source_path = os.path.join("models", file)

            # إذا الملف الهدف موجود، احذفه أولاً
            if os.path.exists(target_path):
                os.remove(target_path)

            os.rename(source_path, target_path)
            print("✅ النموذج الأفضل محفوظ باسم models/logistic_model_best.pkl")
            break

if __name__ == "__main__":
    download_best_model()
