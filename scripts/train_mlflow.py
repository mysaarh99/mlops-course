import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# إعداد تجربة MLflow
mlflow.set_experiment("Advanced_Text_Classification")

# تحميل البيانات
df = pd.read_csv("data/dataset.csv")
# تنظيف البيانات: إزالة القيم الفارغة أو استبدالها بسلسلة فارغة
df["text"] = df["text"].fillna("")

# تقسيم البيانات

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)


# تعريف Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# شبكة المعلمات للتجريب
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}

# البحث عن أفضل المعلمات
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)

with mlflow.start_run() as run:
    grid.fit(X_train, y_train)
    
    # تقييم الأداء
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # تسجيل معلمات أفضل نموذج
    mlflow.log_param("best_C", grid.best_params_['clf__C'])
    mlflow.log_param("penalty", grid.best_params_['clf__penalty'])
    mlflow.log_metric("accuracy", acc)

    # حفظ النموذج
    os.makedirs("models", exist_ok=True)
    mlflow.sklearn.log_model(grid.best_estimator_, "best_model")
    print(f"Best params: {grid.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Run ID: {run.info.run_id}")
    print(f"URL: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

mlflow.end_run()