import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os

def save_dataset():
    categories = ['alt.atheism', 'sci.space', 'rec.sport.hockey']
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame({'text': data.data, 'label': data.target})

    # إزالة القيم الفارغة أو استبدالها بنص فارغ
    df['text'] = df['text'].fillna("")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset.csv", index=False)
    print("✅ Dataset saved to data/dataset.csv (NaNs handled)")

if __name__ == "__main__":
    save_dataset()
