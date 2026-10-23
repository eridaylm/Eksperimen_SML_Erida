import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "..", "telco_raw", "telco_churn_raw.csv")
OUT_DIR = os.path.join(BASE, "telco_preprocessing")
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def preprocess(raw_path=RAW_PATH, out_dir=OUT_DIR):
    df = pd.read_csv(raw_path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop_duplicates()

    for c in NUM_COLS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])
    train, test = train.copy(), test.copy()
    scaler = StandardScaler()
    train[NUM_COLS] = scaler.fit_transform(train[NUM_COLS])
    test[NUM_COLS] = scaler.transform(test[NUM_COLS])

    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    return train, test


if __name__ == "__main__":
    tr, te = preprocess()
    print("Selesai:", tr.shape, te.shape)