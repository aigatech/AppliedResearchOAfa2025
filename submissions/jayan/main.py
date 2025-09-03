# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from agent import TCGATumorClassifier

DATA_PATH = "data/mini_tcga.csv"  # replace with your big TCGA file path

def train_and_save(data_path=DATA_PATH, test_size=0.2, random_state=42):
    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)
    print("Data shape:", df.shape)

    # Split (stratify by last column, if safe)
    label_col = df.columns[-1]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col] if df[label_col].nunique() > 1 else None)
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    clf = TCGATumorClassifier()
    clf.fit(train_df, save=True, n_saved_examples=10)

    # Evaluate on test set
    X_test, y_raw_test, _, _ = clf.prepare_xy(test_df)
    preds = []
    for row in X_test:
        preds.append(clf.predict_row(row))
    print("Test Accuracy:", accuracy_score(y_raw_test, preds))
    print(classification_report(y_raw_test, preds))

if __name__ == "__main__":
    train_and_save()
