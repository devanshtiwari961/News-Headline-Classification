import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Dataset Collection
# Expect a CSV file "news_headlines.csv" with columns: "headline" and "category"
# For practice, you can manually create or export ~250 rows from Kaggle dataset.
INPUT_CSV = "news_headlines.csv"
MODEL_PATH = "models/tfidf_logreg_pipeline.joblib"
LABELS_PATH = "models/labels.joblib"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove non-letters
    text = re.sub(r"\s+", " ", text)       # collapse spaces
    return text.strip()

# Normalize labels to required set
ALLOWED_LABELS = {"tech", "health", "business", "entertainment", "sports", "politics"}
LABEL_MAP = {
    "technology": "tech",
    "tech": "tech",
    "science": "tech",   # optional mapping
    "health": "health",
    "business": "business",
    "finance": "business",
    "markets": "business",
    "entertainment": "entertainment",
    "sports": "sports",
    "sport": "sports",
    "politics": "politics",
}

def normalize_label(lbl: str) -> str | None:
    if not isinstance(lbl, str):
        return None
    key = lbl.strip().lower()
    mapped = LABEL_MAP.get(key)
    if mapped in ALLOWED_LABELS:
        return mapped
    # If original label already matches allowed set
    if key in ALLOWED_LABELS:
        return key
    return None


def main():
    # 2. Data Preprocessing
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["headline", "category"]).copy()
    # Apply normalization and filter to allowed labels
    df["norm_category"] = df["category"].apply(normalize_label)
    df = df[df["norm_category"].notna()].copy()

    df["clean_headline"] = df["headline"].apply(clean_text)

    # Optional: sample for mini project
    if len(df) > 250:
        df = df.sample(250, random_state=42)

    X = df["clean_headline"].tolist()
    y = df["norm_category"].astype(str).tolist()

    # 4. Building and Training the ML Model
    # Split into train/test with 200 train / 50 test (approx)
    train_size = min(200, int(0.8 * len(df)))
    classes = sorted(set(y))
    class_counts = Counter(y)
    test_size = len(df) - train_size
    safe_stratify = (
        len(classes) > 1 and min(class_counts.values()) >= 2 and test_size >= len(classes)
    )

    if safe_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42, stratify=None
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and labels
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(sorted(set(y)), LABELS_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()