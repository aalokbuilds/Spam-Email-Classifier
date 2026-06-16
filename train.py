"""
train.py — Spam Email Classifier
Trains a Multinomial Naive Bayes model on the SMS Spam Collection dataset.
Saves the trained model and TF-IDF vectorizer to disk for later use.
"""

import os
import re
import string
import zipfile
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ── paths ──────────────────────────────────────────────────────────────────
DATASET_DIR     = os.path.join(os.path.dirname(__file__), "dataset")
DATASET_FILE    = os.path.join(DATASET_DIR, "SMSSpamCollection")
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
DATASET_URL     = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00228/smsspamcollection.zip"
)

# ── built-in English stop words (no NLTK required) ────────────────────────
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "has", "have", "having", "he", "her",
    "here", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "ll", "m", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "out", "own", "re", "s", "same", "she", "should",
    "so", "some", "such", "t", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "us", "ve", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your",
}


# ── helpers ────────────────────────────────────────────────────────────────

def download_dataset() -> None:
    """Download and extract the SMS Spam Collection dataset if not present."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    if os.path.exists(DATASET_FILE):
        print("[INFO] Dataset already present — skipping download.")
        return

    print("[INFO] Downloading SMS Spam Collection dataset …")
    zip_path = os.path.join(DATASET_DIR, "smsspamcollection.zip")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATASET_DIR)
        os.remove(zip_path)
        print("[INFO] Dataset downloaded and extracted successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the dataset: {exc}\n"
            "Please manually place 'SMSSpamCollection' inside the dataset/ folder."
        ) from exc


def load_dataset() -> pd.DataFrame:
    """Load the tab-separated dataset and return a tidy DataFrame."""
    df = pd.read_csv(
        DATASET_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def clean_text(text: str) -> str:
    """
    Preprocess a single text string:
      1. Lowercase
      2. Remove URLs, e-mail addresses, and phone numbers
      3. Remove punctuation and digits
      4. Strip extra whitespace
      5. Remove stop words
    """
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # remove e-mail addresses
    text = re.sub(r"\S+@\S+", "", text)
    # remove phone numbers (basic pattern)
    text = re.sub(r"\b\d[\d\s\-().]{6,}\d\b", "", text)
    # remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # remove stop words
    tokens = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(tokens)


def print_section(title: str) -> None:
    """Pretty-print a section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── main pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    # 1. Download dataset (skips if already present)
    download_dataset()

    # 2. Load data
    print_section("LOADING DATASET")
    df = load_dataset()
    print(f"  Total samples : {len(df):,}")
    print(f"  Ham  (0)      : {(df['label_num'] == 0).sum():,}")
    print(f"  Spam (1)      : {(df['label_num'] == 1).sum():,}")

    # 3. Preprocess
    print_section("PREPROCESSING TEXT")
    df["clean_message"] = df["message"].apply(clean_text)
    print("  Text cleaning complete.")
    print("\n  Sample transformations:")
    for _, row in df.sample(3, random_state=42).iterrows():
        print(f"    [{row['label'].upper()}]")
        print(f"      Original : {row['message'][:80]}")
        print(f"      Cleaned  : {row['clean_message'][:80]}\n")

    # 4. TF-IDF vectorisation
    print_section("TF-IDF VECTORISATION")
    vectorizer = TfidfVectorizer(
        max_features=5000,   # keep the top 5 000 terms
        ngram_range=(1, 2),  # unigrams + bigrams
        sublinear_tf=True,   # apply log(1+tf) scaling
    )
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label_num"].values
    print(f"  Feature matrix shape : {X.shape}")

    # 5. Train / test split
    print_section("TRAIN / TEST SPLIT  (80 / 20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Training samples : {X_train.shape[0]:,}")
    print(f"  Testing  samples : {X_test.shape[0]:,}")

    # 6. Train Multinomial Naive Bayes
    print_section("TRAINING  —  Multinomial Naive Bayes")
    model = MultinomialNB(alpha=0.1)  # small Laplace smoothing
    model.fit(X_train, y_train)
    print("  Training complete.")

    # 7. Evaluate
    print_section("MODEL EVALUATION")
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall    * 100:.2f}%")
    print(f"  F1-Score  : {f1        * 100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print("              Predicted Ham   Predicted Spam")
    print(f"  Actual Ham       {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"  Actual Spam      {cm[1][0]:>5}           {cm[1][1]:>5}")

    # 8. Save model and vectorizer
    print_section("SAVING MODEL & VECTORIZER")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"  Model saved      → {MODEL_PATH}")
    print(f"  Vectorizer saved → {VECTORIZER_PATH}")
    print("\n  ✓ Training pipeline complete!\n")


if __name__ == "__main__":
    main()
