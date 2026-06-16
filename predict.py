"""
predict.py — Spam Email Classifier
Loads the saved model and vectorizer, then classifies messages entered by the user.

Usage:
    python predict.py                  # interactive mode
    python predict.py "Free money now" # single message mode
"""

import os
import re
import string
import sys

import joblib

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
MODEL_PATH      = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

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


# ── text cleaning (mirrors train.py) ──────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply the same preprocessing used during training."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"\S+@\S+", "", text)                    # e-mail addresses
    text = re.sub(r"\b\d[\d\s\-().]{6,}\d\b", "", text)   # phone numbers
    text = text.translate(
        str.maketrans("", "", string.punctuation + string.digits)
    )
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


# ── loading ────────────────────────────────────────────────────────────────

def load_artifacts():
    """Load and return (model, vectorizer). Raises FileNotFoundError if missing."""
    for path, name in [(MODEL_PATH, "model.pkl"), (VECTORIZER_PATH, "vectorizer.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{name}' not found. Please run  python train.py  first."
            )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# ── prediction ─────────────────────────────────────────────────────────────

def predict(message: str, model, vectorizer) -> dict:
    """
    Classify a single message.

    Returns a dict with keys:
        label       – 'SPAM' or 'HAM'
        confidence  – probability of the predicted class (0–100 %)
        raw_text    – original message
    """
    cleaned   = clean_text(message)
    features  = vectorizer.transform([cleaned])
    label_num = model.predict(features)[0]
    proba     = model.predict_proba(features)[0]

    label      = "SPAM" if label_num == 1 else "HAM"
    confidence = proba[label_num] * 100

    return {"label": label, "confidence": confidence, "raw_text": message}


def print_result(result: dict) -> None:
    """Pretty-print the prediction result."""
    icon = "🚫" if result["label"] == "SPAM" else "✅"
    print("\n" + "-" * 52)
    print(f"  Message    : {result['raw_text'][:70]}")
    print(f"  Prediction : {icon}  {result['label']}")
    print(f"  Confidence : {result['confidence']:.1f}%")
    print("-" * 52 + "\n")


# ── entry point ────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║      Spam Email Classifier  v1.0     ║")
    print("╚══════════════════════════════════════╝")

    # Load model artifacts once
    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── single-message mode (argument supplied) ─────────────────────────
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        result  = predict(message, model, vectorizer)
        print_result(result)
        return

    # ── interactive mode ────────────────────────────────────────────────
    print("\nEnter a message to classify, or type 'quit' / 'exit' to stop.\n")
    while True:
        try:
            message = input("  Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not message:
            print("  [WARN] Please enter a non-empty message.\n")
            continue

        if message.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye!\n")
            break

        result = predict(message, model, vectorizer)
        print_result(result)


if __name__ == "__main__":
    main()
