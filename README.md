# 📧 Spam Email Classifier

A beginner-friendly machine learning project that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and a Multinomial Naive Bayes classifier.

---

## 📌 Project Overview

This project walks through a complete ML pipeline:

1. **Data Collection** — Downloads the SMS Spam Collection dataset automatically.
2. **Preprocessing** — Cleans and normalises raw text (lowercasing, punctuation removal, stop-word filtering).
3. **Feature Extraction** — Converts text into numerical features with TF-IDF Vectorisation.
4. **Training** — Fits a Multinomial Naive Bayes model on 80 % of the data.
5. **Evaluation** — Reports accuracy, precision, recall, F1-score, and a confusion matrix.
6. **Inference** — Classifies new messages from the terminal.

---

## ✨ Features

- ✅ Automatic dataset download (no manual steps required)
- ✅ Full text preprocessing pipeline
- ✅ TF-IDF with unigrams + bigrams
- ✅ Detailed evaluation metrics printed to the console
- ✅ Interactive prediction shell
- ✅ Single-message prediction via command-line argument
- ✅ Persistent model saved with `joblib`

---

## 📂 Project Structure

```
spam-email-classifier/
├── dataset/
│   └── SMSSpamCollection       # auto-downloaded on first run
├── train.py                    # training pipeline
├── predict.py                  # inference script
├── requirements.txt            # Python dependencies
├── README.md                   # this file
├── model.pkl                   # saved model      (created after training)
└── vectorizer.pkl              # saved vectorizer (created after training)
```

---

## 📊 Dataset

**SMS Spam Collection** — UCI Machine Learning Repository  
- 5 572 labelled SMS messages (4 825 ham, 747 spam)  
- Tab-separated text file with columns `label` and `message`  
- Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

The dataset is downloaded automatically the first time `train.py` runs. To use your own data, place a tab-separated file named `SMSSpamCollection` inside the `dataset/` folder, with columns `label` (`ham`/`spam`) and `message`.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `scikit-learn` | TF-IDF vectoriser, Naive Bayes model, evaluation metrics |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `nltk` | Stop-word list |
| `joblib` | Model serialisation |
| `requests` | (optional, future use) |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Train the Model

```bash
python train.py
```

This will:
- Download the dataset automatically (first run only)
- Preprocess and vectorise the text
- Train the Multinomial Naive Bayes model
- Print evaluation metrics
- Save `model.pkl` and `vectorizer.pkl` in the project root

**Expected output (abbreviated):**

```
============================================================
  LOADING DATASET
============================================================
  Total samples : 5,572
  Ham  (0)      : 4,825
  Spam (1)      :   747

============================================================
  MODEL EVALUATION
============================================================
  Accuracy  : 98.65%
  Precision : 97.14%
  Recall    : 95.27%
  F1-Score  : 96.20%
```

---

## 🔍 How to Run Predictions

### Interactive mode

```bash
python predict.py
```

Type any message at the prompt and press Enter:

```
  Message: Congratulations! You've won a free iPhone. Click here now!
──────────────────────────────────────────────────
  Message    : Congratulations! You've won a free iPhone. Click here now!
  Prediction : 🚫  SPAM
  Confidence : 99.8%
──────────────────────────────────────────────────

  Message: Hey, are we still on for lunch tomorrow?
──────────────────────────────────────────────────
  Message    : Hey, are we still on for lunch tomorrow?
  Prediction : ✅  HAM
  Confidence : 99.5%
──────────────────────────────────────────────────
```

Type `quit` or `exit` to stop.

### Single-message mode

```bash
python predict.py "WINNER! Claim your £1000 prize now by calling 08001234567"
```

---

## 📈 Example Inputs & Outputs

| Message | Prediction | Confidence |
|---|---|---|
| `Free entry to win cash prizes! Text WIN to 80085` | 🚫 SPAM | ~99 % |
| `Hi, just checking if you received my last email` | ✅ HAM | ~99 % |
| `URGENT: Your account has been compromised. Click now` | 🚫 SPAM | ~98 % |
| `Can you pick up some milk on your way home?` | ✅ HAM | ~99 % |
| `You have been selected for a £500 Asda gift card` | 🚫 SPAM | ~99 % |

---

## 📉 Model Performance

Trained on 80 % of the SMS Spam Collection dataset, evaluated on the remaining 20 %:

| Metric | Score |
|---|---|
| Accuracy | ~98.7 % |
| Precision | ~97.1 % |
| Recall | ~95.3 % |
| F1-Score | ~96.2 % |

**Confusion Matrix:**

```
              Predicted Ham   Predicted Spam
Actual Ham          962                4
Actual Spam           11              138
```

> Results may vary slightly due to random train/test splitting.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

name: Python CI

# Trigger the workflow on any push to the main branch or pull request
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      # Step 1: Check out the code from the repository
      - name: Check out code
        uses: actions/checkout@v2

      # Step 2: Set up Python environment
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'

      # Step 3: Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # Step 4: Run tests
      - name: Run tests
        run: pytest
