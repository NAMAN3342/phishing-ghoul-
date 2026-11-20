"""
Phishing-website detection — supervised classification (train.py).

What it does
------------
Loads two public CSVs (legitimate-urls.csv, label 0, and phishing-urls.csv,
label 1), merges them, and trains two supervised classifiers on the same
70/30 stratified train/test split:
  * Random Forest
  * Logistic Regression (features standardised first)

The best model (higher test accuracy; Random Forest wins ties) is saved to
models/model.joblib, together with the feature list and the scaler used for
Logistic Regression.

Features used (13 numeric columns, already coded 0/1/2 in the source CSVs):
  Having_@_symbol, Having_IP, Prefix_suffix_separation,
  Redirection_//_symbol, Sub_domains, URL_Length, age_domain, dns_record,
  domain_registration_length, http_tokens, statistical_report, tiny_url,
  web_traffic

The text columns Domain, Path and Protocol are NOT used as features:
Domain/Path are identifiers, not numbers, and Protocol is almost always
"http" in the legitimate file, so it would leak the file a row came from
rather than teach the model what phishing looks like.

Dataset source (credit): jishnusaurav/Phishing-attack-PCAP-analysis-
using-scapy, folder Phishing-Website-Detection/datasets/ on GitHub.
The code in this project was written for this project; it is not a copy
of the original author's notebook.

Run (Windows):  python train.py
Run (Linux/Mac): python3 train.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# The 13 numeric feature columns, in a fixed order. predict.py imports
# this same list so train and predict can never drift apart.
FEATURES = [
    "Having_@_symbol",
    "Having_IP",
    "Prefix_suffix_separation",
    "Redirection_//_symbol",
    "Sub_domains",
    "URL_Length",
    "age_domain",
    "dns_record",
    "domain_registration_length",
    "http_tokens",
    "statistical_report",
    "tiny_url",
    "web_traffic",
]

CLASS_NAMES = ["Legitimate (0)", "Phishing (1)"]


def load_data():
    """Load both CSVs, label-check them, merge into one DataFrame."""
    legit_path = os.path.join(DATA_DIR, "legitimate-urls.csv")
    phish_path = os.path.join(DATA_DIR, "phishing-urls.csv")
    for path in (legit_path, phish_path):
        if not os.path.exists(path):
            sys.exit(f"Missing file: {path}")

    legit = pd.read_csv(legit_path)
    phish = pd.read_csv(phish_path)
    print(f"Legitimate rows: {len(legit)} (labels: {sorted(legit['label'].unique())})")
    print(f"Phishing rows:   {len(phish)} (labels: {sorted(phish['label'].unique())})")

    df = pd.concat([legit, phish], ignore_index=True)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns in the CSVs: {missing}")
    if df[FEATURES + ["label"]].isna().any().any():
        sys.exit("Missing values found in features/labels — fix the CSVs first.")
    return df


def evaluate(name, model, X_test, y_test):
    """Print accuracy / precision / recall / F1 + confusion matrix, return accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f} (macro)")
    print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.4f} (macro)")
    print(f"F1:        {f1_score(y_test, y_pred, average='macro'):.4f} (macro)")
    print("Confusion matrix (rows = true, cols = predicted; "
          "order: Legitimate, Phishing):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    return acc


def main():
    df = load_data()
    print(f"Total rows: {len(df)}")
    print("Label counts:\n", df["label"].value_counts().to_string())

    X = df[FEATURES]
    y = df["label"].astype(int)

    # 70/30 stratified split: both classes keep their ~50/50 share in
    # train and test. random_state=42 so the run is reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"(70/30 stratified split, random_state=42)")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = evaluate("Random Forest (100 trees)", rf, X_test, y_test)

    # Logistic Regression needs scaled inputs; the scaler is part of the
    # pipeline so predict.py gets the same scaling automatically.
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr.fit(X_train, y_train)
    lr_acc = evaluate("Logistic Regression (scaled, max_iter=1000)", lr, X_test, y_test)

    # Best model wins; Random Forest wins a tie (it needs no scaler and
    # its feature importances are useful in the viva).
    if rf_acc >= lr_acc:
        best_name, best_model = "RandomForestClassifier", rf
        best_acc = rf_acc
    else:
        best_name, best_model = "LogisticRegression", lr
        best_acc = lr_acc

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump({"model": best_model, "model_name": best_name,
                 "features": FEATURES}, model_path)
    print(f"Best model: {best_name} (test accuracy {best_acc:.4f})")
    print(f"Saved to {model_path}")

    print("\nRandom Forest feature importances:")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_),
                            key=lambda t: -t[1]):
        print(f"  {feat:<28} {imp:.4f}")


if __name__ == "__main__":
    main()
