"""
Phishing-website detection — prediction (predict.py).

Two ways to use it:
  1. Single URL:   python predict.py --url "http://example.com/login"
  2. Batch CSV:     python predict.py --csv data/batch_sample.csv
                    (add --out predictions.csv to save the results)

Batch CSV is the reliable mode: the CSV must contain the same 13 feature
columns train.py used (see FEATURES below), plus the label column if you
want the predictions checked against the truth.

Single-URL mode — read this before trusting its answer
------------------------------------------------------
A raw URL string only carries some of the 13 features. The rest need a
DNS lookup, a WHOIS/domain-age lookup, or web-traffic data, which this
script deliberately does NOT do (no network calls, no whois, no traffic
rank). So a single-URL prediction is an APPROXIMATION, and the script
says so in its output:

  URL-derived (computed from the URL string itself):
    Having_@_symbol      — 1 if '@' appears in the URL
    Having_IP            — 1 if the host is an IPv4 address
    Prefix_suffix_separation — 1 if '-' appears in the domain
    Redirection_//_symbol    — 1 if '//' appears after the protocol part
    Sub_domains          — 0/1/2 by dot count in the host (approx.)
    URL_Length           — 0/1/2 by total URL length (approx., see below)
    http_tokens          — 1 if 'http' appears again inside the domain/path
    tiny_url             — 1 if the host is a known URL shortener

  Dataset-only (NOT computable from a URL string — set to 0 here):
    age_domain, dns_record, domain_registration_length,
    statistical_report, web_traffic

  The 0/1/2 codes for Sub_domains and URL_Length are approximations:
  the source CSVs do not document the exact thresholds, so this script
  uses the commonly used UCI phishing-dataset thresholds — URL length
  <54 chars = 0, 54–75 = 1, >75 = 2; host with 1 dot = 0, 2 dots = 1,
  3+ dots = 2. A single-URL verdict can therefore differ from the
  verdict the same site would get from its full 13-feature row.

If you need a trustworthy verdict for a real URL, build its full
13-feature row (domain age, DNS record, traffic rank) and use --csv.
"""

import argparse
import os
import re
import sys
from urllib.parse import urlparse

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

# Same feature list and order as train.py (imported, not re-typed).
sys.path.insert(0, BASE_DIR)
from train import FEATURES  # noqa: E402

# Features that CAN be computed from a URL string alone.
URL_DERIVED = [
    "Having_@_symbol",
    "Having_IP",
    "Prefix_suffix_separation",
    "Redirection_//_symbol",
    "Sub_domains",
    "URL_Length",
    "http_tokens",
    "tiny_url",
]

# Features that CANNOT be computed from a URL string alone.
DATASET_ONLY = [
    "age_domain",
    "dns_record",
    "domain_registration_length",
    "statistical_report",
    "web_traffic",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "mcaf.ee", "su.pr", "shorturl.at", "rebrand.ly",
}

IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def url_to_features(url):
    """Compute the 8 URL-derived features from a URL string; the other
    5 dataset-only features are set to 0 (see the module docstring)."""
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    domain = host
    path = parsed.path or ""

    features = {f: 0 for f in FEATURES}
    features["Having_@_symbol"] = 1 if "@" in url else 0
    features["Having_IP"] = 1 if IPV4_RE.match(host) else 0
    features["Prefix_suffix_separation"] = 1 if "-" in domain else 0
    # '//' after the protocol part means a redirection trick.
    rest = url.split("://", 1)[1] if "://" in url else url
    features["Redirection_//_symbol"] = 1 if "//" in rest else 0
    dots = host.count(".")
    features["Sub_domains"] = 0 if dots <= 1 else (1 if dots == 2 else 2)
    length = len(url)
    features["URL_Length"] = 0 if length < 54 else (1 if length <= 75 else 2)
    features["http_tokens"] = 1 if "http" in (domain + path).lower() else 0
    features["tiny_url"] = 1 if host in SHORTENERS else 0
    return features


def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found: {MODEL_PATH}\nRun 'python train.py' first.")
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["model_name"], saved["features"]


def predict_rows(model, X):
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    return preds, probs


def main():
    parser = argparse.ArgumentParser(description="Predict phishing vs legitimate.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="a single URL to check")
    group.add_argument("--csv", help="a CSV with the 13 feature columns")
    parser.add_argument("--out", help="output CSV path (batch mode only)")
    args = parser.parse_args()

    model, model_name, features = load_model()
    print(f"Model: {model_name} | features: {len(features)}")

    if args.url:
        feats = url_to_features(args.url)
        X = pd.DataFrame([feats])[features]
        preds, probs = predict_rows(model, X)
        label = "PHISHING" if preds[0] == 1 else "LEGITIMATE"
        print(f"\nURL: {args.url}")
        print(f"Verdict: {label} (class {preds[0]})")
        if probs is not None:
            print(f"Confidence: P(phishing) = {probs[0][1]:.4f}, "
                  f"P(legitimate) = {probs[0][0]:.4f}")
        print("\nFeatures used (URL-derived):")
        for f in URL_DERIVED:
            print(f"  {f:<28} {feats[f]}")
        print("Features NOT computable from a URL alone (set to 0):")
        for f in DATASET_ONLY:
            print(f"  {f:<28} 0  <-- assumed, not measured")
        print("\nNOTE: this is an approximation. For a trustworthy verdict, "
              "use --csv with the full 13-feature row.")
        return

    # Batch mode.
    df = pd.read_csv(args.csv)
    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing feature columns: {missing}")
    X = df[features]
    preds, probs = predict_rows(model, X)
    df["predicted_label"] = preds
    df["predicted"] = ["Phishing" if p == 1 else "Legitimate" for p in preds]
    if probs is not None:
        df["P_phishing"] = probs[:, 1].round(4)
    for _, row in df.iterrows():
        name = row.get("Domain", row.get("domain", "?"))
        print(f"{name}: {row['predicted']}"
              + (f" (P={row['P_phishing']})" if probs is not None else ""))
    if "label" in df.columns:
        acc = (df["predicted_label"] == df["label"]).mean()
        print(f"\nBatch accuracy vs label column: {acc:.4f} "
              f"({(df['predicted_label'] == df['label']).sum()}/{len(df)} correct)")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
