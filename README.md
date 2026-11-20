# Phishing-Website Detection — Supervised ML

**Supervised classification: is a website legitimate or phishing?**

A small, honest college project for a Supervised Machine Learning course.
Two classifiers — Random Forest and Logistic Regression — are trained on
2,015 labelled websites and learn to tell legitimate sites (label 0) from
phishing sites (label 1) using 13 numeric features of the URL/domain.

Author: Naman Joshi (NAMAN3342), Roll No. 2361353 · Faculty: Dr. Shilpa Jain · Graphic Era Hill University, Bhimtal · Course: Supervised Machine Learning: Regression and Classification — 20+ hours, Scaler · Date: 20 November 2026 · License: MIT
(see the dataset source below — the datasets are public data, the code
here was written for this project).

## What it does

```
legitimate-urls.csv (1,017 rows, label 0)
phishing-urls.csv   (  998 rows, label 1)   ──► train.py ──► models/model.joblib
                                                        │
predict.py ──► single URL check, or batch --csv ──► Legitimate / Phishing + probability
```

## Measured results (real run, 20 Nov 2026)

70/30 stratified split — 1,410 train / 605 test (305 Legitimate, 300 Phishing),
`random_state=42`. These are the numbers from the actual run, not targets:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Random Forest (100 trees)** | **0.8083** | 0.8145 | 0.8077 | 0.8071 |
| Logistic Regression (scaled) | 0.7240 | 0.7281 | 0.7234 | 0.7224 |

Random Forest confusion matrix (rows = true, cols = predicted):

```
                 predicted
                 Legitimate  Phishing
true Legitimate      268        37
     Phishing         79       221
```

Per class (Random Forest): Legitimate precision 0.77 / recall 0.88 / F1 0.82;
Phishing precision 0.86 / recall 0.74 / F1 0.79. So the model misses
about 1 in 4 phishing sites (79 of 300) — that is the honest headline,
and the most important number for a security tool. See
`docs/PROJECT_REPORT.md` for the full discussion.

Top Random Forest feature importances: URL_Length 0.2229, web_traffic
0.1893, statistical_report 0.1542, dns_record 0.0977, Sub_domains 0.0782.

## How to run it (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python predict.py --csv data/batch_sample.csv
python predict.py --url "https://www.google.com"
```

On Linux/Mac use `python3` instead of `python`, and
`source .venv/bin/activate` instead of `.venv\Scripts\activate`.

Single-URL check is an **approximation** (see below); batch `--csv` with
the full 13 features is the reliable mode.

```bat
python predict.py --url "http://192.168.1.1@secure-login-paypal.com/verify-account.php"
python predict.py --csv data/batch_sample.csv --out predictions.csv
```

## Features (13, all coded 0/1/2 in the source CSVs)

`Having_@_symbol`, `Having_IP`, `Prefix_suffix_separation`,
`Redirection_//_symbol`, `Sub_domains`, `URL_Length`, `age_domain`,
`dns_record`, `domain_registration_length`, `http_tokens`,
`statistical_report`, `tiny_url`, `web_traffic`.

Text columns `Domain`, `Path` and `Protocol` are not used as features
(Domain/Path are identifiers; Protocol is almost always "http" in the
legitimate file, so it would leak which file a row came from).

## Single-URL mode — honest caveat

A raw URL string can only give 8 of the 13 features. The other 5 —
`age_domain`, `dns_record`, `domain_registration_length`,
`statistical_report`, `web_traffic` — need DNS / WHOIS / traffic data,
which this project does not fetch, so `predict.py` sets them to 0 and
prints a warning. The 0/1/2 codes for `Sub_domains` and `URL_Length`
are approximations too: the source CSVs do not document their exact
thresholds, so the script uses the commonly used UCI thresholds
(URL <54 chars = 0, 54–75 = 1, >75 = 2). A single-URL verdict can
therefore differ from the verdict the same site would get from its
full 13-feature row.

## Repo structure

```
phishing-detection/
├── README.md
├── requirements.txt        # pandas, numpy, scikit-learn, joblib
├── train.py                # train RF + Logistic Regression, save best model
├── predict.py              # --url single check / --csv batch prediction
├── data/
│   ├── legitimate-urls.csv # 1,017 rows, label 0
│   ├── phishing-urls.csv   #   998 rows, label 1
│   └── batch_sample.csv    # 5-row demo (3 legitimate, 2 phishing)
├── models/
│   └── model.joblib        # built by train.py (gitignored, do NOT commit)
└── docs/
    └── PROJECT_REPORT.md   # college project report
```

## Dataset source (credit)

Datasets: `legitimate-urls.csv` and `phishing-urls.csv` from
**jishnusaurav/Phishing-attack-PCAP-analysis-using-scapy**,
folder `Phishing-Website-Detection/datasets/`, on GitHub
(downloaded from the repo's raw file URLs). The code in this project
was written for this project and is not a copy of the original
author's notebook. No local paths from the original author are used.

## Limitations (say these in the viva)

- 80.8% test accuracy, not 100%: only 13 coded features and 2,015 rows,
  and the two classes overlap on many features.
- The model misses 26% of phishing sites (recall 0.74) — for a security
  tool, a missed phishing site is the worst error.
- Single-URL predictions are approximate (5 features assumed 0).
- The dataset is old and small; modern phishing pages may look different.
- This is a classifier, not a browser protector: it does not fetch,
  block, or verify any live website.
