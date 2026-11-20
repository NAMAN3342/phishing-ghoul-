# Phishing-Website Detection — Supervised Machine Learning Project Report

**Course:** Supervised Machine Learning: Regression and Classification — 20+ hours, Scaler
**Project:** Phishing-Website Detection — supervised classification (Legitimate vs Phishing)
**Student:** Naman Joshi — 4th-year B.Tech — Roll No.: 2361353
**Faculty:** Dr. Shilpa Jain
**Institution:** Graphic Era Hill University, Bhimtal
**Date:** 20 November 2026
> All numbers below are from the author's own run on 20 Nov 2026, not copied
> from any tutorial or README. The code was written for this project; only
> the two datasets are taken from another public repo (credited in §9).

---

## 1. Abstract

Phishing websites copy the look of a trusted site to steal passwords and
payment details. This project turns "is this site phishing?" into a
supervised classification problem: 2,015 labelled websites (1,017
legitimate, 998 phishing), each described by 13 numeric features of its
URL and domain, are used to train two classifiers — Random Forest and
Logistic Regression. On a held-out test set of 605 websites the Random
Forest scores 80.83% accuracy, clearly beating Logistic Regression
(72.40%). The model is useful as a warning signal, not a guarantee: it
still misses about 1 in 4 phishing sites, and that limitation is stated
plainly in this report.

## 2. Problem

A user cannot tell a phishing site from a real one by looking at it —
that is the whole point of the attack. But phishing URLs do leave
statistical traces: they are often long, use IP addresses or '@' tricks,
have many sub-domains, use URL shorteners, belong to very young domains,
or have no DNS record. No single trace proves anything, but the
combination can be learned from labelled examples. That is what this
project does.

## 3. Supervised learning used

**Classification (used here):** each website is one labelled example
(features in, label out). The model predicts one of two classes —
Legitimate (0) or Phishing (1). This is the right tool because the
question is a discrete decision ("safe or phishing?"), not a number.

**Regression (course context, not claimed as implemented):** the course
also covers regression. In this project it appears only as future work —
a 0–100 phishing-risk score per URL instead of a hard class. Do not tell
the teacher regression is implemented; it is not, in either `train.py`
or `predict.py`.

## 4. Data

- **Source:** `legitimate-urls.csv` (1,017 rows, label 0) and
  `phishing-urls.csv` (998 rows, label 1), from the public repo
  jishnusaurav/Phishing-attack-PCAP-analysis-using-scapy, folder
  `Phishing-Website-Detection/datasets/` (see §9 for the full credit).
- **Total:** 2,015 rows, almost perfectly balanced (50.5% legitimate,
  49.5% phishing), so no class-balancing was needed.
- **Split:** 70/30 stratified — 1,410 train / 605 test (305 Legitimate,
  300 Phishing in test), `random_state=42`, so the run is reproducible.
- **No missing values** in the features or labels.

### The 13 features (all coded 0/1/2 in the source CSVs)

| Feature | What it measures | Why it helps |
|---|---|---|
| `Having_@_symbol` | '@' in the URL | '@' lets an attacker hide the real host |
| `Having_IP` | Host is an IP address, not a name | Real sites use domain names |
| `Prefix_suffix_separation` | '-' in the domain | `paypal-secure.com` is not `paypal.com` |
| `Redirection_//_symbol` | '//' after the protocol part | Redirection trick |
| `Sub_domains` | How many sub-domains | Phishing URLs stack sub-domains |
| `URL_Length` | Length of the URL, coded 0/1/2 | Phishing URLs tend to be long |
| `age_domain` | Domain age | Phishing domains are usually very new |
| `dns_record` | DNS record exists? | Fake sites often have none |
| `domain_registration_length` | Registration length | Short registrations look suspicious |
| `http_tokens` | 'http' token inside the URL | Token tricks in the domain/path |
| `statistical_report` | URL in a phishing report list | Known-bad indicator |
| `tiny_url` | URL shortener used | Hides the real destination |
| `web_traffic` | Web-traffic rank | Real sites have traffic; new fakes do not |

Text columns `Domain`, `Path` and `Protocol` were **not** used as
features: Domain/Path are identifiers, and Protocol is almost always
"http" in the legitimate file, so it would leak which file a row came
from instead of teaching the model what phishing looks like.

## 5. Models

- **Random Forest** — 100 decision trees, each trained on a bootstrap
  sample with random feature subsets; the trees vote, and the vote share
  is the probability. A forest beats one tree because averaging cancels
  out individual trees' mistakes (variance reduction). It also reports
  feature importance.
- **Logistic Regression** — the simple baseline: one linear decision
  boundary, outputs a probability. Features are standardised first
  (the scaler is part of the pipeline, so prediction uses the same
  scaling). It is weaker here because phishing is non-linear: "long URL
  OR '@' symbol AND young domain" is a combination of conditions, which
  trees express naturally and a straight line does not.

## 6. Results — held-out test set, run 20 Nov 2026

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Random Forest** | **0.8083** | 0.8145 | 0.8077 | 0.8071 |
| Logistic Regression | 0.7240 | 0.7281 | 0.7234 | 0.7224 |

Random Forest confusion matrix (rows = true, cols = predicted):

```
                 predicted
                 Legitimate  Phishing
true Legitimate      268        37
     Phishing         79       221
```

Per class (Random Forest): Legitimate — precision 0.77, recall 0.88,
F1 0.82 (support 305); Phishing — precision 0.86, recall 0.74, F1 0.79
(support 300). Logistic Regression confusion matrix: [[242, 63],
[104, 196]].

**Read this honestly in the viva:** 80.8% is a real, believable score
for 13 coded features and 2,015 rows — it is not 100%, and that is a
strength of this project, not a weakness. The most important number is
phishing recall: 0.74 means **79 of 300 phishing sites were missed**
and called legitimate. For a security tool, a missed phishing site (a
false negative — a user trusts a fake site) is worse than a false alarm
on a safe site. If the model must be tuned one way, tune it for phishing
recall first.

Random Forest feature importances: URL_Length 0.2229, web_traffic
0.1893, statistical_report 0.1542, dns_record 0.0977, Sub_domains
0.0782, age_domain 0.0757, domain_registration_length 0.0646,
Prefix_suffix_separation 0.0519, tiny_url 0.0483, Having_@_symbol
0.0059, Having_IP 0.0058, Redirection_//_symbol 0.0038, http_tokens
0.0016. Length and traffic dominate; the '@' and IP tricks matter far
less than tutorials suggest, because they are rare in this data.

### A small batch check (also measured, not hidden)

`python predict.py --csv data/batch_sample.csv` on 5 held-out-style rows
got **3/5 correct**: two rows were misclassified (one legitimate site
called phishing, one phishing site called legitimate). That is consistent
with 80.8% test accuracy — on 5 rows you expect about 1 error, and 2 is
within normal variation. Single-URL examples also behaved sensibly:
`https://www.google.com` → Legitimate, P(phishing) = 0.2215; a crafted
`@` + hyphen URL → Phishing, P(phishing) = 0.9408. But see the caveat
in §7: single-URL predictions are approximate.

## 7. How to run it (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python predict.py --csv data/batch_sample.csv
python predict.py --url "https://www.google.com"
```

(Linux/Mac: `python3` instead of `python`, `source .venv/bin/activate`.)

**Single-URL caveat:** a raw URL string can only give 8 of the 13
features. The other 5 — `age_domain`, `dns_record`,
`domain_registration_length`, `statistical_report`, `web_traffic` —
need DNS / WHOIS / traffic data, which this project does not fetch, so
`predict.py` sets them to 0 and prints a warning. The 0/1/2 codes for
`Sub_domains` and `URL_Length` are approximations too: the source CSVs
do not document their exact thresholds, so the script uses the commonly
used UCI thresholds (URL <54 chars = 0, 54–75 = 1, >75 = 2). Batch
`--csv` with the full 13-feature row is the reliable mode.

## 8. Limitations (state these in the report — teachers trust them)

- 80.8% accuracy on a 605-row test set; phishing recall is only 0.74
  (79 missed phishing sites). This must not be presented as a
  production-grade detector.
- Only 13 pre-coded features and 2,015 rows; the dataset is old and
  small, and modern phishing pages may look different.
- Single-URL predictions are approximate (5 features assumed 0).
- The model never fetches a live website — it classifies features, it
  does not verify, block, or protect anything in a browser.
- Logistic Regression vs Random Forest was compared on one 70/30 split
  only; no cross-validation was run yet (see future work).

## 9. Future work

Add cross-validation (several splits, averaged) instead of one split;
add the 0–100 regression risk score; try XGBoost and compare on the
same split; collect a larger, newer dataset; build the full 13-feature
row for a live URL (WHOIS + DNS + traffic rank) so single-URL checks
stop being approximate.

## 10. Likely viva questions (short answers)

- **"Why is accuracy only 80.8%, not 99%?"** — Because the classes
  overlap on these 13 features and the dataset is small and old. A
  99% claim on this data would be suspicious; 80.8% is what the model
  actually measured on 605 unseen websites.
- **"Why Random Forest, not one decision tree?"** — A forest averages
  100 trees, so one noisy feature cannot decide the verdict alone, and
  it reports feature importance (URL length matters most here).
- **"Why is Logistic Regression worse?"** — Phishing is a combination
  of conditions, not a straight line; trees express combinations
  naturally, a linear boundary does not.
- **"What is the worst mistake the model makes?"** — Missing a phishing
  site (recall 0.74): 79 of 300 phishing sites were called legitimate.
  In security, that false negative is worse than a false alarm.
- **"Where is the ML — isn't this just rules?"** — The forest trains on
  labelled data, its accuracy and confusion matrix are computed from a
  held-out test set, and changing the inputs changes the verdict. No
  URL is pre-marked; the batch check proves it (2 of 5 rows were
  misclassified, which a hard-coded list would never do).
- **"Did you copy this project?"** — No. The two datasets are public
  data from jishnusaurav's repo (below); the training code, prediction
  code, README and this report were written for this project.

## 11. Source credit

Datasets `legitimate-urls.csv` and `phishing-urls.csv`:
**jishnusaurav/Phishing-attack-PCAP-analysis-using-scapy**,
folder `Phishing-Website-Detection/datasets/`, GitHub
(https://github.com/jishnusaurav/Phishing-attack-PCAP-analysis-using-scapy).
Downloaded from the repo's raw file URLs on 20 Nov 2026. All other
files in this project (`train.py`, `predict.py`, `README.md`, this
report) were written for this project and are not copies of the
original author's notebook.
