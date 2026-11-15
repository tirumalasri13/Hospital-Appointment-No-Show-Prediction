# Technical Report: Hospital Appointment No-Show Prediction

Candidate: Ampolu Tirumala Sri | Reg. no: 12312672 | Role: AI & Machine Learning
Demo video: https://www.loom.com/share/1205d1910e304da3bf48f5adf1a537f6

All metrics below were computed directly by running `python -m src.train` with fixed random seed 42.

---

## 1. Problem Definition
* **Decision Supported:** Identification of scheduled hospital appointments at high risk of a no-show
  so administrative staff can schedule human-reviewed reminder phone calls or plan booking buffers.
* **Unit of Prediction:** One scheduled appointment. Target is `no_show` (1 = missed, 0 = attended).
* **Prediction Time:** The exact point of booking. Only features known at booking time are utilized.
* **Out of Scope:** Clinical diagnoses, clinical urgency, medical triage, or automated cancellations.

---

## 2. Dataset and Quality Summary
* **Generator (`src/generate_data.py`):** 12,000 synthetic appointment records with 18 columns
  (shape: `(12000, 18)`, comprising IDs, timestamps, target, and 14 original candidate features,
  expanded to 16 with engineered features) for 3,000 unique patients spanning 2024-01-01 to
  2025-06-23 (seed 42). Overall no-show prevalence is 18.57% (class imbalance: 9,771 attended vs
  2,229 missed).
* **Deliberately Injected Data Quality Issues:**
  - `distance_km`: 262 missing entries (2.18%).
  - `booking_channel`: 223 missing entries (1.86%).
  - Impossible ages (>110 years): 36 records. Negative lead time: 0 records.
* **Cleaning Pipeline (`src/features.py`):**
  - Impossible ages (>110) are coerced to NaN and imputed using median training age (40.0 years).
  - Missing numeric distances are median-imputed; missing categorical values are mapped to `"unknown"`.
  - Continuous numeric features are standardized via `StandardScaler`.
* **Exploratory Observations (`reports/eda_no_show_rates.png`, `notebooks/eda.ipynb`):**
  - No-show rate climbs with longer lead time: 11.42% for lead time <= 3 days (n = 1,515) to 35.06%
    for lead time > 30 days (n = 793), compared to 17.41% for lead time <= 30 days (n = 11,207).
  - Prior misses strongly predict future no-shows: 44.03% for >= 3 prior no-shows (n = 268) vs
    17.99% for < 3 prior misses (n = 11,732).
  - Reminders lower no-show probability: 14.52% with scheduled reminder (n = 7,247) vs 24.76% without
    (n = 4,753) — an absolute difference of 10.24 percentage points.
  - Age stratification: pediatric/adolescent patients (`0-17`) are the highest-risk group (23.58% rate,
    n = 1,370), followed by young adults `18-39` (20.35%, n = 4,530), mature adults `40-59` (17.42%,
    n = 4,500), and seniors `60+` who exhibit the lowest no-show rate (12.50%, n = 1,600).

![Exploratory Data Analysis - No-Show Rates](../reports/eda_no_show_rates.png)

---

## 3. Leakage Control
* **Temporal Precedence:** Patient history counters (`prev_visits`, `prev_no_shows`) are computed
  strictly from appointments dated *prior to the current booking date* (verified in unit tests).
* **Code Guard:** `src/features.assert_no_forbidden()` blocks post-appointment columns
(`appointment_status`,
  `check_in_time`, `wait_minutes`) and clinical fields (`diagnosis`, `treatment`, `urgency`).
* **Protected Attributes:** `gender` is excluded from model training and reserved solely for
fairness audits.

---

## 4. Feature Engineering
The model consumes 8 numeric and 8 categorical features known at booking time:
* **Numeric (8):** `age`, `distance_km`, `lead_time_days`, `appt_hour`, `prev_visits`, `prev_no_shows`,
  `prev_no_show_rate`, and `prev_attended`.
* **Categorical (8):** `appt_weekday`, `department`, `booking_channel`, `reminder_scheduled`,
  `is_follow_up`, `hour_bucket`, `lead_bucket`, and `is_first_visit`.

### Notes on New Features (`prev_attended` and `is_first_visit`)
Two supplementary features were engineered:
1. `prev_attended = max(0, prev_visits - prev_no_shows)`
2. `is_first_visit = (prev_visits == 0)`
Both are deterministic functions of existing visit counters. Incorporating them did not change the test set
discrimination (test PR-AUC remained identical at 0.466 before and after). The slight change in expected
business cost (from 1900.0 to 1870.0) was driven by the validation-selected decision threshold
adjusting from
0.17 to 0.16, rather than an inherent increase in model ranking capability.

---

## 5. Chronological Split Strategy
To emulate live operational deployment without temporal leakage, data was partitioned chronologically
by appointment date:
* **Training Set (70%):** 8,400 rows | 2024-01-01 to 2025-01-14 | 1,560 no-shows (18.6%)
* **Validation Set (15%):** 1,800 rows | 2025-01-14 to 2025-04-02 | 330 no-shows (18.3%)
* **Test Set (15%):** 1,800 rows | 2025-04-02 to 2025-06-23 | 339 no-shows (18.8%)

Hyperparameter tuning and threshold selection were conducted exclusively on the validation set.
The held-out test set was evaluated once at the very end.

---

## 6. Model Exploration and Training

### Candidate Models
1. **Rule Baseline:** Flags appointments where `lead_time_days >= 21` or `prev_no_shows >= 2`.
2. **Flag Everyone Policy:** Trivial policy flagging all appointments (costs 2 units per patient).
3. **Dummy Prior Classifier:** Always predicts the majority class prevalence.
4. **Logistic Regression:** Linear classifier with `class_weight="balanced"`.
5. **Random Forest:** Ensemble of 300 trees with `class_weight="balanced_subsample"`.
6. **XGBoost:** Gradient boosted trees with `scale_pos_weight=4.38` (ratio of attenders to no-shows).
7. **HistGradientBoosting:** Histogram-based boosting tuned over an 8-configuration grid on
validation PR-AUC.

### Analysis of XGBoost Performance
XGBoost achieved a validation PR-AUC of 0.327 and a test PR-AUC of 0.418, placing behind Logistic Regression
(0.365 val / 0.469 test). XGBoost was evaluated with fixed baseline hyperparameters and was not extensively
grid-searched. More importantly, the synthetic ground truth was generated using a linear combination
of log-odds
passed through a sigmoid function. Consequently, a well-calibrated generalized linear model naturally aligns
with the true underlying data manifold.

---

## 7. Model Evaluation and Comparison

### Validation Comparison (`reports/model_comparison_validation.csv`)
Used exclusively to select the champion architecture and cost-minimizing threshold:

| Model | Threshold | Precision | Recall | F1 | PR-AUC | Brier | Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rule_baseline` | 0.50 | 0.308 | 0.448 | 0.365 | 0.239 | 0.286 | 2486.0 |
| `flag_everyone` | 0.50 | 0.183 | 1.000 | 0.310 | 0.183 | 0.817 | 2940.0 |
| `dummy_prior` | 0.50 | 0.000 | 0.000 | 0.000 | 0.183 | 0.150 | 3300.0 |
| **logreg (Sel)** | **0.40** | **0.262** | **0.752** | **0.388** | **0.365** | **0.210** | **2218.0** |
| `random_forest` | 0.44 | 0.280 | 0.676 | 0.396 | 0.355 | 0.198 | 2216.0 |
| `xgboost` | 0.43 | 0.279 | 0.658 | 0.391 | 0.327 | 0.202 | 2254.0 |
| `grad_boosting` | 0.39 | 0.261 | 0.782 | 0.391 | 0.342 | 0.211 | 2184.0 |

*Winner:* Logistic Regression achieved the highest validation PR-AUC (0.365) and was selected.

### Final Held-Out Test Comparison (`reports/model_comparison_test.csv`)
Evaluated once on untouched future appointments (n = 1,800, 339 no-shows):

| Model | Threshold | Precision | Recall | F1 | PR-AUC | Brier | FP | FN | Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rule_baseline` | 0.50 | 0.319 | 0.575 | 0.410 | 0.263 | 0.312 | 417 | 144 | 2274.0 |
| `flag_everyone` | 0.50 | 0.188 | 1.000 | 0.317 | 0.188 | 0.812 | 1461 | 0 | 2922.0 |
| `dummy_prior` | 0.50 | 0.000 | 0.000 | 0.000 | 0.188 | 0.153 | 0 | 339 | 3390.0 |
| `logreg` (raw) | 0.40 | 0.285 | 0.841 | 0.426 | 0.469 | 0.204 | 714 | 54 | 1968.0 |
| `random_forest` | 0.44 | 0.306 | 0.788 | 0.441 | 0.447 | 0.194 | 606 | 72 | 1932.0 |
| `xgboost` | 0.43 | 0.306 | 0.711 | 0.428 | 0.418 | 0.193 | 546 | 98 | 2072.0 |
| `grad_boosting` | 0.39 | 0.274 | 0.820 | 0.411 | 0.443 | 0.203 | 736 | 61 | 2082.0 |
| **FINAL** | **0.16** | **0.309** | **0.811** | **0.448** | **0.466** | **0.128** | 615 | 64 | 1870.0 |

*Confusion Matrix (Final Calibrated Model at Threshold 0.16):*
* True Negatives (TN): **846**
* False Positives (FP): **615**
* False Negatives (FN): **64**
* True Positives (TP): **275**
* Total Expected Business Cost: **1,870.0 units** (64 x 10 + 615 x 2)

![Confusion Matrix](../reports/confusion_matrix.png)

---

## 8. Calibration and Threshold Analysis
* **Probability Calibration:** Balanced class weights distort raw log-odds, inflating raw probabilities
  (validation Brier score of 0.2101). Post-hoc isotonic regression on 3-fold cross-validation
reduced validation
  Brier score to 0.1373 and test Brier score to 0.1282, ensuring predicted probabilities mirror empirical
  observed risk.

![Calibration Curve](../reports/calibration_curve.png)

* **Cost Assumptions:** A missed no-show wastes a clinic appointment slot (COST_FN = 10.0); an
unnecessary
  reminder requires staff outreach (COST_FP = 2.0). Minimizing expected cost on the validation
set yielded
  an optimal operating threshold of **0.16**.

![Decision Threshold vs Expected Cost](../reports/threshold_curve.png)

### Cost Sensitivity Analysis (`reports/cost_sensitivity.csv`)
Because cost weights are administrative assumptions, we audited how varying (COST_FN, COST_FP)
shifts the
optimal threshold:

| Cost FN | Cost FP | Ratio | Threshold | Precision | Recall | Flagged | Test Cost |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10.0 | 2.0 | 5.0 | **0.16** | 0.309 | 0.811 | 890 (49.4%) | 1870.0 |
| 10.0 | 1.0 | 10.0 | **0.09** | 0.240 | 0.929 | 1311 (72.8%) | 1236.0 |
| 10.0 | 5.0 | 2.0 | **0.35** | 0.504 | 0.342 | 230 (12.8%) | 2800.0 |
| 20.0 | 2.0 | 10.0 | **0.09** | 0.240 | 0.929 | 1311 (72.8%) | 2472.0 |
| 5.0 | 2.0 | 2.5 | **0.35** | 0.504 | 0.342 | 230 (12.8%) | 1343.0 |
| 10.0 | 10.0 | 1.0 | **0.48** | 0.685 | 0.180 | 89 (4.9%) | 3060.0 |

*Operational Takeaway:* Under our baseline (10, 2), the model flags 890 appointments (~49.4% of the
schedule)
to catch 81.1% of no-shows. If clinic phone staff capacity is constrained, setting COST_FP = 5.0
increases
the threshold to 0.35, flagging only 230 patients (12.8%) with much higher precision (50.4%).

---

## 9. Feature Importance and Model Explainability
* **Global Importance (`reports/permutation_importance.csv`):** Permutation importance on the test set
  showed that `lead_time_days` (+0.0673 PR-AUC drop), `prev_no_shows` (+0.0619), and `reminder_scheduled`
  (+0.0562) exert the strongest influence on risk predictions.
* **Local Explainability:** Each API inference call computes an occlusion explanation, replacing each input
  with its training median/mode to determine whether that feature raises or lowers patient risk.

### Worked Prediction Example

To see how the model works on a single case, I sent it one appointment. The
patient is 27 years old and booked a Dental visit for a Monday at 8 am,
34 days in advance, by phone. No reminder was scheduled, the patient lives
18.5 km away, and has 4 earlier visits of which 2 were missed. The model gave
a no-show probability of 79.84%. My chosen threshold is 0.16, so this
appointment is flagged as high risk. The model also shows why. Having no
reminder raised the risk the most (+0.204), then the long 34-day wait (+0.079),
then the 2 earlier no-shows (+0.048). These numbers show how the model's answer
changes if one input is replaced by a typical value. For example, with a
reminder the estimate would be about 0.20 lower. They do not prove that a
reminder causes people to attend. The flag only helps staff decide whether to
make a reminder call or send an SMS, and a staff member makes that decision.

### False-Negative Error Analysis

A false negative is a patient who missed the appointment although the model did
not flag it. One example from my test set is an 84-year-old booked for
Physiotherapy on a Monday at 1 pm, only 3 days ahead, with a reminder scheduled
and 2 earlier visits, none missed. The model predicted a risk of only 4.9%, far
below the 0.16 threshold, yet the patient did not come. The model had good
reasons to think this patient would attend: the age and the short wait each
lowered the risk by about 0.03, and the clean attendance history lowered it by
about 0.014. In my synthetic data, patients aged 60 and over miss appointments
less often (12.5%), so the model learned that pattern. A 4.9% risk still means
about 1 in 20 similar patients will miss, and the data contains nothing about
things like sudden illness or transport problems, so the model cannot catch
this case. Each missed appointment costs 10 units in my cost assumption.
Lowering the threshold enough to catch cases like this would flag far more than
the 49% of appointments already flagged. In a real hospital I would add
information known at booking, such as whether the patient responded to the
reminder, and I would check the model regularly on real data.

---

## 10. Subgroup Fairness and Responsible AI Auditing

### Age Group Fairness (`reports/fairness_age_group.csv`)
| Age Group | Sample Size (n) | Observed No-Show Rate | Flag Rate | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0-17 | 208 | 0.269 | 0.630 | 0.389 | 0.911 |
| 18-39 | 663 | 0.205 | 0.555 | 0.329 | 0.890 |
| 40-59 | 670 | 0.173 | 0.469 | 0.280 | 0.759 |
| 60+ | 259 | 0.120 | 0.297 | 0.195 | 0.484 |

*Finding:* Recall for patients aged 60+ is 0.484, compared to 0.890 for ages 18-39. Older patients
have a lower
baseline no-show rate (12.0%) and fewer past missed visits, causing the model to flag them less frequently.
*Mitigation:* Before live clinical deployment, clinic operations should implement age-specific
monitoring and
consider adjusted reminder policies for senior demographics.

### Gender Audit (`reports/fairness_gender.csv`)
| Gender | Sample Size (n) | Observed No-Show Rate | Flag Rate | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Female (F) | 904 | 0.173 | 0.475 | 0.291 | 0.801 |
| Male (M) | 896 | 0.204 | 0.515 | 0.325 | 0.820 |

*Finding:* Model recall is equitable between male (0.820) and female (0.801) cohorts.

---

## 11. Known Limitations and Production Risks
1. **Synthetic Nature:** Results prove end-to-end pipeline integrity, not real-world clinical efficacy.
2. **Cold Start for New Patients:** Patients with zero past history rely entirely on demographic and
scheduling
   features, where uncertainty is inherently higher.
3. **Distribution Shift:** Changes in clinic reminder protocols or seasonal attendance require continuous
   monitoring of Brier calibration and periodic model retraining.
4. **Mandatory Human Oversight:** Predictions are strictly decision support; no automated cancellations or
   punitive scheduling actions may be triggered without staff verification.

---

## 12. Complete Number Audit Table
Every single quantitative value in this report was verified against source artifact files on disk:

| Report Metric / Quantity | Value | Source File on Disk |
| :--- | :---: | :--- |
| Total appointments generated | 12,000 | `reports/data_quality.txt` |
| Unique patients generated | 3,000 | `src/generate_data.py` / `data/appointments.csv` |
| Dataset overall no-show rate | 18.57% | `reports/data_quality.txt` |
| Missing `distance_km` values | 262 | `reports/data_quality.txt` |
| Missing `booking_channel` values | 223 | `reports/data_quality.txt` |
| Impossible ages (>110) | 36 | `reports/data_quality.txt` |
| Appointment date range | 2024-01-01 to 2025-06-23 | `reports/data_quality.txt` |
| Train split size and dates | 8,400 (2024-01-01 to 2025-01-14) | `models/model_meta.json` |
| Validation split size and dates | 1,800 (2025-01-14 to 2025-04-02) | `models/model_meta.json` |
| Test split size and dates | 1,800 (2025-04-02 to 2025-06-23) | `models/model_meta.json` |
| Validation PR-AUC (`logreg`) | 0.365 | `reports/model_comparison_validation.csv` |
| Validation PR-AUC (`random_forest`) | 0.355 | `reports/model_comparison_validation.csv` |
| Validation PR-AUC (`grad_boosting`) | 0.342 | `reports/model_comparison_validation.csv` |
| Validation PR-AUC (`xgboost`) | 0.327 | `reports/model_comparison_validation.csv` |
| Validation Brier (raw -> calibrated) | 0.2101 -> 0.1373 | `models/model_meta.json` |
| Optimal validation threshold | 0.16 | `models/model_meta.json` |
| Test Precision (`logreg+isotonic`) | 0.309 | `reports/model_comparison_test.csv` |
| Test Recall (`logreg+isotonic`) | 0.811 | `reports/model_comparison_test.csv` |
| Test F1 (`logreg+isotonic`) | 0.448 | `reports/model_comparison_test.csv` |
| Test PR-AUC (`logreg+isotonic`) | 0.466 | `reports/model_comparison_test.csv` |
| Test Brier score (`logreg+isotonic`) | 0.128 | `reports/model_comparison_test.csv` |
| Confusion Matrix (TN, FP, FN, TP) | 846, 615, 64, 275 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`logreg+isotonic`)| 1870.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`logreg` raw) | 1968.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`random_forest`) | 1932.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`xgboost`) | 2072.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`grad_boosting`) | 2082.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`rule_baseline`) | 2274.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`flag_everyone`) | 2922.0 | `reports/model_comparison_test.csv` |
| Test Expected Cost (`dummy_prior`) | 3390.0 | `reports/model_comparison_test.csv` |
| Fairness age 60+ recall | 0.484 | `reports/fairness_age_group.csv` |
| Fairness age 18-39 recall | 0.890 | `reports/fairness_age_group.csv` |
| Fairness female recall | 0.801 | `reports/fairness_gender.csv` |
| Fairness male recall | 0.820 | `reports/fairness_gender.csv` |
| Cost sensitivity (10, 2) flagged / thr | 890 (49.4%) / 0.16 | `reports/cost_sensitivity.csv` |
| Cost sensitivity (10, 5) flagged / thr | 230 (12.8%) / 0.35 | `reports/cost_sensitivity.csv` |
| Permutation top feature | `lead_time_days` (+0.0673) | `reports/permutation_importance.csv` |

