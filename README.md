# Hospital Appointment No-Show Prediction

Demo video: https://www.loom.com/share/c0a63371dfc845c79ccba46e50c89501

Predicts the probability that an appointment will be missed, so staff can plan human-reviewed reminders
and scheduling buffers. It never predicts diagnosis, urgency or clinical outcomes.

* **Prediction unit:** one appointment
* **Target:** `no_show` (1 = missed)
* **Prediction time:** when the appointment is booked (only information known then is used)
* **Data:** fully synthetic (`src/generate_data.py`, seeded) - no real patient data

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Calibration:** Isotonic Regression
- **Backend:** FastAPI
- **Data Validation:** Pydantic
- **Model Serialization:** Joblib
- **Dashboard:** Streamlit
- **API Documentation:** Swagger UI
- **Testing:** Pytest

## End-to-End Architecture

```
[ appointments.csv ]
         |
         v
[ src/features.py ] -------> [ Derives: prev_attended, is_first_visit; OneHot + Scaler ]
         |
         v
[ src/train.py ] ----------> [ Chronological Split -> 4 Models -> Calibrate -> Threshold ]
         |
         v
[ models/model.joblib ] ---> [ Champion Pipeline (LogReg + Isotonic) + Threshold (0.16) ]
         |
         v
[ api/app.py (FastAPI) ] --> [ Live REST Endpoint (/predict, /predict_batch) + Explanations ]
         |
         v
[ Staff Review ] ----------> [ Clinical Administrative Staff Manual Triage (Dashboard) ]
```

## Saved Model Artifacts

All trained model artifacts and associated metadata are versioned in `models/`:
* `models/model.joblib`: Complete serialized inference bundle containing:
  - `model`: Calibrated Classifier (Logistic Regression + Isotonic Calibrator via 3-fold CV)
  - `preprocessor`: ColumnTransformer fitted on training split (OneHotEncoder + StandardScaler)
  - `threshold`: Cost-optimal operating threshold (**0.16**) chosen on validation set (FN=10, FP=2)
  - `feature_names`: Exact list of expected raw feature inputs
* `models/model_meta.json`: Full reproducibility metadata including train/validation/test date ranges,
  sample sizes, no-show counts, validation Brier scores before/after calibration, hyperparameters,
  and test metrics.

## Setup


# Create virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

## Reproduce everything (from the repo root)

```bash
python -m src.generate_data     # 1. create data/appointments.csv (seed=42)
python -m src.eda               # 2. EDA + data-quality report -> reports/
python -m src.train             # 3. split, baselines, 4 models, tuning, calibration, test, save
python -m src.cost_sensitivity  # 4. cost sensitivity analysis -> reports/cost_sensitivity.csv
python -m src.export_report_pdf # 5. export technical report with plots -> docs/TECHNICAL_REPORT.pdf
python -m pytest -q             # 6. 14 unit and integration tests
```

## Exploratory Data Analysis Notebook

An interactive Jupyter notebook with visual and tabular exploratory analysis is provided:
* `notebooks/eda.ipynb`: Pre-executed notebook examining dataset dimensions (shape: `(12000, 18)`),
  schema, missingness, impossible age coercion, class balance (18.57%), and bivariate no-show rates
  across lead time (>30 days: 35.06%), prior history (>=3 no-shows: 44.03%), reminders (unreminded: 24.76%
  vs reminded: 14.52%), and age groups (highest risk: 0-17 at 23.58%; lowest risk: 60+ at 12.50%).
* To launch:
  ```bash
  jupyter notebook notebooks/eda.ipynb
  ```

## Run the inference interface

```bash
uvicorn api.app:app --reload          # API + Swagger UI at http://127.0.0.1:8000/docs
streamlit run api/dashboard.py        # optional staff dashboard
```

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
 "age":27,"distance_km":18.5,"lead_time_days":34,"appt_weekday":"Monday","appt_hour":8,
 "department":"Dental","booking_channel":"phone","reminder_scheduled":"no","is_follow_up":"no",
 "prev_visits":4,"prev_no_shows":2}'
```

Every response loads the saved model, returns a probability, risk band, top factors and a human-review
disclaimer.

## Method summary

Chronological 70/15/15 split -> baselines (majority/prior, business rule, flag-everyone) -> Logistic
Regression, Random Forest, XGBoost, HistGradientBoosting (class weights/scale_pos_weight for imbalance; grid
tuned on validation only) -> isotonic calibration -> threshold chosen on validation by minimising expected
cost (FN=10, FP=2) -> one final evaluation on the untouched test set. See `docs/leakage_audit.md` and
`docs/TECHNICAL_REPORT.md`.

## Limitations

Synthetic data (results show the pipeline works, not real-world accuracy); simple costs are assumptions;
recall is lower for the 60+ group (see `reports/fairness_age_group.csv`); new patients have no history;
needs monitoring/re-training for drift. Human review is mandatory.

## AI Assistance Statement

## AI Assistance and Contributions

AI tools, including ChatGPT, Claude, and the Antigravity coding agent, were used to assist with project planning, code development, debugging, and report drafting.

My contributions included setting up and running the project locally, testing the FastAPI backend, executing prediction requests through Swagger UI, reviewing the model outputs, updating the technical report, and documenting the project demonstration.

I reviewed the implementation and results to understand the workflow and can explain the major components of the project.

Original project author and repository history are retained where applicable.

