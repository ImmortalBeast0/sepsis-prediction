# Early Sepsis Prediction Framework — Core ML Pipeline (Phase 1 Baseline)

An Machine Learning framework designed for early sepsis prediction using clinical Intensive Care Unit (ICU) patient vitals. Sepsis is a rapid-onset, life-threatening clinical emergency where early prediction can significantly reduce mortality rates. This repository hosts Phase 1 of the research project, focusing on core tabular machine learning prediction using ensemble methods to classify sepsis risk from physiological vital signs.

> **Current Status: Work-in-Progress (~40% Completed)**  
> This repository represents an early-stage, functional baseline of the planned three-phase prescriptive AI framework.

---

## Current Project Status

### What is Completed (Phase 1 Baseline — ~40%)
- [x] **Project Repository & Environment Setup:** Standard directory layout (`data/`, `src/`, `scripts/`, `requirements.txt`, `.gitignore`).
- [x] **Data Ingestion:** Module to load clinical ICU vital signs (`src/data_loader.py`) from CSV format.
- [x] **Exploratory Data Analysis (EDA):** Automated summary logging of target class imbalance, missing value statistics, feature distributions, and correlations (`src/preprocess.py`).
- [x] **Data Preprocessing & Imputation:** Median imputation strategy for handling missing ICU vital sign telemetry (`src/preprocess.py`).
- [x] **Baseline Ensemble Model Training:** Random Forest Classifier baseline (`src/train.py`) with balanced class weights for sepsis classification.
- [x] **Basic Performance Evaluation:** Reporting of overall model accuracy and confusion matrix metrics (`src/evaluate.py`).

### What is NOT Done Yet (Out of Scope / Planned Future Phases)
- [ ] **Phase 2 — Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) and LIME feature importance engines for diagnostic clinical interpretability.
- [ ] **Phase 3 — Counterfactual Reasoning:** Counterfactual optimization (DiCE algorithm) for actionable, prescriptive treatment recommendations.
- [ ] **Hyperparameter Tuning & Model Comparison:** Comparative evaluation against XGBoost, LightGBM, or CatBoost ensembles.
- [ ] **User Interface / Web Dashboard:** Streamlit / web interface for clinical decision support deployment.

---

## Dataset Source & Information

- **Primary Dataset:** Kaggle Sepsis / ICU Patient Dataset (derived from the PhysioNet / Computing in Cardiology Challenge 2019).
- **Source Link:** [Kaggle Sepsis Dataset](https://www.kaggle.com/datasets/salmanfaris/sepsis-prediction-dataset) / [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/)
- **Clinical Vital Features:**
  - `HR`: Heart Rate (beats per minute)
  - `Temp`: Body Temperature (°C)
  - `SBP`: Systolic Blood Pressure (mmHg)
  - `Resp`: Respiration Rate (breaths per minute)
  - `WBC`: White Blood Cell Count (10^3/uL)
  - `Age`: Patient Age (years)
  - `SepsisLabel`: Binary target (0 = Non-Sepsis, 1 = Sepsis)

---

## How to Run What Exists So Far

### 1. Environment Setup

Clone the repository and set up a Python 3.11+ virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate or Inspect Dataset

To generate the clinical vitals dataset (`data/sepsis_dataset.csv`):

```bash
python scripts/create_sepsis_data.py
```

### 3. Run the Full Sepsis Prediction Pipeline

Execute the main pipeline script to load data, run EDA, preprocess vitals, train the baseline Random Forest model, and print evaluation metrics:

```bash
python main.py
```

Or run individual module scripts standalone:

```bash
# Test data loading module
python src/data_loader.py

# Test preprocessing & EDA module
python src/preprocess.py

# Test model training module
python src/train.py

# Test evaluation metrics module
python src/evaluate.py
```
