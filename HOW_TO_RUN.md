# Sepsis Prediction Framework — USAGE & GITHUB GUIDE

This guide explains:
1. **How to create and push this project to a new public GitHub repository**
2. **How to set up and run the project on any computer**

---

## Part 1: How to Push This Project to GitHub

Follow these simple steps to add this repository to your GitHub profile:

### Step 1: Create a New Repository on GitHub
1. Go to [GitHub.com](https://github.com) and log in to your account.
2. Click the **`+`** icon in the top right corner and select **New repository**.
3. Fill in the repository details:
   - **Repository name:** `sepsis-prediction-phase1` *(or any name you prefer)*
   - **Description:** `Early Sepsis Prediction Framework — Core ML Prediction Pipeline (Phase 1 Baseline)`
   - **Public / Private:** Select **Public**
   - **Initialize repository with:** **DO NOT** check "Add a README file", ".gitignore", or "Choose a license" *(leave them unchecked as we already have local files)*.
4. Click **Create repository**.

### Step 2: Push Local Code to GitHub

Open terminal in this project folder (`E:\Desktop\Final Year Project`) and run:

```bash
# 1. Rename default branch to main (if not done already)
git branch -M main

# 2. Link your local project to your new GitHub repository
# Replace <YOUR_GITHUB_USERNAME> with your actual GitHub username
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/sepsis-prediction-phase1.git

# 3. Push all commits to GitHub
git push -u origin main
```

---

## Part 2: How to Run the Project (Step-by-Step)

### Step 1: Open Terminal in Project Directory
```powershell
cd "E:\Desktop\Final Year Project"
```

### Step 2: Set Up Python Environment & Install Dependencies

```powershell
# Create a virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required ML libraries
pip install -r requirements.txt
```

### Step 3: Execute Sepsis Prediction Pipeline

Run the main pipeline entrypoint:

```powershell
python main.py
```

---

## Modular Execution Options

You can also run individual components of the pipeline standalone:

```powershell
# 1. Test Data Loader module
python src/data_loader.py

# 2. Test Preprocessing & EDA report module
python src/preprocess.py

# 3. Test Random Forest Model Training module
python src/train.py

# 4. Test Evaluation Metrics module
python src/evaluate.py

# 5. Regenerate Clinical Dataset CSV
python scripts/create_sepsis_data.py
```

---

## What the Pipeline Does When Executed

1. **Loads Dataset:** Ingests clinical ICU vitals from `data/sepsis_dataset.csv`.
2. **Exploratory Data Analysis (EDA):** Prints class balance stats (~10.6% positive sepsis prevalence), missing value counts, and feature correlations.
3. **Data Preprocessing:** Imputes missing clinical vitals using median statistics.
4. **Model Training:** Splits data (80% train / 20% test) and trains a baseline Random Forest Ensemble Classifier.
5. **Model Evaluation:** Reports overall accuracy score and confusion matrix metrics.
