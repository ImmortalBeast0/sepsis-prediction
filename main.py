"""
Sepsis Prediction Pipeline — Phase 1 Baseline Execution Entrypoint.

Runs the complete Phase 1 sequence:
1. Load dataset (src/data_loader.py)
2. Perform Exploratory Data Analysis & Preprocessing (src/preprocess.py)
3. Train Random Forest Baseline Model (src/train.py)
4. Evaluate Model with Accuracy & Confusion Matrix (src/evaluate.py)
"""

import os
import sys

# Ensure src module can be imported cleanly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_sepsis_data
from src.preprocess import perform_eda, preprocess_data
from src.train import train_baseline_model
from src.evaluate import evaluate_model


def main():
    print("============================================================")
    print("     SEPSIS PREDICTION PIPELINE (PHASE 1 - BASELINE)        ")
    print("============================================================")
    
    # Step 1: Load clinical ICU dataset
    raw_df = load_sepsis_data(filepath="data/sepsis_dataset.csv")
    
    # Step 2: Exploratory Data Analysis & Data Preprocessing
    perform_eda(raw_df)
    X_clean, y, feature_cols = preprocess_data(raw_df)
    
    # Step 3: Baseline Random Forest Model Training
    model, X_test, y_test = train_baseline_model(X_clean, y)
    
    # Step 4: Baseline Model Evaluation
    results = evaluate_model(model, X_test, y_test)
    
    print("[SUCCESS] Phase 1 baseline pipeline execution completed.")
    print("============================================================")


if __name__ == "__main__":
    main()
