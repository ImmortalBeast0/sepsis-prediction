"""
Data Preprocessing & Exploratory Data Analysis (EDA) Module.

This module provides simple, clear functions for:
1. Conducting basic Exploratory Data Analysis (EDA) on ICU vitals.
2. Imputing missing values using median statistics (robust to outliers).
3. Splitting features and targets for machine learning model training.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def perform_eda(df: pd.DataFrame, target_col: str = "SepsisLabel"):
    """
    Perform basic exploratory data analysis (EDA) and print summary statistics.
    
    Parameters:
        df (pd.DataFrame): Raw sepsis clinical dataframe.
        target_col (str): Name of the binary target column.
    """
    print("\n" + "="*60)
    print("      EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("="*60)
    
    # 1. Class Balance Analysis
    print("\n[EDA 1/3] Target Class Balance ('SepsisLabel'):")
    class_counts = df[target_col].value_counts()
    class_props = df[target_col].value_counts(normalize=True) * 100
    
    for cls in class_counts.index:
        label_str = "Sepsis (Positive)" if cls == 1 else "Non-Sepsis (Negative)"
        print(f"  - Class {cls} [{label_str}]: {class_counts[cls]} cases ({class_props[cls]:.2f}%)")
        
    # 2. Missing Values Analysis
    print("\n[EDA 2/3] Missing Values Report:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Percent': missing_pct})
    missing_features = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_features) > 0:
        print("  Found missing values in the following clinical vitals:")
        for col, row in missing_features.iterrows():
            print(f"    * {col}: {int(row['Missing_Count'])} missing values ({row['Missing_Percent']:.1f}%)")
    else:
        print("  No missing values detected.")
        
    # 3. Feature Distributions & Correlations
    numeric_cols = [c for c in df.columns if c not in ["Patient_ID", target_col]]
    print("\n[EDA 3/3] Feature Summary & Correlations with Sepsis Target:")
    
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        col_median = df[col].median()
        
        # Calculate Pearson correlation with SepsisLabel
        corr = df[col].corr(df[target_col])
        print(f"  - Feature '{col}': Mean={col_mean:.2f}, Std={col_std:.2f}, Median={col_median:.2f} | Sepsis Corr={corr:+.3f}")
        
    print("="*60 + "\n")


def preprocess_data(df: pd.DataFrame, target_col: str = "SepsisLabel"):
    """
    Preprocess clinical data by separating features and target, and handling missing values.
    
    Parameters:
        df (pd.DataFrame): Raw clinical dataframe.
        target_col (str): Target column name.
        
    Returns:
        tuple: (X_clean, y, feature_names) where X_clean is a clean DataFrame with imputed NaNs.
    """
    print("\n[INFO] Starting Data Preprocessing Pipeline...")
    
    # Drop identifier column if present (Patient_ID is not a predictive feature)
    feature_cols = [c for c in df.columns if c not in ["Patient_ID", target_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"[INFO] Selected {len(feature_cols)} clinical features: {feature_cols}")
    
    # Handle missing values using Median Imputation
    # Median is preferred over Mean for medical vitals because it is robust to extreme outliers
    print("[INFO] Imputing missing values using feature medians...")
    imputer = SimpleImputer(strategy="median")
    
    X_imputed = imputer.fit_transform(X)
    X_clean = pd.DataFrame(X_imputed, columns=feature_cols)
    
    # Verify no missing values remain
    remaining_nans = X_clean.isnull().sum().sum()
    print(f"[INFO] Preprocessing complete. Remaining NaNs in feature matrix: {remaining_nans}")
    
    return X_clean, y, feature_cols


if __name__ == "__main__":
    from data_loader import load_sepsis_data
    
    # Standalone preprocessing test
    raw_df = load_sepsis_data()
    perform_eda(raw_df)
    X, y, features = preprocess_data(raw_df)
    print(f"[INFO] Cleaned feature matrix shape: {X.shape}")
