"""
Baseline Ensemble Model Training Module.

This module trains ONE baseline tree-based ensemble model:
- Random Forest Classifier (sklearn.ensemble.RandomForestClassifier)

Random Forest is chosen because tree ensemble methods handle tabular medical vitals,
non-linear clinical feature interactions, and feature scaling natively.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_baseline_model(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data and train a baseline Random Forest ensemble model.
    
    Parameters:
        X (pd.DataFrame): Preprocessed feature matrix.
        y (pd.Series): Binary SepsisLabel target series.
        test_size (float): Proportion of dataset to hold out for evaluation (default: 0.2).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (model, X_test, y_test)
    """
    print("\n[INFO] Train-Test Split:")
    print(f"  - Holding out {int(test_size*100)}% of data for test evaluation...")
    
    # Stratified split ensures equal proportion of positive sepsis cases in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples:  {X_test.shape[0]}")
    
    print("\n[INFO] Initializing baseline Random Forest Classifier...")
    # Baseline configuration with 100 decision trees
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced"  # Handles class imbalance in sepsis data
    )
    
    print("[INFO] Training Random Forest model on clinical vitals...")
    model.fit(X_train, y_train)
    print("[INFO] Random Forest training completed successfully.")
    
    return model, X_test, y_test


if __name__ == "__main__":
    from data_loader import load_sepsis_data
    from preprocess import preprocess_data
    
    # Standalone training test
    raw_df = load_sepsis_data()
    X, y, _ = preprocess_data(raw_df)
    model, X_test, y_test = train_baseline_model(X, y)
