"""
Basic Model Evaluation Module.

Prints essential evaluation metrics for the baseline model:
1. Overall Accuracy Score
2. Confusion Matrix breakdown (TN, FP, FN, TP)

Per Phase 1 scope requirements, no complex metrics (ROC-AUC, PR-AUC) or explainability 
tools (SHAP, DiCE) are included in this basic evaluation step.
"""

from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance on test dataset.
    
    Parameters:
        model: Trained classifier.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True ground truth test labels.
        
    Returns:
        dict: Basic evaluation results containing accuracy and confusion matrix.
    """
    print("\n" + "="*60)
    print("      BASELINE MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Generate predictions on test data
    y_pred = model.predict(X_test)
    
    # 1. Compute Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL 1/2] Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # 2. Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n[EVAL 2/2] Confusion Matrix Breakdown:")
    print("  +-----------------------+-----------------------+")
    print("  |                       |  Predicted Negative   |  Predicted Positive   |")
    print("  +-----------------------+-----------------------+")
    print(f"  | Actual Negative (0)   |  TN = {tn:<14} |  FP = {fp:<14} |")
    print(f"  | Actual Positive (1)   |  FN = {fn:<14} |  TP = {tp:<14} |")
    print("  +-----------------------+-----------------------+")
    print(f"\n  * True Negatives  (TN): {tn} (Correct non-sepsis predictions)")
    print(f"  * False Positives (FP): {fp} (False alarms)")
    print(f"  * False Negatives (FN): {fn} (Missed sepsis cases)")
    print(f"  * True Positives  (TP): {tp} (Correctly detected sepsis cases)")
    print("="*60 + "\n")
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    from data_loader import load_sepsis_data
    from preprocess import preprocess_data
    from train import train_baseline_model
    
    # Standalone evaluation test
    raw_df = load_sepsis_data()
    X, y, _ = preprocess_data(raw_df)
    model, X_test, y_test = train_baseline_model(X, y)
    evaluate_model(model, X_test, y_test)
