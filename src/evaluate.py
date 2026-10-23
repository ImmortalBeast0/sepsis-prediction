"""
Basic Model Evaluation Module.

Prints essential evaluation metrics for the baseline model:
1. Overall Accuracy Score
2. Class-Wise Performance Report (Precision, Recall, F1-Score for Class 0 & 1)
3. Confusion Matrix Breakdown (TN, FP, FN, TP)
4. Clinical Utility Benchmarks (Sensitivity, Specificity, Alert Precision)
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance on test dataset.
    
    Parameters:
        model: Trained classifier.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True ground truth test labels.
        
    Returns:
        dict: Basic evaluation results containing accuracy, confusion matrix, and report.
    """
    print("\n" + "="*68)
    print("      BASELINE MODEL EVALUATION & CLASS-WISE PERFORMANCE")
    print("="*68)
    
    # Generate predictions on test data
    y_pred = model.predict(X_test)
    
    # 1. Compute Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL 1/3] Overall Diagnostic Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # 2. Detailed Class-Wise Performance Breakdown
    print("\n[EVAL 2/3] Class-Wise Performance Breakdown:")
    print("--------------------------------------------------------------------")
    report = classification_report(
        y_test, y_pred,
        target_names=["Class 0 (Non-Sepsis)", "Class 1 (Sepsis)"],
        digits=4
    )
    print(report.strip())
    print("--------------------------------------------------------------------")
    
    # 3. Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n[EVAL 3/3] Confusion Matrix Breakdown:")
    print("  +-----------------------------+-----------------------------+")
    print("  |                             |  Predicted Negative (0)     |  Predicted Positive (1)     |")
    print("  +-----------------------------+-----------------------------+")
    print(f"  | Actual Negative (0)         |  TN = {tn:<21} |  FP = {fp:<21} |")
    print(f"  | Actual Positive (1)         |  FN = {fn:<21} |  TP = {tp:<21} |")
    print("  +-----------------------------+-----------------------------+")
    
    # Clinical metrics
    sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    
    print(f"\n  * True Negatives  (TN): {tn:<4} (Correct stable non-sepsis predictions)")
    print(f"  * False Positives (FP): {fp:<4} (False alarms - unnecessary antibiotic alerts)")
    print(f"  * False Negatives (FN): {fn:<4} (Missed sepsis cases - delayed treatment)")
    print(f"  * True Positives  (TP): {tp:<4} (Accurately flagged sepsis cases)")
    print("\n  Clinical Surveillance Benchmarks:")
    print(f"  - Positive-Class Recall (Sensitivity): {sensitivity:.2f}%  [Primary Safety Metric]")
    print(f"  - Negative-Class Recall (Specificity): {specificity:.2f}%  [Alarm Fatigue Reducer]")
    print(f"  - Positive Predictive Value (PPV):     {precision:.2f}%")
    print("="*68 + "\n")
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
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
