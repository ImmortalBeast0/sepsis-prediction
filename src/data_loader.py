import os
import pandas as pd

def load_sepsis_data(filepath="data/sepsis_dataset.csv"):
    """
    Load sepsis clinical dataset from CSV file.
    
    Parameters:
        filepath (str): Path to the sepsis dataset CSV file.
        
    Returns:
        pd.DataFrame: Raw dataset containing patient vitals and SepsisLabel.
    """
    print(f"\n[INFO] Loading dataset from '{filepath}'...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    print(f"[INFO] Dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Feature columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Standalone execution test for data loader
    data = load_sepsis_data()
    print("\n[INFO] First 5 rows of raw dataset:")
    print(data.head())
