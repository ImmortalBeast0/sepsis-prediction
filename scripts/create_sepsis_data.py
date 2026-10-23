"""
Script to generate a realistic clinical ICU Sepsis dataset based on PhysioNet / Kaggle specifications.
Includes patient vitals (HR, Temp, SBP, Resp, WBC), demographic data (Age), and SepsisLabel.
Realistic missing values (NaN) and clinical distribution overlap (e.g., ICU stress response)
are modeled to simulate empirical critical-care telemetry data.
"""
import os
import random
import csv
import numpy as np

def generate_dataset(output_path="data/sepsis_dataset.csv", num_samples=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fieldnames = ["Patient_ID", "Age", "HR", "Temp", "SBP", "Resp", "WBC", "SepsisLabel"]
    records = []
    
    for i in range(1, num_samples + 1):
        patient_id = f"P{i:04d}"
        
        # Sepsis outcome: ~12% positive class ratio (typical ICU sepsis prevalence)
        is_sepsis = 1 if random.random() < 0.12 else 0
        
        if is_sepsis == 1:
            # Septic physiological profile (SIRS criteria / septic shock indicators)
            age = int(np.clip(np.random.normal(64, 14), 18, 95))
            hr = round(float(np.clip(np.random.normal(98, 16), 40, 200)), 1)
            temp = round(float(np.clip(np.random.normal(38.1, 0.9), 34, 41)), 1)
            sbp = round(float(np.clip(np.random.normal(104, 16), 60, 200)), 1)
            resp = round(float(np.clip(np.random.normal(23, 4.2), 10, 45)), 1)
            wbc = round(float(np.clip(np.random.normal(14.5, 4.5), 1, 35)), 1)
        else:
            # Non-septic physiological profile (with ~20% ICU post-op / inflammatory stress overlap)
            stress = random.random() < 0.20
            age = int(np.clip(np.random.normal(58, 15), 18, 95))
            hr = round(float(np.clip(np.random.normal(88 if stress else 76, 12), 40, 200)), 1)
            temp = round(float(np.clip(np.random.normal(37.4 if stress else 36.8, 0.6), 34, 41)), 1)
            sbp = round(float(np.clip(np.random.normal(118, 14), 60, 200)), 1)
            resp = round(float(np.clip(np.random.normal(19 if stress else 16, 3), 10, 45)), 1)
            wbc = round(float(np.clip(np.random.normal(10.2 if stress else 7.5, 2.8), 1, 35)), 1)
            
        # Simulate real-world ICU data missingness
        hr_val = "" if random.random() < 0.04 else str(hr)
        temp_val = "" if random.random() < 0.08 else str(temp)
        sbp_val = "" if random.random() < 0.05 else str(sbp)
        resp_val = "" if random.random() < 0.06 else str(resp)
        wbc_val = "" if random.random() < 0.12 else str(wbc)
        
        records.append({
            "Patient_ID": patient_id,
            "Age": age,
            "HR": hr_val,
            "Temp": temp_val,
            "SBP": sbp_val,
            "Resp": resp_val,
            "WBC": wbc_val,
            "SepsisLabel": is_sepsis
        })
        
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"[INFO] Generated realistic ICU dataset with {num_samples} records saved to '{output_path}'.")

if __name__ == "__main__":
    generate_dataset()
