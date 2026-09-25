"""
Script to generate a realistic clinical ICU Sepsis dataset based on PhysioNet / Kaggle specifications.
Includes patient vitals (HR, Temp, SBP, Resp, WBC), demographic data (Age), and SepsisLabel.
Realistic missing values (NaN) are added to simulate real-world EHR / ICU telemetry data.
"""
import os
import random
import csv

def generate_dataset(output_path="data/sepsis_dataset.csv", num_samples=1000, seed=42):
    random.seed(seed)
    
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
            age = max(18, min(95, int(random.gauss(65, 14))))
            hr = round(random.gauss(108, 15), 1)         # Tachycardia
            temp = round(random.gauss(38.6, 1.1), 1)     # Fever / hypothermia
            sbp = round(random.gauss(94, 14), 1)         # Hypotension
            resp = round(random.gauss(26, 4.5), 1)       # Tachypnea
            wbc = round(random.gauss(16.8, 4.8), 1)      # Leukocytosis
        else:
            # Non-septic physiological profile
            age = max(18, min(95, int(random.gauss(56, 16))))
            hr = round(random.gauss(76, 10), 1)          # Normal HR
            temp = round(random.gauss(36.8, 0.4), 1)     # Normal body temp
            sbp = round(random.gauss(122, 11), 1)        # Normal blood pressure
            resp = round(random.gauss(16, 2.5), 1)       # Normal respiration
            wbc = round(random.gauss(7.2, 2.0), 1)       # Normal WBC count
            
        # Simulate real-world ICU data missingness
        hr_val = "" if random.random() < 0.04 else str(max(30.0, min(220.0, hr)))
        temp_val = "" if random.random() < 0.09 else str(max(33.0, min(42.0, temp)))
        sbp_val = "" if random.random() < 0.05 else str(max(50.0, min(230.0, sbp)))
        resp_val = "" if random.random() < 0.07 else str(max(6.0, min(50.0, resp)))
        wbc_val = "" if random.random() < 0.14 else str(max(0.5, min(45.0, wbc)))
        
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
        
    print(f"[INFO] Generated dataset with {num_samples} records saved to '{output_path}'.")

if __name__ == "__main__":
    generate_dataset()
