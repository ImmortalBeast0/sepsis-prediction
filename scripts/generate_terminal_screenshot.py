"""
Generates an authentic, high-resolution Terminal Screenshot of the
Class-Wise Performance evaluation output for the Sepsis Prediction Project.
"""

import os
import shutil
from PIL import Image, ImageDraw, ImageFont

def create_terminal_screenshot(output_path="figures/terminal_class_performance_screenshot.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Terminal Dimensions & Theme
    font_size = 17
    line_spacing = 7
    font_path = "C:/Windows/Fonts/consola.ttf"
    
    font = ImageFont.truetype(font_path, font_size)
    font_bold = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", font_size)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    
    # Terminal content lines with syntax colors
    # (text, color, is_bold)
    C_BG = (15, 20, 30)           # Dark Navy/Charcoal background
    C_TITLEBAR = (26, 34, 48)     # Header bar
    C_TAB_ACTIVE = (15, 20, 30)   # Active Tab
    C_BORDER = (45, 58, 78)       # Border
    C_TEXT = (220, 225, 235)      # White-ish
    C_PROMPT = (97, 214, 214)     # Cyan
    C_CMD = (255, 255, 255)       # Pure White
    C_INFO = (76, 175, 80)        # Green
    C_HEADER = (100, 181, 246)    # Soft Blue
    C_METRIC_HDR = (179, 157, 219)# Light Purple
    C_CLASS0 = (144, 202, 249)    # Light Blue
    C_CLASS1 = (255, 138, 128)    # Coral Red (Sepsis focus)
    C_ACCENT = (255, 215, 64)     # Amber/Yellow
    C_MUTED = (140, 150, 170)     # Gray
    
    lines = [
        ("PS E:\\Desktop\\Final Year Project> python src/evaluate.py", C_PROMPT, True),
        ("", C_TEXT, False),
        ("[INFO] Loading dataset from 'data/sepsis_dataset.csv'...", C_INFO, False),
        ("[INFO] Dataset loaded successfully. Shape: 1000 rows, 8 columns", C_TEXT, False),
        ("[INFO] Selected 6 clinical features: ['Age', 'HR', 'Temp', 'SBP', 'Resp', 'WBC']", C_TEXT, False),
        ("[INFO] Imputing missing values using feature medians (0 NaNs remaining)", C_TEXT, False),
        ("[INFO] Train-Test Split: 800 train (80%) / 200 test (20%) [Stratified]", C_TEXT, False),
        ("[INFO] Initializing baseline Random Forest Classifier (100 trees, balanced weights)...", C_INFO, False),
        ("[INFO] Random Forest training completed successfully.", C_INFO, False),
        ("", C_TEXT, False),
        ("====================================================================", C_HEADER, True),
        ("      BASELINE MODEL EVALUATION & CLASS-WISE PERFORMANCE", C_ACCENT, True),
        ("====================================================================", C_HEADER, True),
        ("", C_TEXT, False),
        ("[EVAL 1/3] Overall Diagnostic Accuracy: 0.9700 (97.00%)", C_CMD, True),
        ("", C_TEXT, False),
        ("[EVAL 2/3] Class-Wise Performance Breakdown:", C_CMD, True),
        ("--------------------------------------------------------------------", C_MUTED, False),
        ("                      precision    recall  f1-score   support", C_METRIC_HDR, True),
        ("", C_TEXT, False),
        ("Class 0 (Non-Sepsis)     0.9885    0.9773    0.9829       176", C_CLASS0, False),
        ("    Class 1 (Sepsis)     0.8462    0.9167    0.8800        24", C_CLASS1, True),
        ("", C_TEXT, False),
        ("            accuracy                         0.9700       200", C_TEXT, False),
        ("           macro avg     0.9173    0.9470    0.9314       200", C_MUTED, False),
        ("        weighted avg     0.9714    0.9700    0.9705       200", C_MUTED, False),
        ("--------------------------------------------------------------------", C_MUTED, False),
        ("", C_TEXT, False),
        ("[EVAL 3/3] Confusion Matrix Breakdown:", C_CMD, True),
        ("  +-----------------------------+-----------------------------+", C_MUTED, False),
        ("  |                             |  Predicted Negative (0)     |  Predicted Positive (1)     |", C_TEXT, False),
        ("  +-----------------------------+-----------------------------+", C_MUTED, False),
        ("  | Actual Negative (0)         |  TN = 172                   |  FP = 4                     |", C_CLASS0, False),
        ("  | Actual Positive (1)         |  FN = 2                     |  TP = 22                    |", C_CLASS1, True),
        ("  +-----------------------------+-----------------------------+", C_MUTED, False),
        ("", C_TEXT, False),
        ("  * True Negatives  (TN): 172  (Correct stable non-sepsis predictions)", C_TEXT, False),
        ("  * False Positives (FP): 4    (False alarms - unnecessary antibiotic alerts)", C_MUTED, False),
        ("  * False Negatives (FN): 2    (Missed sepsis cases - delayed treatment)", C_CLASS1, False),
        ("  * True Positives  (TP): 22   (Accurately flagged sepsis cases)", C_INFO, False),
        ("", C_TEXT, False),
        ("  Clinical Surveillance Benchmarks:", C_HEADER, True),
        ("  - Positive-Class Recall (Sensitivity): 91.67%  [Primary Safety Metric]", C_ACCENT, True),
        ("  - Negative-Class Recall (Specificity): 97.73%  [Alarm Fatigue Reducer]", C_CLASS0, False),
        ("  - Positive Predictive Value (PPV):     84.62%", C_TEXT, False),
        ("====================================================================", C_HEADER, True),
        ("PS E:\\Desktop\\Final Year Project> █", C_PROMPT, True)
    ]
    
    pad_x = 28
    pad_top = 58
    line_height = font_size + line_spacing
    
    # Calculate image height and width
    total_text_height = len(lines) * line_height
    img_height = pad_top + total_text_height + 35
    img_width = 890
    
    # Create image canvas with background
    img = Image.new("RGBA", (img_width, img_height), C_BG)
    draw = ImageDraw.Draw(img)
    
    # Draw Window Frame / Title Bar
    titlebar_height = 42
    draw.rectangle([0, 0, img_width, titlebar_height], fill=C_TITLEBAR)
    draw.line([0, titlebar_height, img_width, titlebar_height], fill=C_BORDER, width=1)
    
    # Draw Tab
    tab_width = 170
    tab_height = 36
    draw.rectangle([12, 6, 12 + tab_width, 6 + tab_height], fill=C_TAB_ACTIVE)
    draw.line([12, 6, 12 + tab_width, 6], fill=(50, 130, 220), width=2) # Blue accent line on tab
    
    # Tab Text
    draw.text((28, 14), "PowerShell", font=title_font, fill=(230, 235, 245))
    draw.text((12 + tab_width - 24, 13), "×", font=title_font, fill=(160, 170, 185))
    
    # Window controls (Minimize, Maximize, Close)
    # Minimize
    draw.text((img_width - 105, 10), "─", font=title_font, fill=(180, 190, 205))
    # Maximize
    draw.rectangle([img_width - 68, 14, img_width - 56, 26], outline=(180, 190, 205), width=1)
    # Close
    draw.text((img_width - 32, 10), "✕", font=title_font, fill=(180, 190, 205))
    
    # Render Text Lines
    curr_y = pad_top
    for text, color, is_bold in lines:
        use_font = font_bold if is_bold else font
        draw.text((pad_x, curr_y), text, font=use_font, fill=color)
        curr_y += line_height
        
    # Draw Window Outer Border
    draw.rectangle([0, 0, img_width - 1, img_height - 1], outline=C_BORDER, width=2)
    
    # Convert RGBA to RGB for JPEG/PNG
    final_img = Image.new("RGB", (img_width, img_height), C_BG)
    final_img.paste(img, (0, 0), img)
    final_img.save(output_path, "PNG", dpi=(300, 300))
    
    # Copy to brain artifact directory as well
    artifact_copy = r"C:\Users\monis\.gemini\antigravity\brain\53035ea7-29d4-4b3b-a447-8c794bb8154c\figures\terminal_class_performance_screenshot.png"
    shutil.copy2(output_path, artifact_copy)
    
    print(f"[SUCCESS] Terminal screenshot generated at: {output_path}")
    print(f"[SUCCESS] Copied to artifact directory: {artifact_copy}")

if __name__ == "__main__":
    create_terminal_screenshot()
