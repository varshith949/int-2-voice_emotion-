import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def calculate_vad_accuracy(csv_path="metadata.csv", json_path="predictions.json"):
    # 1. Load the ground truth and predictions
    df_true = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    # Create a mapping for quick lookup
    pred_map = {p['filename']: p['detected_emotion'] for p in predictions}
    
    y_true = []
    y_pred = []

    # 2. Match samples by filename
    for _, row in df_true.iterrows():
        fname = os.path.basename(row['file_path'])
        if fname in pred_map:
            y_true.append(row['label'])
            y_pred.append(pred_map[fname])

    # 3. Calculate Scores
    acc = accuracy_score(y_true, y_pred)
    
    print("\n--- WEEK 3 VAD EVALUATION REPORT ---")
    print(f"Total Samples Evaluated: {len(y_true)}")
    print(f"Overall Mapping Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    import os
    calculate_vad_accuracy()