import torch
import json
import os
import sys
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.extractor import extract_features
from modules.trainer import EmotionRegressor

def run_combined_inference(model_path="emotion_model.pth"):
    model = EmotionRegressor(input_size=88)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # VAD Reference Map (matching your dataset.py)
    vad_map = {
        "angry":   [-0.5,  0.8,  0.7],
        "fear":    [-0.6,  0.6, -0.5],
        "happy":   [ 0.8,  0.6,  0.3],
        "neutral": [ 0.0, -0.1,  0.0],
        "sad":     [-0.8, -0.6, -0.4]
    }

    chunk_files = [f for f in os.listdir("chunks/") if f.endswith(".wav")]
    results = []

    with torch.no_grad():
        for i, filename in enumerate(sorted(chunk_files)):
            path = os.path.join("chunks/", filename)
            feats = extract_features(path)
            feats_tensor = torch.tensor(feats.values.reshape(1, -1), dtype=torch.float32)
            
            # 1. Get VAD coordinates from model
            vad_output = model(feats_tensor).numpy()[0]
            v, a, d = vad_output[0], vad_output[1], vad_output[2]

            # 2. Calculate Distance to each emotion
            distances = {}
            for emotion, target_vad in vad_map.items():
                dist = np.linalg.norm(vad_output - np.array(target_vad))
                distances[emotion] = dist

            # 3. Best Match is the smallest distance
            detected_emotion = min(distances, key=distances.get)

            results.append({
                "chunk_id": i,
                "filename": filename,
                "detected_emotion": detected_emotion,
                "valence": round(float(v), 3),
                "arousal": round(float(a), 3),
                "dominance": round(float(d), 3)
            })

    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"SUCCESS: Combined VAD and Emotion output saved.")

if __name__ == "__main__":
    run_combined_inference()