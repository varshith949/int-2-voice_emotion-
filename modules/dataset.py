import torch
from torch.utils.data import Dataset
import pandas as pd
from extractor import extract_features

class VoiceEmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.metadata = pd.read_csv(csv_file)
        
        # Week 3 VAD Mapping: Continuous 3D coordinates
        # Format: [Valence, Arousal, Dominance]
        self.vad_map = {
            "angry":   [-0.5,  0.8,  0.7],  # High energy, negative positivity
            "fear":    [-0.6,  0.6, -0.5],  # High energy, low control
            "happy":   [ 0.8,  0.6,  0.3],  # High energy, positive positivity
            "neutral": [ 0.0, -0.1,  0.0],  # Center point
            "sad":     [-0.8, -0.6, -0.4]   # Low energy, negative positivity
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx, 0]
        label_name = self.metadata.iloc[idx, 1]
        
        # 1. Extract features (returns 88 eGeMAPS features)
        features = extract_features(audio_path)
        
        # Ensure tensor is float32 and squeezed to (88,)
        features_tensor = torch.tensor(features.values, dtype=torch.float32).squeeze()
        
        # 2. Get the VAD vector instead of an integer index
        # This converts "happy" -> [0.8, 0.6, 0.3]
        vad_values = self.vad_map[label_name]
        label_tensor = torch.tensor(vad_values, dtype=torch.float32)
        
        return features_tensor, label_tensor