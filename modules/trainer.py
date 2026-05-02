import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.dataset import VoiceEmotionDataset

# 1. Simple Neural Network for VAD Regression
class EmotionRegressor(nn.Module):
    def __init__(self, input_size=88):
        super(EmotionRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 3 Outputs: Valence, Arousal, Dominance
        )

    def forward(self, x):
        return self.fc(x)

def train_model():
    # Setup
    dataset = VoiceEmotionDataset("metadata.csv")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = EmotionRegressor(input_size=88)
    
    # CRITICAL CHANGE: Use MSELoss for Regression (VAD)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- STARTING WEEK 3 VAD TRAINING (77 SAMPLES) ---")
    for epoch in range(50):
        total_loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "emotion_model.pth")
    print("SUCCESS: VAD Regression Model Saved.")

if __name__ == "__main__":
    train_model()