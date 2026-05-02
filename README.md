# Voice Emotion Detection - VAD Regression Pipeline

This project implements a dimensional emotion recognition system using the **Valence-Arousal-Dominance (VAD)** model. Unlike traditional categorical classification, this pipeline maps vocal features to a 3D emotional space using a Deep Neural Network (DNN) regressor.

## 🚀 Project Overview
*   **Target Task:** Multi-output regression to predict continuous VAD coordinates.
*   **Dataset:** Indian Emotional Speech Corpora (IESC) consisting of 78 audio chunks across 5 emotion categories (Angry, Fear, Happy, Neutral, Sad).
*   **Framework:** PyTorch & openSMILE.

## 🏗️ Architecture
The system utilizes a **Multi-Layer Perceptron (MLP)** with the following specifications:
*   **Input Layer:** 88 acoustic features extracted via the **eGeMAPS** standard.
*   **Hidden Layers:** Fully connected layers with ReLU activation and Dropout (0.2) for regularization.
*   **Output Layer:** 3 neurons representing Valence, Arousal, and Dominance.
*   **Objective Function:** Mean Squared Error (MSE) Loss.

## 📊 Evaluation Results (Week 3 Baseline)
The model was evaluated by mapping predicted 3D coordinates back to the nearest emotion label using **Euclidean Distance**.

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 27% |
| **Highest F1-Score** | Happy (0.42) |
| **Lowest F1-Score** | Sad (0.00) |

### Key Findings
*   The pipeline successfully establishes a baseline for Indian vocal tonalities.
*   **Arousal tracking** is functional, effectively distinguishing high-energy emotions like "Happy" and "Angry."
*   **Calibration Need:** The 0.00 F1-score for "Sad" indicates that predicted coordinates for sadness are currently drifting into Neutral or Fear zones, marking a priority for future calibration.

## 📁 Repository Structure
*   `modules/extractor.py`: Feature extraction using openSMILE eGeMAPS.
*   `modules/dataset.py`: Custom PyTorch dataset mapping labels to 3D VAD coordinates.
*   `modules/trainer.py`: MLP Regressor training logic.
*   `modules/inference.py`: Prediction logic with Euclidean Distance emotion mapping.
*   `metadata.csv`: Labeled dataset tracking 78 Indian audio chunks.

## 🛠️ Setup & Usage
1.  **Activate Environment:** `.\venv\Scripts\Activate.ps1`
2.  **Training:** `python modules/trainer.py`
3.  **Inference:** `python modules/inference.py`