import opensmile
import pandas as pd
import os

def extract_features(chunk_path):
    # Fixed the attribute name from eGeMAPS_v02 to eGeMAPSv02
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02, # <--- Corrected line
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    # Extract features
    y_features = smile.process_file(chunk_path)
    
    # Print intermediate outputs as per lead's instructions
    print(f"Extracted features for {os.path.basename(chunk_path)} - Shape: {y_features.shape}")
    
    return y_features

if __name__ == "__main__":
    # Test on one chunk - ensure you run chunker.py first!
    test_chunk = "chunks/chunk_0.wav"
    if os.path.exists(test_chunk):
        feats = extract_features(test_chunk)
        print("\nSUCCESS: Feature extraction module is running.")
    else:
        print("Error: No chunks found in 'chunks/' folder. Run chunker.py first.")