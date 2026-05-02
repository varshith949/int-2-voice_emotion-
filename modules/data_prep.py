import os
import pandas as pd

# LEAD'S ADVICE: Don't wait for a perfect dataset. Start with what you have.
def generate_metadata_csv(chunks_dir="chunks/", output_file="metadata.csv"):
    if not os.path.exists(chunks_dir):
        print(f"Error: {chunks_dir} not found. Run chunker.py first!")
        return

    # List all generated .wav chunks
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".wav")]
    
    # Create a simple mapping: [File Path, Emotion Label]
    data = {
        "file_path": [os.path.join(chunks_dir, f) for f in chunk_files],
        "label": ["unlabeled"] * len(chunk_files)  # Placeholder for your intuition
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"--- DAILY RUN COMPLETE ---")
    print(f"Generated {output_file} with {len(chunk_files)} entries.")
    print(f"Next: Open the CSV and replace 'unlabeled' with: happy, sad, angry, or neutral.")

if __name__ == "__main__":
    generate_metadata_csv()