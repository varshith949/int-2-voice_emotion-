import librosa
import soundfile as sf
import os

# LEAD'S ADVICE: Process the entire directory to build your 200+ sample dataset.
AUDIO_FOLDER = "audio_sample.wav" # This is your folder name
OUTPUT_DIR = "chunks/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def chunk_audio_files(folder_path, chunk_duration=1.0):
    """
    Loops through the folder and splits each .wav file into 1.0s chunks.
    """
    # Get all .wav files from the folder
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"No .wav files found in {folder_path}!")
        return []

    all_metadata = []

    for filename in audio_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\nProcessing File: {filename}")
        
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)
        samples_per_chunk = int(chunk_duration * sr)
        num_chunks = int(len(audio) / samples_per_chunk)

        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk_data = audio[start:end]
            
            # UNIQUE NAME: Includes original filename to avoid overwriting
            chunk_name = f"{filename.split('.')[0]}_c{i}.wav"
            chunk_path = os.path.join(OUTPUT_DIR, chunk_name)
            sf.write(chunk_path, chunk_data, sr)
            
            all_metadata.append({
                "chunk_id": f"{filename}_{i}",
                "path": chunk_path
            })
            
    return all_metadata

if __name__ == "__main__":
    if os.path.isdir(AUDIO_FOLDER):
        metadata = chunk_audio_files(AUDIO_FOLDER)
        print(f"\n--- DAILY RUN COMPLETE ---")
        print(f"SUCCESS: Produced {len(metadata)} chunks from the folder.")
    else:
        print(f"ERROR: '{AUDIO_FOLDER}' is not a directory!")