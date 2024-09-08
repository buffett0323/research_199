import librosa
import numpy as np
import os
from tqdm import tqdm

input_data_path = "/mnt/gestalt/home/ddmanddman/moisesdb/moisesdb_v0.1"
output_npy_path = "/mnt/gestalt/home/ddmanddman/moisesdb/npy2"
os.makedirs(output_npy_path, exist_ok=True)

# Function to preprocess audio files into .npy format
def preprocess_audio_to_npy(input_path, output_path):
    for song_id in tqdm(os.listdir(input_path)):
        if song_id.endswith(".csv"): continue
        song_dir = os.path.join(input_path, song_id)
        output_song_dir = os.path.join(output_path, song_id)
        os.makedirs(output_song_dir, exist_ok=True)

        for stem_folder in os.listdir(song_dir):
            if stem_folder == "data.json": continue

            for stem_file in os.listdir(os.path.join(song_dir, stem_folder)):
                if stem_file.endswith(".wav"):
                    stem_name = os.path.splitext(stem_file)[0]
                    audio_path = os.path.join(song_dir, stem_folder, stem_file)
                    audio, sr = librosa.load(audio_path, sr=44100, mono=False)

                    # Save as .npy
                    np.save(os.path.join(output_song_dir, f"{stem_name}.npy"), audio)

                    print(f"Processed {stem_file} -> {stem_name}.npy")

# Preprocess the audio files
preprocess_audio_to_npy(input_data_path, output_npy_path)