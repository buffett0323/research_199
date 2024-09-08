import librosa
import numpy as np
import os
from scipy.signal import find_peaks
from tqdm import tqdm

""" Process Query """
# Define paths
input_data_path = "/mnt/gestalt/home/ddmanddman/moisesdb/moisesdb_v0.1"
output_query_path = "/mnt/gestalt/home/ddmanddman/moisesdb/npyq"
os.makedirs(output_query_path, exist_ok=True)

# Function to extract the 10-second chunk with the strongest onset
def extract_strongest_onset(audio, sr, window_size=10, hop_size=512):
    # Process each channel separately
    strongest_onset_chunks = []
    
    for channel in range(audio.shape[0]):
        onset_env = librosa.onset.onset_strength(y=audio[channel], sr=sr)
        window_frames = int(window_size * sr / hop_size)
        avg_onset_strength = np.convolve(onset_env, np.ones(window_frames), mode='valid')
        max_index = np.argmax(avg_onset_strength)
        start_sample = max_index * hop_size
        end_sample = start_sample + (window_size * sr)

        # Ensure end_sample doesn't exceed the audio length
        if end_sample > len(audio[channel]):
            end_sample = len(audio[channel])
            start_sample = end_sample - (window_size * sr)

        strongest_onset_chunks.append(audio[channel, start_sample:end_sample])
    
    # Stack the channels back together
    return np.stack(strongest_onset_chunks, axis=0)


# Iterate over each song and stem
for song_id in tqdm(os.listdir(input_data_path)):
    if song_id.endswith(".csv"): continue
    
    song_dir = os.path.join(input_data_path, song_id)
    output_song_dir = os.path.join(output_query_path, song_id)
    os.makedirs(output_song_dir, exist_ok=True)

    for stem_folder in os.listdir(song_dir):
        if stem_folder == "data.json": continue

        for stem_file in os.listdir(os.path.join(song_dir, stem_folder)):
            if stem_file.endswith(".wav"):
                stem_name = os.path.splitext(stem_file)[0]
                audio_path = os.path.join(song_dir, stem_folder, stem_file)
                audio, sr = librosa.load(audio_path, sr=44100, mono=False)

                # Extract the strongest onset
                query_audio = extract_strongest_onset(audio, sr)

                # Save as .npy
                np.save(os.path.join(output_song_dir, f"{stem_name}.query.npy"), query_audio)

                print(f"Processed {stem_file} -> {stem_name}.query.npy")
