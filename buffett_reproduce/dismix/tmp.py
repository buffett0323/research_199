import pretty_midi
import os
import librosa
from tqdm import tqdm 


# Define the directory containing the MIDI files
directory = '/home/buffett/NAS_189/cocochorales_output/main_dataset/train/'
# midi_directory = '/home/buffett/NAS_189/cocochorales_output/main_dataset/train/string_track000001/stems_midi'

# Function to extract pitch information from MIDI files
def extract_pitch_labels(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    pitch_labels = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_labels.append(note.pitch)
    print(pitch_labels)
    return pitch_labels



cnt = 0
low, high = 100, 0
for f in tqdm(os.listdir(directory)):
    cnt += 1
    if cnt % 100 == 0: 
        print(low, high)
    midi_directory = os.path.join(directory, f, "stems_midi")
    
    pitches = []
    for filename in os.listdir(midi_directory):
        if filename.endswith('.mid'):
            midi_file_path = os.path.join(midi_directory, filename)
            pitches += extract_pitch_labels(midi_file_path)
            

            # Load the audio file
            wav_file_path = midi_file_path.replace("stems_midi", "stems_audio").replace(".mid", ".wav")
            y, sr = librosa.load(wav_file_path, sr=None)  # sr=None preserves the original sample rate

            # Calculate the duration in seconds
            duration = librosa.get_duration(y=y, sr=sr)

            print(f"Duration of the audio file: {duration} seconds")

        

    # Display unique pitch labels for violin
    sorted_pitch = sorted(set(pitches))
    # print("Unique pitch labels for violin:", sorted_pitch)
    
    if sorted_pitch[0] < low:
        low = sorted_pitch[0]
    if sorted_pitch[-1] > high:
        high = sorted_pitch[-1]


print(low, high)