# import pretty_midi
# import os
# import librosa
# from tqdm import tqdm 


# # Define the directory containing the MIDI files
# directory = '/home/buffett/NAS_189/cocochorales_output/main_dataset/train/'
# # midi_directory = '/home/buffett/NAS_189/cocochorales_output/main_dataset/train/string_track000001/stems_midi'

# # Function to extract pitch information from MIDI files
# def extract_pitch_labels(midi_file):
#     midi_data = pretty_midi.PrettyMIDI(midi_file)
#     pitch_labels = []
#     for instrument in midi_data.instruments:
#         if not instrument.is_drum:
#             for note in instrument.notes:
#                 pitch_labels.append(note.pitch)
#     print(pitch_labels)
#     return pitch_labels



# cnt = 0
# low, high = 100, 0
# for f in tqdm(os.listdir(directory)):
#     cnt += 1
#     if cnt % 100 == 0: 
#         print(low, high)
#     midi_directory = os.path.join(directory, f, "stems_midi")
    
#     pitches = []
#     for filename in os.listdir(midi_directory):
#         if filename.endswith('.mid'):
#             midi_file_path = os.path.join(midi_directory, filename)
#             pitches += extract_pitch_labels(midi_file_path)
            

#             # Load the audio file
#             wav_file_path = midi_file_path.replace("stems_midi", "stems_audio").replace(".mid", ".wav")
#             y, sr = librosa.load(wav_file_path, sr=None)  # sr=None preserves the original sample rate

#             # Calculate the duration in seconds
#             duration = librosa.get_duration(y=y, sr=sr)

#             print(f"Duration of the audio file: {duration} seconds")

        

#     # Display unique pitch labels for violin
#     sorted_pitch = sorted(set(pitches))
#     # print("Unique pitch labels for violin:", sorted_pitch)
    
#     if sorted_pitch[0] < low:
#         low = sorted_pitch[0]
#     if sorted_pitch[-1] > high:
#         high = sorted_pitch[-1]


# print(low, high)

import torch
import torch.nn as nn
import torch.nn.functional as F
from dismix_model import MixtureQueryEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dismix_model import DisMixModel
from dataset import CocoChoralesTinyDataset
from dismix_loss import ELBOLoss, BarlowTwinsLoss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 4
lr = 4e-4
epochs = 10

# Usage Example
data_dir = '/home/buffett/NAS_189/cocochorales_output/main_dataset/'
train_dataset = CocoChoralesTinyDataset(data_dir, split='train')
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = CocoChoralesTinyDataset(data_dir, split='valid')
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CocoChoralesTinyDataset(data_dir, split='test')
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Models and other settings
model = DisMixModel(
    input_dim=128, 
    latent_dim=64, 
    hidden_dim=256, 
    gru_hidden_dim=256,
    num_frames=10,
    pitch_classes=52,
    output_dim=128,    
).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Loss function
elbo_loss_fn = ELBOLoss(lambda_recon=1.0, lambda_kl=0.1) # For ELBO
bce_loss_fn = nn.BCEWithLogitsLoss()  # For pitch supervision
bt_loss_fn = BarlowTwinsLoss(lambda_off_diag=0.0051) # Barlow Twins


# Training
for epoch in tqdm(range(epochs)):
    for batch in tqdm(train_data_loader):
        mixture, query, ground_truth_pitch = batch["mixture"].to(device), batch["query"].to(device), batch["pitch_label"].to(device)
        
        rec_mixture, pitch_latent, pitch_logits, timbre_mean, timbre_logvar = model(mixture, query)
        print(rec_mixture.shape, pitch_latent.shape, pitch_logits.shape, timbre_mean.shape, timbre_logvar.shape)
        
        # # Loss
        # elbo_loss = elbo_loss_fn(
        #     rec_mixture, mixture, 
        #     timbre_mean, timbre_logvar, 
        #     pitch_logits, pitch_labels
        # )
        # bce_loss = bce_loss_fn(pitch_latents, ground_truth_pitch)
        # bt_loss = bt_loss_fn(timbre_mean, query)
        break


