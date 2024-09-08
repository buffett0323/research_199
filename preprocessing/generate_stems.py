import os
import pandas as pd
from tqdm import tqdm

# Define the paths
npy_path = "/mnt/gestalt/home/ddmanddman/moisesdb/moisesdb_v0.1"  # Directory where your .npy files are stored
output_csv_path = "/mnt/gestalt/home/ddmanddman/moisesdb/moisesdb_v0.1/stems.csv"  # Output path for the stems.csv file

# List of possible stems (adjust based on your dataset)
stems_list = [
    "mixture",
    "drums",
    "bass",
    "vocals",
    "guitar",
    # Add other stems as necessary
]

# Initialize a list to store metadata
metadata = []

# Iterate over each song directory in the dataset
for song_id in tqdm(os.listdir(npy_path)):
    if song_id.endswith(".csv"): continue
    
    song_dir = os.path.join(npy_path, song_id)
    if os.path.isdir(song_dir):
        # Initialize a dictionary for the current song
        song_metadata = {"song_id": song_id}
        
        # Check for each stem if the .npy file exists
        for stem in stems_list:
            npy_file = os.path.join(song_dir, f"{stem}")
            song_metadata[stem] = 1 if os.path.exists(npy_file) else 0
        
        # Append the song's metadata to the list
        metadata.append(song_metadata)


# Convert the metadata list to a DataFrame
metadata_df = pd.DataFrame(metadata)

# Save the DataFrame to a CSV file
metadata_df.to_csv(output_csv_path, index=False)

print(f"Generated stems.csv at {output_csv_path}")
