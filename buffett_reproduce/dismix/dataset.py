import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

class MusicDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, n_mels=128):
        """
        Initialize the dataset.

        Parameters:
        - data_dir: Directory containing the dataset with mixture and query audio files.
        - sample_rate: Sample rate for audio processing.
        - n_mels: Number of mel frequency bins for spectrogram transformation.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.file_list = self._load_file_list(data_dir)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_mels=self.n_mels
        )
    
    def _load_file_list(self, data_dir):
        """
        Load the list of files in the dataset directory.
        
        Parameters:
        - data_dir: The directory containing the dataset.
        
        Returns:
        - List of file paths.
        """
        file_list = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):  # Assuming audio files are in .wav format
                    file_list.append(os.path.join(root, file))
        return file_list

    def _load_audio(self, file_path):
        """
        Load an audio file and convert it to a mel spectrogram.
        
        Parameters:
        - file_path: Path to the audio file.
        
        Returns:
        - Mel spectrogram tensor.
        """
        waveform, _ = torchaudio.load(file_path)
        mel_spectrogram = self.mel_transform(waveform)
        return mel_spectrogram

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        - idx: Index of the sample.
        
        Returns:
        - A dictionary containing the mixture, query, pitch, and timbre.
        """
        # For simplicity, assume that files are named in a structured way to pair mixture and query.
        file_path = self.file_list[idx]
        
        # Load mixture and query. (Modify path structure as needed.)
        mixture_path = file_path.replace('query', 'mixture')
        query_path = file_path.replace('mixture', 'query')
        
        # Load mel spectrograms
        mixture = self._load_audio(mixture_path)
        query = self._load_audio(query_path)
        
        # Placeholder for pitch and timbre labels. Replace with actual label extraction.
        pitch_label = torch.randint(0, 2, (52,)).float()  # Example pitch label
        timbre_label = torch.randn(64)  # Example timbre label
        
        return {
            'mixture': mixture,       # Mixture mel spectrogram
            'query': query,           # Query mel spectrogram
            'pitch_label': pitch_label,  # Ground-truth pitch label
            'timbre_label': timbre_label  # Ground-truth timbre label
        }


if __name__ == "__main__":
    # Usage Example
    data_dir = "/path/to/dataset"  # Replace with the actual path to your dataset
    dataset = MusicDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate over data_loader
    for batch in data_loader:
        mixture = batch['mixture']
        query = batch['query']
        pitch_label = batch['pitch_label']
        timbre_label = batch['timbre_label']
        # Use these in your training loop
        print("Mixture Shape:", mixture.shape)
        print("Query Shape:", query.shape)
