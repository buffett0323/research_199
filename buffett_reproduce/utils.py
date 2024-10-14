from omegaconf import OmegaConf
import torch
import torchaudio

def _load_config(config_path: str) -> OmegaConf:
    config = OmegaConf.load(config_path)

    config_dict = {}

    for k, v in config.items():
        if isinstance(v, str) and v.endswith(".yml"):
            config_dict[k] = OmegaConf.load(v)
        else:
            config_dict[k] = v

    config = OmegaConf.merge(config_dict)

    return config


def audio_to_complex_spectrogram(audio, n_fft=2048, hop_length=512):
    """
    Converts the input audio waveform (which could be multi-channel and batched) into a complex-valued spectrogram using STFT.
    
    Parameters:
        audio (torch.Tensor): Input audio tensor of shape (batch_size, num_channels, num_samples).
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for STFT.
    
    Returns:
        torch.Tensor: Complex-valued spectrogram of shape (batch_size, num_channels, frequency_bins, time_frames).
    """
    # Ensure audio is 3D (batch_size, num_channels, num_samples)
    assert len(audio.shape) == 3, "Input audio must be 3D: (batch_size, num_channels, num_samples)"
    
    # Initialize the list to store spectrograms for each batch
    spectrograms = []
    
    # Apply STFT for each batch element
    for i in range(audio.shape[0]):  # Loop over the batch dimension
        batch_spectrogram = torch.stft(audio[i], n_fft=n_fft, hop_length=hop_length, 
                                       return_complex=True, window=torch.hann_window(n_fft).to(audio.device))
        spectrograms.append(batch_spectrogram)

    # Stack the spectrograms back into a tensor of shape (batch_size, num_channels, frequency_bins, time_frames)
    spectrogram = torch.stack(spectrograms, dim=0)
    
    return spectrogram





def get_non_stem_audio(mixture_audio, stem_audio, n_fft=2048, hop_length=512):
    """
    Removes the guitar stem from the mixture audio by subtracting the guitar spectrogram from the mixture spectrogram.
    
    Parameters:
        mixture_audio (torch.Tensor): The input mixture audio waveform (batch_size, num_channels, num_samples).
        stem_audio (torch.Tensor): The target guitar audio waveform (batch_size, num_channels, num_samples).
        n_fft (int): Number of FFT components for STFT.
        hop_length (int): Hop length for STFT.
        
    Returns:
        torch.Tensor: The "non-guitar" audio waveform with the same shape as the input.
    """
    # Ensure mixture and guitar audio have the same length
    min_length = min(mixture_audio.shape[-1], stem_audio.shape[-1])
    mixture_audio = mixture_audio[..., :min_length]
    stem_audio = stem_audio[..., :min_length]

    # Initialize a list to store the "non-guitar" audio for each batch
    non_stem_audios = []
    
    # Process each batch
    for i in range(mixture_audio.shape[0]):  # Iterate over the batch dimension
        batch_non_stem = []
        
        # Process each channel within the batch
        for j in range(mixture_audio.shape[1]):  # Iterate over the channel dimension
            # Perform STFT on mixture and guitar audio for the current batch and channel
            mixture_spec = torch.stft(mixture_audio[i, j], n_fft=n_fft, hop_length=hop_length, 
                                      return_complex=True, window=torch.hann_window(n_fft).to(mixture_audio.device))
            guitar_spec = torch.stft(stem_audio[i, j], n_fft=n_fft, hop_length=hop_length, 
                                     return_complex=True, window=torch.hann_window(n_fft).to(stem_audio.device))
            
            # Subtract guitar spectrogram from mixture spectrogram to get "non-guitar" spectrogram
            non_guitar_spec = mixture_spec - guitar_spec
            
            # Convert the "non-guitar" spectrogram back to time domain using ISTFT
            non_stem_audio = torch.istft(non_guitar_spec, n_fft=n_fft, hop_length=hop_length, 
                                         window=torch.hann_window(n_fft).to(non_guitar_spec.device), 
                                         length=min_length)
            
            # Append the result for the current channel
            batch_non_stem.append(non_stem_audio)
        
        # Stack the channels back and append to the result for the current batch
        non_stem_audios.append(torch.stack(batch_non_stem, dim=0))
    
    # Stack all the batches to form the final output
    non_stem_audio_batch = torch.stack(non_stem_audios, dim=0)
    
    return non_stem_audio_batch
