import torch
import torch.nn as nn
import torch.nn.functional as F

class TimbreEncoder(nn.Module):
    def __init__(
        self,          
        input_dim=128, 
        hidden_dim=768, 
        output_dim=64
    ):
        super(TimbreEncoder, self).__init__()
        
        # Define Conv1D layers as specified in the paper
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=1)
        
        # Gaussian parameterization layers
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_logvar = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        
        # Average pooling along the temporal dimension
        x = torch.mean(x, dim=-1)  # Shape: [batch_size, output_dim]

        # Compute mean and log-variance for Gaussian distribution
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Reparameterization trick to sample from Gaussian
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std  # Latent vector for timbre
        
        return z, mu, logvar  # Return timbre latent, mean, and log-variance





# Pitch Encoder Implementation from Table 5
class PitchEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        hidden_dim=256, 
        pitch_classes=52, 
        output_dim=64
    ):
        super(PitchEncoder, self).__init__()

        # Transcriber: Linear layers for pitch classification
        self.transcriber = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pitch_classes),  # Output logits for pitch classification
            nn.Sigmoid()  # Apply Sigmoid for binary classification
        )

        # Stochastic Binarization (SB) Layer: Converts pitch logits to a binary representation, using rounding to binarize
        self.sb_layer = nn.Sequential(
            nn.Sigmoid(),  # Apply sigmoid to get probabilities (same as previous layer)
        )

        # Projection Layer: Project the binarized pitch representation to the latent space
        self.fc_proj = nn.Sequential(
            nn.Linear(pitch_classes, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # Pass through the transcriber to get pitch logits
        pitch_logits = self.transcriber(x)  # Shape: [batch_size, pitch_classes]
        
        # Binarize the pitch logits
        pitch_probs = torch.sigmoid(pitch_logits)
        pitch_binarized = torch.round(pitch_probs)  # Convert to binary (0/1)
        
        # Project the binarized pitch to the latent space
        pitch_latent = self.fc_proj(pitch_binarized)  # Shape: [batch_size, output_dim]

        return pitch_latent, pitch_logits




class Decoder(nn.Module):
    def __init__(
        self, 
        input_dim=64, 
        hidden_dim=128, 
        output_dim=128, 
        num_frames=10
    ):
        super(Decoder, self).__init__()
        self.num_frames = num_frames
        
        # FiLM layer to modulate pitch latent with timbre latent
        self.film_layer = FiLM(input_dim, input_dim)
        
        # Bi-directional GRU layers to transform the latent representation
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Linear layer to transform GRU output to match the original spectrogram dimension
        self.linear = nn.Linear(2 * hidden_dim, output_dim)  # 2 for bidirectional GRU
    
    def forward(self, pitch_latent, timbre_latent):
        # Combine pitch and timbre latent using FiLM
        s = self.film_layer(pitch_latent, timbre_latent)
        
        # Broadcast along the time axis to match the number of frames
        s = s.unsqueeze(1).repeat(1, self.num_frames, 1)  # (batch_size, num_frames, input_dim)
        
        # Pass through the GRU layers
        gru_output, _ = self.gru(s)
        
        # Pass through the linear layer to reconstruct the spectrogram
        reconstructed_spectrogram = self.linear(gru_output)  # (batch_size, num_frames, output_dim)
        
        return reconstructed_spectrogram


# Main Model
class DisMixModel(nn.Module):
    def __init__(self):
        super(DisMixModel, self).__init__()
        self.pitch_encoder = PitchEncoder()
        self.timbre_encoder = TimbreEncoder()
        self.decoder = Decoder()
        
    def forward(self, mixture, query):
        pitch_latents = self.pitch_encoder(mixture)
        timbre_mean, timbre_var = self.timbre_encoder(query)
        s = torch.cat([pitch_latents, timbre_mean], dim=1)  # Example combination
        """ FiLM """
        reconstructed_mixture = self.decoder(s)
        return reconstructed_mixture, pitch_latents, timbre_mean, timbre_var




if __name__ == "__main__":
    # Example usage
    batch_size = 8
    input_dim = 128
    time_steps = 100
    dummy_input = torch.randn(batch_size, input_dim, time_steps)

    # Create the encoder
    timbre_encoder = TimbreEncoder(input_dim=input_dim)

    # Forward pass
    timbre_embedding, mu, logvar = timbre_encoder(dummy_input)
    print(f"Timbre Embedding Shape: {timbre_embedding.shape}")  # Expected shape: [batch_size, output_dim]


    # # Create the pitch encoder
    # pitch_encoder = PitchEncoder(input_dim=input_dim)

    # # Forward pass
    # pitch_latent, pitch_logits = pitch_encoder(dummy_input)
    # print(f"Pitch Latent Shape: {pitch_latent.shape}")  # Expected shape: [batch_size, output_dim]
    # print(f"Pitch Logits Shape: {pitch_logits.shape}")  # Expected shape: [batch_size, pitch_classes]
