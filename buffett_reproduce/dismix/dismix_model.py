import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dismix_loss import ELBOLoss

# Reparameterization function to obtain the latent from mean and log variance
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


class StochasticBinarizationLayer(nn.Module):
    def __init__(self):
        super(StochasticBinarizationLayer, self).__init__()
    
    def forward(self, logits):
        """
        Forward pass of the stochastic binarization layer.
        
        Parameters:
        - logits: The raw outputs from the pitch encoder (before sigmoid activation).
        
        Returns:
        - binarized_output: Binary tensor after stochastic binarization.
        """
        
        probabilities = torch.sigmoid(logits) # Apply sigmoid to get probabilities
        h = torch.rand_like(probabilities) # Sample a uniform random threshold for binarization
        binarized_output = (probabilities > h).float() # Binarize based on the threshold h
        
        return binarized_output # Return the binarized output with a straight-through estimator



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
            nn.Linear(hidden_dim, pitch_classes)  # Output logits for pitch classification
            # No Sigmoid here, SB layer will handle it
        )

        # Stochastic Binarization (SB) Layer: Converts pitch logits to a binary representation
        self.sb_layer = StochasticBinarizationLayer()

        # Projection Layer: Project the binarized pitch representation to the latent space
        self.fc_proj = nn.Sequential(
            nn.Linear(pitch_classes, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        # Pass through the transcriber to get pitch logits
        pitch_logits = self.transcriber(x)  # Shape: [batch_size, pitch_classes]
        
        # Apply Stochastic Binarization Layer to logits
        pitch_binarized = self.sb_layer(pitch_logits)  # Binarize using stochastic binarization
        
        # Project the binarized pitch to the latent space
        pitch_latent = self.fc_proj(pitch_binarized)  # Shape: [batch_size, output_dim]

        return pitch_latent, pitch_logits



# Combining FiLM layer
class FiLM(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(latent_dim, input_dim)
        self.shift = nn.Linear(latent_dim, input_dim)
    
    def forward(self, pitch_latent, timbre_latent):
        scale = self.scale(timbre_latent)
        shift = self.shift(timbre_latent)
        return scale * pitch_latent + shift
    
    


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
    def __init__(
        self, 
        input_dim=128, 
        latent_dim=64, 
        hidden_dim=128, 
        num_frames=10
    ):
        super(DisMixModel, self).__init__()
        self.pitch_encoder = PitchEncoder(input_dim, latent_dim)
        self.timbre_encoder = TimbreEncoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_frames=num_frames)
        
    def forward(self, mixture, query):
        # Encode pitch and timbre latents
        pitch_latent, pitch_logits = self.pitch_encoder(mixture)
        _, timbre_mean, timbre_logvar = self.timbre_encoder(query)
        
        # Reparameterize to get timbre latent
        timbre_latent = reparameterize(timbre_mean, timbre_logvar)
        
        # Decode to reconstruct the mixture
        reconstructed_spectrogram = self.decoder(pitch_latent, timbre_latent)
        return reconstructed_spectrogram, pitch_latent, pitch_logits, timbre_mean, timbre_logvar



if __name__ == "__main__":
    # Example usage
    batch_size = 8
    time_steps = 100
    input_dim = 128
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    mixture = torch.randn(batch_size, time_steps, input_dim).to(device)
    query = torch.randn(batch_size, time_steps, input_dim).to(device)
    
    # Models and other settings
    model = DisMixModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    
    # Loss function
    elbo_loss_fn = ELBOLoss(lambda_recon=1.0, lambda_kl=0.1) # For ELBO
    bce_loss_fn = nn.BCEWithLogitsLoss()  # For pitch supervision
    bt_loss_fn = BarlowTwinsLoss(lambda_off_diag=0.0051) # Barlow Twins

    rec_mixture, pitch_latents, pitch_logits, timbre_mean, timbre_logvar = model(mixture, query)
    print(rec_mixture.shape, mixture.shape, pitch_latents.shape, timbre_mean.shape, timbre_logvar.shape)
    
    # Loss
    # Pitch labels represents Ground-truth pitch labels
    elbo_loss = elbo_loss_fn(
        rec_mixture, mixture, 
        timbre_mean, timbre_logvar, 
        pitch_logits, pitch_labels
    )
    bce_loss = bce_loss_fn(pitch_latents, ground_truth_pitch)
    bt_loss = bt_loss_fn(timbre_mean, query)
    
    loss = elbo_loss + bce_loss - bt_loss
    loss.backward()
    optimizer.step()
    
    print(loss.item())
    # print(f"Epoch {epoch} Loss: {loss.item()}")
