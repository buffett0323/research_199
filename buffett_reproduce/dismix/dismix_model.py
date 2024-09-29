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


class Conv1DEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, norm_layer, activation):
        super(Conv1DEncoder, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.norm = norm_layer(output_channels) if norm_layer else None
        self.activation = activation() if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class MixtureQueryEncoder(nn.Module):
    def __init__(self):
        super(MixtureQueryEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            Conv1DEncoder(128, 768, 3, 1, 0, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(768, 768, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(768, 768, 4, 2, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(768, 768, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(768, 768, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(768, 64, 1, 1, 1, None, None)
        )

    def forward(self, x):
        x = self.encoder_layers(x)
        return torch.mean(x, dim=-1)  # Mean pooling along the temporal dimension


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



# class TimbreEncoder(nn.Module):
#     def __init__(
#         self,          
#         input_dim=128, 
#         hidden_dim=768, 
#         output_dim=64
#     ):
#         super(TimbreEncoder, self).__init__()
        
#         # Define Conv1D layers as specified in the paper
#         self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=0)
#         self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
#         self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=1)
        
#         # Gaussian parameterization layers
#         self.fc_mu = nn.Linear(output_dim, output_dim)
#         self.fc_logvar = nn.Linear(output_dim, output_dim)

#     def forward(self, x):
#         # Apply convolutional layers with ReLU activation
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = self.conv6(x)
        
#         # Average pooling along the temporal dimension
#         x = torch.mean(x, dim=-1)  # Shape: [batch_size, output_dim]

#         # Compute mean and log-variance for Gaussian distribution
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
        
#         # Reparameterization trick to sample from Gaussian
#         std = torch.exp(0.5 * logvar)
#         epsilon = torch.randn_like(std)
#         z = mu + epsilon * std  # Latent vector for timbre
        
#         return z, mu, logvar  # Return timbre latent, mean, and log-variance


class TimbreEncoder(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=256,
        output_dim=64
    ):
        super(TimbreEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            Conv1DEncoder(input_dim, input_dim, 5, 2, 0, lambda x: nn.GroupNorm(1, x), nn.ReLU),
            Conv1DEncoder(input_dim, input_dim, 5, 2, 0, lambda x: nn.GroupNorm(1, x), nn.ReLU),
            Conv1DEncoder(input_dim, input_dim, 5, 2, 0, lambda x: nn.GroupNorm(1, x), nn.ReLU),
            nn.Conv1d(input_dim, hidden_dim, 1, 1, 0)
        )
        self.mean_layer = nn.Linear(hidden_dim, output_dim)  # Mean of the Gaussian
        self.var_layer = nn.Linear(hidden_dim, output_dim)   # Variance of the Gaussian

    def forward(self, em, eq):
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate em and eq
        x = self.encoder_layers(concat_input)
        x = x.squeeze(-1)  # Remove unnecessary dimensions
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        return mean, var


# Pitch Encoder Implementation from Table 5
class PitchEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        hidden_dim=256, 
        pitch_classes=52, # true labels not 0-51
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


    def forward(self, em, eq):
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate em and eq
        pitch_logits = self.transcriber(concat_input)
        y_bin = self.sb_layer(pitch_logits)  # Apply binarisation
        pitch_latent = self.fc_proj(y_bin)
        return pitch_latent, pitch_logits
    
    

class FiLM(nn.Module):
    def __init__(self, pitch_dim, timbre_dim):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(timbre_dim, pitch_dim)
        self.shift = nn.Linear(timbre_dim, pitch_dim)
    
    def forward(self, pitch_latent, timbre_latent):
        scale = self.scale(timbre_latent)
        shift = self.shift(timbre_latent)
        return scale * pitch_latent + shift


class DisMixDecoder(nn.Module):
    def __init__(
        self, 
        pitch_dim=64, 
        timbre_dim=64, 
        gru_hidden_dim=256, 
        output_dim=128, 
        num_layers=2
    ):
        super(DisMixDecoder, self).__init__()
        self.film = FiLM(pitch_dim, timbre_dim)
        self.gru = nn.GRU(input_size=pitch_dim, hidden_size=gru_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(gru_hidden_dim * 2, output_dim)  # Bi-directional GRU output dimension is doubled
        self.output_transform = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, 1, kernel_size=1),  # Convert back to 1D waveform or spectrogram
        )
    
    def forward(self, pitch_latents, timbre_latents):
        # FiLM layer: modulates pitch latents based on timbre latents
        source_latents = self.film(pitch_latents, timbre_latents)
        
        # Temporal Broadcasting: Expand source_latents along time axis if necessary
        source_latents = source_latents.unsqueeze(1).repeat(1, 10, 1)
        
        gru_output, _ = self.gru(source_latents)
        gru_output = self.linear(gru_output)
        
        # Output transform to convert to spectrogram or waveform
        # Transpose to (batch, features, time) for Conv1d layers
        output = self.output_transform(gru_output.transpose(1, 2))
        
        return output # reconstructed spectrogram




# Main Model
class DisMixModel(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        latent_dim=64, 
        hidden_dim=256, 
        gru_hidden_dim=256,
        num_frames=10,
        pitch_classes=52,
        output_dim=128,
    ):
        super(DisMixModel, self).__init__()
        self.mixture_encoder = MixtureQueryEncoder()
        self.query_encoder = MixtureQueryEncoder()
        self.pitch_encoder = PitchEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            pitch_classes=pitch_classes, # true labels not 0-51
            output_dim=latent_dim
        )
        self.timbre_encoder = TimbreEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim
        )
        self.decoder = DisMixDecoder(
            pitch_dim=latent_dim, 
            timbre_dim=latent_dim, 
            gru_hidden_dim=gru_hidden_dim, 
            output_dim=output_dim, 
            num_layers=2
        )
        
    def forward(self, mixture, query):
        
        # Encode mixture and query
        em = self.mixture_encoder(mixture)
        eq = self.query_encoder(query)
        
        # Encode pitch and timbre latents
        pitch_latent, pitch_logits = self.pitch_encoder(em, eq)
        timbre_mean, timbre_logvar = self.timbre_encoder(em, eq)
        
        # pitch_latent, pitch_logits = self.pitch_encoder(mixture)
        # _, timbre_mean, timbre_logvar = self.timbre_encoder(query)
        
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
