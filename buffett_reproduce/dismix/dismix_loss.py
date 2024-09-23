import torch
import torch.nn as nn
import torch.nn.functional as F

class ELBOLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_kl=1.0):
        super(ELBOLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.reconstruction_loss = nn.MSELoss()  # Can use other loss functions depending on the task

    def forward(self, reconstructed_mixture, original_mixture, timbre_mean, timbre_logvar, pitch_logits, pitch_labels):
        """
        Calculate the ELBO Loss for the DisMix model.

        Parameters:
        - reconstructed_mixture: The mixture reconstructed by the decoder.
        - original_mixture: The original input mixture.
        - timbre_mean: The mean of the timbre latent distribution.
        - timbre_logvar: The log-variance of the timbre latent distribution.
        - pitch_logits: The logits output from the pitch encoder.
        - pitch_labels: Ground-truth pitch labels.

        Returns:
        - elbo_loss: The total ELBO loss.
        """
        # 1. Reconstruction Loss
        recon_loss = self.reconstruction_loss(reconstructed_mixture, original_mixture)
        
        # 2. KL Divergence Loss for Timbre Latent
        # KL(q(z) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = -0.5 * torch.sum(1 + timbre_logvar - timbre_mean.pow(2) - timbre_logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_divergence)  # Mean over batch
        
        # 3. Pitch Log-Likelihood Loss
        # We use BCE with logits loss as the pitch loss
        pitch_loss = nn.BCEWithLogitsLoss()(pitch_logits, pitch_labels)
        
        # Combine all losses to calculate ELBO
        elbo_loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss - pitch_loss

        return elbo_loss


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_off_diag=0.0051):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_off_diag = lambda_off_diag

    def forward(self, query_embedding, timbre_latent):
        """
        Calculate the Barlow Twins loss (L_BT).

        Parameters:
        - query_embedding: The embedding of the query (e_q).
        - timbre_latent: The timbre latent representation (τ).

        Returns:
        - loss: The Barlow Twins loss value.
        """
        # Normalize the embeddings along the batch dimension
        query_norm = (query_embedding - query_embedding.mean(0)) / query_embedding.std(0)
        timbre_norm = (timbre_latent - timbre_latent.mean(0)) / timbre_latent.std(0)

        # Compute the cross-correlation matrix
        batch_size = query_embedding.size(0)
        cross_corr = torch.mm(query_norm.T, timbre_norm) / batch_size

        # Calculate the Barlow Twins loss
        # 1. Loss for diagonal elements (close to 1)
        on_diag = torch.diagonal(cross_corr).add_(-1).pow(2).sum()
        
        # 2. Loss for off-diagonal elements (close to 0)
        off_diag = off_diagonal_elements(cross_corr).pow(2).sum()

        # Final loss combining both parts
        loss = on_diag + self.lambda_off_diag * off_diag

        return loss

def off_diagonal_elements(matrix):
    """
    Extracts the off-diagonal elements from a square matrix.
    
    Parameters:
    - matrix: A square matrix.

    Returns:
    - A tensor of off-diagonal elements.
    """
    n, _ = matrix.shape
    assert n == matrix.shape[1], "Input must be a square matrix"
    return matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()


if __name__ == "__main__":
    # Example usage
    batch_size, num_pitch_classes, num_frames = 4, 52, 10
    input_dim = 128

    # Dummy data for illustration
    original_mixture = torch.randn(batch_size, num_frames, input_dim)
    reconstructed_mixture = torch.randn(batch_size, num_frames, input_dim)
    timbre_mean = torch.randn(batch_size, input_dim)
    timbre_logvar = torch.randn(batch_size, input_dim)
    pitch_logits = torch.randn(batch_size, num_pitch_classes)
    pitch_labels = torch.randint(0, 2, (batch_size, num_pitch_classes)).float()

    # Instantiate and compute ELBO loss
    elbo_loss_fn = ELBOLoss(lambda_recon=1.0, lambda_kl=0.1)  # Adjust coefficients as needed
    print(reconstructed_mixture.shape, original_mixture.shape, timbre_mean.shape, timbre_logvar.shape, pitch_logits.shape, pitch_labels.shape)
    elbo_loss = elbo_loss_fn(reconstructed_mixture, original_mixture, timbre_mean, timbre_logvar, pitch_logits, pitch_labels)

    print("ELBO Loss:", elbo_loss.item())
    
    # Example usage
    batch_size, dim = 4, 64
    query_embedding = torch.randn(batch_size, dim)  # Example query embeddings
    timbre_latent = torch.randn(batch_size, dim)    # Example timbre latents

    bt_loss_fn = BarlowTwinsLoss(lambda_off_diag=0.0051)  # Adjust lambda as needed
    bt_loss = bt_loss_fn(query_embedding, timbre_latent)
    print("Barlow Twins Loss:", bt_loss.item())


