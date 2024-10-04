import torch
import torch.nn as nn
import torch.nn.functional as F

class ELBOLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(ELBOLoss, self).__init__()
        self.beta = beta  # Weight for the KL-divergence term

    def forward(self, x, x_hat, timbre_mean, timbre_logvar):
        """
        Compute the ELBO loss, including reconstruction loss and KL-divergence.
        
        Args:
            x (torch.Tensor): Original input mixture (shape: [batch_size, num_features]).
            x_hat (torch.Tensor): Reconstructed mixture (shape: [batch_size, num_features]).
            timbre_mean (torch.Tensor): Mean of the timbre latent distribution.
            timbre_logvar (torch.Tensor): Log variance of the timbre latent distribution.
        
        Returns:
            elbo_loss (torch.Tensor): The total ELBO loss (reconstruction + KL-divergence).
        """
        # 1. Reconstruction loss (MSE or L1)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

        # 2. KL-divergence for timbre latent space
        kl_timbre = -0.5 * torch.sum(1 + timbre_logvar - timbre_mean.pow(2) - timbre_logvar.exp())

        # Final ELBO loss
        elbo_loss = reconstruction_loss + self.beta * kl_timbre

        return elbo_loss


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=0.005):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z1, z2):
        """
        Computes the Barlow Twins loss between two latent representations z1 (pitch) and z2 (timbre).
        
        Args:
            z1 (torch.Tensor): Latent representation from the pitch encoder (shape: [batch_size, latent_dim]).
            z2 (torch.Tensor): Latent representation from the timbre encoder (shape: [batch_size, latent_dim]).
        
        Returns:
            loss (torch.Tensor): Barlow Twins loss.
        """
        # Normalize the representations along the batch dimension, adding epsilon to prevent division by zero
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)

        # Compute the cross-correlation matrix
        batch_size, latent_dim = z1.shape
        c = torch.matmul(z1_norm.T, z2_norm) / batch_size

        # Compute the Barlow Twins loss
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()  # Loss on the diagonal
        off_diag = self.off_diagonal(c).pow(2).sum()  # Loss on the off-diagonal

        loss = on_diag + self.lambda_param * off_diag
        return loss

    def off_diagonal(self, x):
        """
        Returns the off-diagonal elements of a square matrix.
        
        Args:
            x (torch.Tensor): Square matrix of shape [n, n].
        
        Returns:
            torch.Tensor: Flattened tensor of off-diagonal elements.
        """
        n, m = x.shape
        assert n == m, "Input matrix must be square"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



# Example usage
if __name__ == '__main__':
    batch_size = 8
    latent_dim = 64

    # Example pitch and timbre latent representations
    pitch_latent = torch.randn(batch_size, latent_dim)  # ν(i)
    timbre_latent = torch.randn(batch_size, latent_dim)  # τ(i)

    # Initialize Barlow Twins loss
    loss_fn = BarlowTwinsLoss(lambda_param=0.005)

    # Compute the Barlow Twins loss
    loss = loss_fn(pitch_latent, timbre_latent)
    print(f"Barlow Twins Loss: {loss.item()}")

