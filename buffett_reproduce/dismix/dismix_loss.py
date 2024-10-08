import torch
import torch.nn as nn
import torch.nn.functional as F

# class ELBOLoss(nn.Module):
#     def __init__(self, beta=1.0):
#         super(ELBOLoss, self).__init__()
#         self.beta = beta  # Weight for the KL-divergence term

#     def forward(self, x, x_hat, timbre_mean, timbre_logvar):
#         """
#         Compute the ELBO loss, including reconstruction loss and KL-divergence.
        
#         Args:
#             x (torch.Tensor): Original input mixture (shape: [batch_size, num_features]).
#             x_hat (torch.Tensor): Reconstructed mixture (shape: [batch_size, num_features]).
#             timbre_mean (torch.Tensor): Mean of the timbre latent distribution.
#             timbre_logvar (torch.Tensor): Log variance of the timbre latent distribution.
        
#         Returns:
#             elbo_loss (torch.Tensor): The total ELBO loss (reconstruction + KL-divergence).
#         """
#         # 1. Reconstruction loss (MSE or L1)
#         reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

#         # 2. KL-divergence for timbre latent space
#         kl_timbre = -0.5 * torch.sum(1 + timbre_logvar - timbre_mean.pow(2) - timbre_logvar.exp())

#         # Final ELBO loss
#         elbo_loss = reconstruction_loss + self.beta * kl_timbre

#         return elbo_loss


class ELBOLoss(nn.Module):
    def __init__(self):
        super(ELBOLoss, self).__init__()

    def forward(self, x_m, x_m_recon, tau_means, tau_logvars, nu_logits, y_pitch):
        """
        Computes the ELBO loss.

        Args:
            x_m (torch.Tensor): Original mixture data of shape (batch_size, data_dim).
            x_m_recon (torch.Tensor): Reconstructed mixture data of shape (batch_size, data_dim).
            tau_means (list of torch.Tensor): List of timbre latent means for each source.
            tau_logvars (list of torch.Tensor): List of timbre latent log variances for each source.
            nu_logits (list of torch.Tensor): List of pitch latent logits for each source.
            y_pitch (list of torch.Tensor): List of ground truth pitch labels for each source.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        # Number of sources in the mixture
        N_s = len(tau_means)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_m_recon, x_m, reduction='mean')

        # Pitch supervision loss
        pitch_loss = 0.0
        for nu_logit, y in zip(nu_logits, y_pitch):
            # Assume nu_logit is raw logits; apply CrossEntropyLoss
            pitch_loss += F.cross_entropy(nu_logit, y, reduction='mean')

        # KL divergence loss for timbre latents
        kl_loss = 0.0
        for mu, logvar in zip(tau_means, tau_logvars):
            # KL divergence between N(mu, sigma^2) and N(0, 1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss += kl

        # Average KL loss over the batch
        kl_loss = kl_loss / x_m.size(0)

        # Total ELBO loss (negative ELBO)
        loss = recon_loss + pitch_loss + kl_loss

        return loss
    
    

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=0.0051):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    # def forward(self, z1, z2):
    #     """
    #     Computes the Barlow Twins loss between two latent representations z1 (pitch) and z2 (timbre).
        
    #     Args:
    #         z1 (torch.Tensor): Latent representation from the pitch encoder (shape: [batch_size, latent_dim]).
    #         z2 (torch.Tensor): Latent representation from the timbre encoder (shape: [batch_size, latent_dim]).
        
    #     Returns:
    #         loss (torch.Tensor): Barlow Twins loss.
    #     """
    #     # Normalize the representations along the batch dimension, adding epsilon to prevent division by zero
    #     z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
    #     z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)

    #     # Compute the cross-correlation matrix
    #     batch_size, latent_dim = z1.shape
    #     c = torch.mm(z1_norm.T, z2_norm) / batch_size

    #     # Compute the Barlow Twins loss
    #     on_diag = torch.diagonal(c).add_(-1).pow(2).sum()  # Loss on the diagonal
    #     off_diag = self.off_diagonal(c).pow(2).sum()  # Loss on the off-diagonal

    #     loss = on_diag + self.lambda_param * off_diag
    #     return loss
    
    # def off_diagonal(self, x):
    #     """
    #     Returns the off-diagonal elements of a square matrix.
        
    #     Args:
    #         x (torch.Tensor): Square matrix of shape [n, n].
        
    #     Returns:
    #         torch.Tensor: Flattened tensor of off-diagonal elements.
    #     """
    #     n, m = x.shape
    #     assert n == m, "Input matrix must be square"
    #     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    
    def forward(self, e, tau):
        """
        Computes the simplified Barlow Twins loss between query embeddings and timbre latents.

        Args:
            e (torch.Tensor): Query embeddings of shape (batch_size, feature_dim).
            tau (torch.Tensor): Timbre latents of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size, feature_dim = e.shape

        # Normalize embeddings
        e_norm = (e - e.mean(dim=0)) / (e.std(dim=0) + 1e-9)
        tau_norm = (tau - tau.mean(dim=0)) / (tau.std(dim=0) + 1e-9)

        # Compute cross-correlation matrix
        c = torch.mm(e_norm.T, tau_norm) / batch_size

        # Extract diagonal elements
        c_diag = c.diag()

        # Compute loss
        loss = torch.sum((1 - c_diag) ** 2)

        return loss
    

    



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

