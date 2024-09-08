import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(
        self,
        cond_embedding_dim: int,
        channels: int,
        additive: bool = True,
        multiplicative: bool = False,
        depth: int = 1,
        activation: str = "ELU",
    ):
        super().__init__()

        self.additive = additive
        self.multiplicative = multiplicative
        self.depth = depth

        # Define activation function
        Activation = getattr(nn, activation)

        if self.multiplicative:
            if depth == 1:
                self.gamma = nn.Linear(cond_embedding_dim, channels)
            else:
                layers = [nn.Linear(cond_embedding_dim, channels)]
                for _ in range(depth - 1):
                    layers += [Activation(), nn.Linear(channels, channels)]
                self.gamma = nn.Sequential(*layers)
        else:
            self.gamma = None

        if self.additive:
            if depth == 1:
                self.beta = nn.Linear(cond_embedding_dim, channels)
            else:
                layers = [nn.Linear(cond_embedding_dim, channels)]
                for _ in range(depth - 1):
                    layers += [Activation(), nn.Linear(channels, channels)]
                self.beta = nn.Sequential(*layers)
        else:
            self.beta = None

    def forward(self, x, w):
        # Apply gamma (multiplicative) modulation
        if self.multiplicative:
            gamma = self.gamma(w)
            gamma = gamma[:, :, None, None]  # Reshape for broadcasting over spatial dimensions
            x = gamma * x

        # Apply beta (additive) modulation
        if self.additive:
            beta = self.beta(w)
            beta = beta[:, :, None, None]  # Reshape for broadcasting over spatial dimensions
            x = x + beta

        return x