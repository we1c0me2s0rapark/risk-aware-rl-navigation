import torch
import torch.nn as nn

class PPOCritic(nn.Module):
    """
    @brief Proximal Policy Optimisation (PPO) critic network.
    
    Estimates the state-value function V(s), providing a scalar value
    for each input state latent representation.
    """

    def __init__(self, latent_dim):
        """
        @brief Initialise the PPO critic network.
        
        @param latent_dim Dimensionality of the input latent vector, 
                          typically produced by an encoder or feature extractor.
        """
        super().__init__()
        # Multi-layer perceptron (MLP) mapping latent features to scalar value
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # outputs a single scalar value
        )

    def forward(self, x):
        """
        @brief Forward pass through the critic network.
        
        @param x Input tensor of shape (batch_size, latent_dim)
        @return Tensor of shape (batch_size,) representing V(s) for each state.
        """
        value = self.mlp(x)
        return value.squeeze(-1) # remove last dimension to get shape [batch]