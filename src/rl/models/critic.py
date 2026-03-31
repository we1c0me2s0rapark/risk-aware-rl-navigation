import torch
import torch.nn as nn

class PPOCritic(nn.Module):
    """
    @class PPOCritic
    @brief Multi-head critic for PPO supporting multiple reward components.

    @details
    Implements a shared trunk with separate linear heads for each objective.
    Each head estimates the value function for a specific reward component:
        [0] navigation
        [1] safety
        [2] risk

    This allows the PPO trainer to compute per-objective critic losses
    and combine them with learnable or fixed weights.
    """

    def __init__(self, latent_dim: int, n_objectives: int = 3):
        """
        @brief Initialise the multi-head critic network.

        @param latent_dim int Dimension of the latent feature vector input.
        @param n_objectives int Number of reward objectives to predict.
        """
        super().__init__()

        # ---- Shared trunk ----
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ---- Separate head per objective ----
        # Each head predicts the value for one component of the decomposed reward
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_objectives)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass through the critic.

        @param x torch.Tensor Input latent features of shape [B, latent_dim].
        @return torch.Tensor Predicted values of shape [B, n_objectives].
        """
        feat = self.trunk(x)
        # Concatenate outputs of each head along the last dimension
        return torch.cat([head(feat) for head in self.heads], dim=-1)