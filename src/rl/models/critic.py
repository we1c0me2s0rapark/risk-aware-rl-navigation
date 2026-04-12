import torch
import torch.nn as nn

class PPOCritic(nn.Module):
    """
    @class PPOCritic
    @brief Multi-head critic for PPO supporting multiple reward components.

    @details
    Implements a shared trunk with separate linear heads for each objective.
    Each head estimates the value function for one reward component
    (e.g. navigation, safety, risk when n_reward_components=3).

    This allows the PPO trainer to compute per-objective critic losses
    and combine them with learnable or fixed weights.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_reward_components: int = 3):
        """
        @brief Initialise the multi-head critic network.

        @param latent_dim int Dimension of the latent feature vector input.
        @param hidden_dim int Width of hidden layers in the shared trunk.
        @param n_reward_components int Number of reward objectives to predict.
        """
        super().__init__()

        # ---- Shared trunk ----
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ---- Separate head per objective ----
        # Each head predicts the value for one component of the decomposed reward
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_reward_components)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass through the critic.

        @param x torch.Tensor Input latent features of shape [B, latent_dim].
        @return torch.Tensor Predicted values of shape [B, n_reward_components].
        """
        feat = self.trunk(x)
        # Concatenate outputs of each head along the last dimension
        return torch.cat([head(feat) for head in self.heads], dim=-1)

class SACCritic(nn.Module):
    """
    @class SACCritic
    @brief Double Q-network critic for SAC.
 
    @details
    SAC requires two independent Q-networks Q1(s, a) and Q2(s, a) to reduce
    overestimation bias (double Q-learning). Both networks share the same
    architecture but have separate weights.
 
    The critic takes both the latent state encoding and the action as input,
    concatenates them, and passes through an MLP to predict Q-values.
 
    During training, the minimum of Q1 and Q2 is used as the target to
    prevent overestimation.
    """
 
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        @brief Initialise the double Q-network SAC critic.
 
        @param latent_dim int Dimension of the latent state encoding.
        @param action_dim int Dimension of the action vector.
        @param hidden_dim int Width of hidden layers in each Q-network.
        """
        super().__init__()
 
        input_dim = latent_dim + action_dim
 
        # ---- Q1 network ----
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
 
        # ---- Q2 network ----
        # Identical architecture but separate weights — essential for double Q
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
 
    def forward(self, latent: torch.Tensor, action: torch.Tensor):
        """
        @brief Forward pass through both Q-networks.
 
        @param latent torch.Tensor Latent state encoding of shape [B, latent_dim].
        @param action torch.Tensor Action tensor of shape [B, action_dim].
        @return tuple[torch.Tensor, torch.Tensor] Q1 and Q2 values, each [B, 1].
        """
        x = torch.cat([latent, action], dim=-1)
        return self.q1(x), self.q2(x)
 
    def q1_forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass through Q1 only.
 
        @details
        Used during actor updates where only Q1 is needed to compute
        the policy gradient, avoiding redundant computation through Q2.
 
        @param latent torch.Tensor Latent state encoding of shape [B, latent_dim].
        @param action torch.Tensor Action tensor of shape [B, action_dim].
        @return torch.Tensor Q1 value of shape [B, 1].
        """
        x = torch.cat([latent, action], dim=-1)
        return self.q1(x)