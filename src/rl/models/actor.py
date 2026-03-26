import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActor(nn.Module):
    """
    @brief Proximal Policy Optimisation (PPO) actor network.
    
    This network approximates the policy for reinforcement learning. Depending
    on the action space type, it produces either action probabilities (for 
    discrete actions) or a Gaussian distribution's mean and standard deviation
    (for continuous actions).
    """

    def __init__(self, latent_dim, action_dim, continuous=False):
        """
        @brief Initialise the PPO actor network.
        
        @param latent_dim Dimensionality of the input latent representation, 
                          typically produced by an encoder or feature extractor.
        @param action_dim Dimensionality of the action space.
        @param continuous Flag indicating whether the action space is continuous 
                          (True) or discrete (False).
        """
        super().__init__()
        self.continuous = continuous

        # Shared multilayer perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        if continuous:
            # Continuous action output: mean and learnable log standard deviation
            self.mean = nn.Linear(128, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim)) # learnable log std
        else:
            # Discrete action output: logits for categorical distribution
            self.logits = nn.Linear(128, action_dim)

    def forward(self, x):
        """
        @brief Forward pass through the actor network.
        
        @param x Input tensor of shape (batch_size, latent_dim).
        @return If continuous: tuple (mean, std) of Gaussian distribution.
                If discrete: logits for each action.
        """
        feat = self.mlp(x)

        if self.continuous:
            mean = self.mean(feat)
            std = torch.exp(self.log_std) # ensure positive standard deviation
            return mean, std
        else:
            logits = self.logits(feat)
            return logits