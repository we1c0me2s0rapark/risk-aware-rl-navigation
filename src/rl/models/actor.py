import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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

            # Initialise biases to favour acceleration over braking
            # so the agent moves immediately rather than hesitating
            nn.init.constant_(self.mean.bias, 0.0) # steer: centred
            self.mean.bias.data[1] = 1.0 # throttle → ~0.73 after rescaling
            self.mean.bias.data[2] = -2.0 # brake → ~0.12 after rescaling
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

class SACActor(nn.Module):
    """
    @class SACActor
    @brief Squashed Gaussian actor for SAC.

    @details
    Outputs a tanh-squashed Gaussian distribution over actions. The tanh
    squashing enforces action bounds and requires a log-probability correction
    to account for the change of variables:

        log π(a|s) = log N(u|s) - Σ log(1 - tanh²(u_i))

    where u is the pre-squash sample and a = tanh(u).

    This correction is critical for SAC's entropy maximisation objective -
    without it, the entropy estimate is biased and training is unstable.
    """

    # Numerical stability clamps
    __LOG_STD_MIN = -5          # lower clamp on log std to prevent vanishing std
    __LOG_STD_MAX = 2           # upper clamp on log std to prevent exploding std
    __LOG_PROB_EPSILON = 1e-6   # prevent log(0) in tanh correction

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        @brief Initialise the SAC actor.

        @param latent_dim int Dimension of the latent state encoding.
        @param action_dim int Dimension of the continuous action space.
        @param hidden_dim int Width of hidden layers.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor):
        """
        @brief Forward pass - compute mean and log_std of the Gaussian.

        @param x torch.Tensor Latent features [B, latent_dim].
        @return tuple[torch.Tensor, torch.Tensor] (mean, log_std), each [B, action_dim].
        """
        feat = self.mlp(x)
        mean = self.mean_layer(feat)
        log_std = self.log_std_layer(feat)
        log_std = torch.clamp(log_std, self.__LOG_STD_MIN, self.__LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        """
        @brief Sample a tanh-squashed action and compute its log probability.

        @details
        Uses the reparameterisation trick for differentiable sampling:
            u ~ N(mean, std)
            a = tanh(u)

        Log probability is corrected for the tanh transformation:
            log π(a|s) = log N(u|s) - Σ log(1 - tanh²(u_i) + ε)

        @param x torch.Tensor Latent features [B, latent_dim].
        @return tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - action: tanh-squashed action [B, action_dim]
            - log_prob: log probability of action [B, 1]
            - mean: deterministic action (for evaluation) [B, action_dim]
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()

        # Reparameterisation trick
        dist = Normal(mean, std)
        u = dist.rsample()

        # Tanh squashing
        action = torch.tanh(u)

        # Log prob with tanh correction
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + self.__LOG_PROB_EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # [B, 1]

        return action, log_prob, torch.tanh(mean)