import os
import sys
import torch
from torch import Tensor
from torch.distributions import Normal
from typing import Dict, Tuple

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from managers.utils.logger import Log
    from rl.common.policy import Policy
    from rl.models.actor import PPOActor
    from rl.models.critic import PPOCritic
    from rl.models.encoders import ObservationEncoder
except ImportError as e:
    Log.error(__file__, e)


class PPOPolicy(Policy):
    """
    @class PPOPolicy
    @brief Proximal Policy Optimisation (PPO) policy with encoder, actor, and critic.

    This class implements a continuous-action PPO policy. Observations are first
    encoded via a multi-modal encoder into a latent vector, which is then passed
    through the actor to generate a Gaussian action distribution and the critic
    to estimate state values.
    """

    def __init__(self, obs_config: dict, action_dim: int, device: str = "cpu"):
        """
        @brief Initialise the PPOPolicy.

        @param obs_config dict Configuration parameters for the ObservationEncoder
        @param action_dim int Dimensionality of the action space
        @param device str Torch device to use ('cpu' or 'cuda')
        """
        super().__init__()

        self.device = torch.device(device)

        # Multi-modal observation encoder
        self.encoder = ObservationEncoder(**obs_config).to(self.device)

        # Actor and critic networks
        latent_dim = obs_config["latent_dim"]
        hidden_dim = obs_config["hidden_dim"]
        n_reward_components = obs_config["n_reward_components"]
        continuous = True # PPO with continuous actions
        self.actor = PPOActor(latent_dim=latent_dim, action_dim=action_dim, continuous=continuous).to(self.device)
        self.critic = PPOCritic(latent_dim=latent_dim, hidden_dim=hidden_dim, n_reward_components=n_reward_components).to(self.device)

    def _encode(self, obs: Dict[str, Tensor]) -> Tensor:
        """
        @brief Encode observations into latent representation.

        Moves all observation tensors to the correct device before encoding.

        @param obs dict Observations (e.g., camera, LiDAR, ego_state, risk_features)
        @return torch.Tensor Latent vector representation of the input
        """
        obs = {k: v.to(self.device) for k, v in obs.items()}
        return self.encoder(**obs)

    def _get_distribution(self, latent: Tensor) -> Normal:
        """
        @brief Create a Normal distribution from the actor's outputs.

        @param latent torch.Tensor Latent representation from encoder
        @return Normal Torch distribution representing action probabilities
        """
        mean, std = self.actor(latent)
        return Normal(mean, std)

    def act(self, obs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        @brief Sample an action for environment interaction.

        Encodes the observation, computes the action distribution, samples an
        action, and evaluates its log probability and state value.

        @param obs dict Current environment observation
        @return action torch.Tensor Sampled action
        @return log_prob torch.Tensor Log probability of the action
        @return value torch.Tensor Estimated state value
        """
        latent = self._encode(obs)
        dist = self._get_distribution(latent)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(latent)

        return action, log_prob, value

    def evaluate(self, obs: Dict[str, Tensor], action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        @brief Evaluate a given action for PPO updates.

        Computes the log probability, entropy, and state value for a provided
        action, given the current observation.

        @param obs dict Observations corresponding to the state
        @param action torch.Tensor Action to evaluate
        @return log_prob torch.Tensor Log probability of the action
        @return entropy torch.Tensor Entropy of the action distribution
        @return value torch.Tensor Estimated state value
        """
        latent = self._encode(obs)
        dist = self._get_distribution(latent)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent)

        return log_prob, entropy, value