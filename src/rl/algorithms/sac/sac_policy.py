import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from rl.models.encoders import ObservationEncoder
    from rl.models.critic import SACCritic
    from rl.models.actor import SACActor
except ImportError as e:
    Log.error(__file__, e)

class SACPolicy(nn.Module):
    """
    @class SACPolicy
    @brief Full SAC policy combining encoder, actor, and double Q-critic.

    @details
    Encodes multimodal observations into a latent vector, then passes it
    through the SAC actor for action selection and the double Q-critic
    for value estimation.

    The encoder is shared between actor and critic to reduce computation,
    but their gradients are kept separate via detach where appropriate.
    """

    def __init__(self, obs_config: dict, action_dim: int, device: torch.device):
        """
        @brief Initialise the SAC policy.

        @param obs_config dict Observation configuration with keys:
            - camera_shape, lidar_shape, ego_state_dim, latent_dim, hidden_dim, n_reward_components
        @param action_dim int Dimension of the continuous action space.
        @param device torch.device Device to run inference on.
        """
        super().__init__()

        self.device = device
        self.action_dim = action_dim

        latent_dim = obs_config.get('latent_dim', 256)
        hidden_dim = obs_config.get('hidden_dim', 128)
        n_reward_components = obs_config.get('n_reward_components', 3)

        # ---- Shared encoder ----
        self.encoder = ObservationEncoder(
            camera_shape=obs_config['camera_shape'],
            lidar_shape=obs_config['lidar_shape'],
            ego_state_dim=obs_config['ego_state_dim'],
            risk_feature_dim=obs_config['risk_feature_dim'],
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_reward_components=n_reward_components,
            use_lidar=True,
            use_risk=True,
        )

        # ---- Actor ----
        self.actor = SACActor(latent_dim, action_dim)

        # ---- Double Q-critic ----
        self.critic = SACCritic(latent_dim, action_dim)

        # ---- Target critic (soft update target) ----
        self.critic_target = SACCritic(latent_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target network is never directly trained; only soft-updated
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.to(device)

    def _encode(self, obs: dict) -> torch.Tensor:
        """
        @brief Encode a multimodal observation dict into a latent vector.

        @param obs dict Batched observation tensors.
        @return torch.Tensor Latent encoding [B, latent_dim].
        """
        obs = {k: v.to(self.device) for k, v in obs.items()}
        return self.encoder(**obs)

    def act(self, obs: dict, deterministic: bool = False):
        """
        @brief Select an action given an observation.

        @details
        During training, samples stochastically from the squashed Gaussian.
        During evaluation, uses the deterministic mean action.

        @param obs dict Batched observation dict with tensors [1, ...].
        @param deterministic bool If True, use mean action (no sampling).
        @return tuple[torch.Tensor, torch.Tensor]:
            - action [1, action_dim]
            - log_prob [1, 1] (None if deterministic)
        """
        with torch.no_grad():
            latent = self._encode(obs)
            action, log_prob, mean = self.actor.sample(latent)

        if deterministic:
            return mean, None

        return action, log_prob

    def evaluate(self, obs: dict):
        """
        @brief Sample action and compute log prob with gradients (for actor update).

        @param obs dict Batched observation dict [B, ...].
        @return tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - action [B, action_dim]
            - log_prob [B, 1]
            - latent [B, latent_dim] (detached, for critic input)
        """
        latent = self._encode(obs)
        action, log_prob, _ = self.actor.sample(latent)
        return action, log_prob, latent.detach()

    def get_q_values(self, obs: dict, action: torch.Tensor):
        """
        @brief Compute Q1 and Q2 values for a given obs-action pair.

        @param obs dict Batched observation dict [B, ...].
        @param action torch.Tensor Action tensor [B, action_dim].
        @return tuple[torch.Tensor, torch.Tensor] Q1, Q2 each [B, 1].
        """
        latent = self._encode(obs)
        return self.critic(latent, action)

    def get_target_q_values(self, obs: dict, action: torch.Tensor):
        """
        @brief Compute target Q values using the target critic network.

        @details
        Target network is used for stable Bellman backup computation.
        It is updated via soft (Polyak) averaging, not gradient descent.

        @param obs dict Batched next observation dict [B, ...].
        @param action torch.Tensor Next action tensor [B, action_dim].
        @return tuple[torch.Tensor, torch.Tensor] Target Q1, Q2 each [B, 1].
        """
        with torch.no_grad():
            latent = self._encode(obs)
            return self.critic_target(latent, action)

    def soft_update_target(self, tau: float = 0.005):
        """
        @brief Soft update the target critic via Polyak averaging.

        @details
        θ_target ← τ * θ_online + (1 - τ) * θ_target

        Small τ (e.g. 0.005) keeps the target stable, which is critical
        for convergence of the Bellman update.

        @param tau float Polyak averaging coefficient.
        """
        for target, online in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(tau * online.data + (1 - tau) * target.data)