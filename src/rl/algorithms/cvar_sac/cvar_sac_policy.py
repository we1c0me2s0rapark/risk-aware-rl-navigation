import torch
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

    from rl.algorithms.sac.sac_policy import SACPolicy
    from rl.models.critic import DistributionalCritic
except ImportError as e:
    Log.error(__file__, e)

class CVaRSACPolicy(SACPolicy):
    """
    @class CVaRSACPolicy
    @brief CVaR-SAC policy extending SACPolicy with a distributional critic.

    @details
    Inherits the full SACPolicy (encoder, actor, soft target update) and
    replaces the standard double Q-network (SACCritic) with a distributional
    double Q-network (DistributionalCritic) that outputs N quantiles per
    forward pass instead of a scalar Q-value.

    This enables the CVaRSACTrainer to:
        1. Fit the quantile distribution via quantile regression loss
        2. Compute CVaR_α(Z) for the actor objective - penalising worst-case outcomes
        3. Use the distributional Bellman backup for stable target computation

    All encoder and actor components are inherited unchanged from SACPolicy.
    Only the critic and critic_target are swapped for distributional variants.
    """

    def __init__(
        self,
        obs_config: dict,
        action_dim: int,
        device: torch.device,
        n_quantiles: int = 32,
        cvar_alpha: float = 0.1,
    ):
        """
        @brief Initialise CVaRSACPolicy.

        @details
        Calls SACPolicy.__init__() which builds the encoder, actor, and
        standard SACCritic. Then replaces self.critic and self.critic_target
        with DistributionalCritic instances of the same dimensions.

        @param obs_config dict Observation encoder configuration.
        @param action_dim int Dimension of the continuous action space.
        @param device torch.device Device to run inference on.
        @param n_quantiles int Number of quantiles for the distributional critic.
        @param cvar_alpha float CVaR confidence level (default 0.1 = worst 10%).
        """

        # Build base policy (encoder + actor + standard SACCritic)
        super().__init__(obs_config, action_dim, device)

        self.n_quantiles = n_quantiles
        self.cvar_alpha = cvar_alpha

        latent_dim = obs_config.get('latent_dim', 256)
        hidden_dim = obs_config.get('hidden_dim', 256)

        # ---- Replace scalar critic with distributional critic ----
        self.critic = DistributionalCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_quantiles=self.n_quantiles,
            hidden_dim=hidden_dim,
        ).to(device)

        # ---- Replace target critic with distributional variant ----
        self.critic_target = DistributionalCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_quantiles=self.n_quantiles,
            hidden_dim=hidden_dim,
        ).to(device)

        # Sync target weights to online critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target network is never directly trained
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def get_q_values(self, obs: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief Compute Z1 and Z2 quantile distributions for a given obs-action pair.

        @param obs dict Batched observation dict [B, ...].
        @param action torch.Tensor Action tensor [B, action_dim].
        @return tuple[torch.Tensor, torch.Tensor] Z1, Z2 quantiles each [B, N].
        """

        latent = self._encode(obs)
        return self.critic(latent, action)

    def get_target_q_values(self, obs: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief Compute target Z1 and Z2 quantile distributions.

        @details
        Uses the target critic for stable Bellman backup computation.
        No gradients are tracked.

        @param obs dict Batched next observation dict [B, ...].
        @param action torch.Tensor Next action tensor [B, action_dim].
        @return tuple[torch.Tensor, torch.Tensor] Target Z1, Z2 each [B, N].
        """

        with torch.no_grad():
            latent = self._encode(obs)
            return self.critic_target(latent, action)

    def get_cvar(self, obs: dict, action: torch.Tensor) -> torch.Tensor:
        """
        @brief Compute CVaR_alpha from Z1 quantiles for actor update.

        @param obs dict Batched observation dict [B, ...].
        @param action torch.Tensor Action tensor [B, action_dim].
        @return torch.Tensor CVaR values [B, 1].
        """

        latent = self._encode(obs)
        return self.critic.cvar(latent, action, alpha=self.cvar_alpha)
