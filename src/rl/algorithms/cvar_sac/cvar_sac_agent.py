import os
import sys
import torch

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from managers.utils.logger import Log
    from rl.algorithms.sac.replay_buffer import ReplayBuffer
    from rl.algorithms.cvar_sac.cvar_sac_trainer import CVaRSACTrainer
    from rl.algorithms.cvar_sac.cvar_sac_policy import CVaRSACPolicy
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

class CVaRSACAgent:
    """
    @class CVaRSACAgent
    @brief High-level CVaR-SAC agent coordinating policy, buffer, and training.

    @details
    Mirrors SACAgent but uses CVaRSACPolicy (distributional critic) and
    CVaRSACTrainer (quantile regression loss + CVaR actor objective).

    The replay buffer is shared - same off-policy ReplayBuffer as SAC.
    Only the policy and trainer differ from the standard SAC agent.
    """

    def __init__(
        self,
        obs_config: dict,
        action_dim: int,
        n_quantiles: int = 32,
        cvar_alpha: float = 0.1,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        """
        @brief Initialise the CVaR-SAC agent.

        @param obs_config dict Observation encoder configuration.
        @param action_dim int Dimension of the continuous action space.
        @param n_quantiles int Number of quantiles for distributional critic.
        @param cvar_alpha float CVaR confidence level (default 0.1 = worst 10%).
        @param buffer_capacity int Maximum replay buffer size.
        @param device str Torch device ('cpu' or 'cuda').
        """
        self.device = device

        # --- Distributional policy ---
        self.policy = CVaRSACPolicy(
            obs_config=obs_config,
            action_dim=action_dim,
            device=device,
            n_quantiles=n_quantiles,
            cvar_alpha=cvar_alpha,
        )

        # --- Replay buffer (same as SAC) ---
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            device=torch.device(device),
            min_samples=batch_size,
        )

        # --- CVaR trainer ---
        self.trainer = CVaRSACTrainer(self.policy)

    def act(self, obs: dict, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief Select an action using the current policy.

        @param obs dict Current observation batch [1, ...].
        @param deterministic bool If True, use mean action (evaluation mode).
        @return tuple(torch.Tensor, torch.Tensor)
            - action [1, action_dim]
            - log_prob [1, 1] (None if deterministic)
        """

        return self.policy.act(obs, deterministic=deterministic)

    def store(self, obs: dict, action: torch.Tensor, reward, next_obs: dict, done: bool):
        """
        @brief Store a transition in the replay buffer.

        @param obs dict Unbatched observation at current timestep.
        @param action torch.Tensor Action taken [action_dim].
        @param reward Reward received (scalar or vector).
        @param next_obs dict Unbatched observation at next timestep.
        @param done bool Episode termination flag.
        """
        self.buffer.store(obs, action, reward, next_obs, done)

    def update(self, batch_size: int = 256) -> dict | None:
        """
        @brief Perform a CVaR-SAC update if buffer has enough transitions.

        @param batch_size int Minibatch size for the update.
        @return dict Loss metrics or None if buffer not ready.
        """

        if len(self.buffer) < batch_size:
            return None
        return self.trainer.update(self.buffer, batch_size=batch_size)
