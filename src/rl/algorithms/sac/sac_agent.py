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
    from rl.algorithms.sac.sac_trainer import SACTrainer
    from rl.algorithms.sac.sac_policy import SACPolicy
except ImportError as e:
    Log.error(__file__, e)

class SACAgent:
    """
    @class SACAgent
    @brief High-level SAC agent coordinating policy, buffer, and training.

    @details
    The agent integrates:
      - `SACPolicy` for action selection and evaluation.
      - `ReplayBuffer` for storing experiences (off-policy).
      - `SACTrainer` for performing the SAC update using stored experiences.
    """

    def __init__(self, obs_config: dict, action_dim: int, buffer_capacity: int = 100_000, min_samples: int = 256, device: str = "cpu"):
        """
        @brief Initialise the SAC agent.

        @param obs_config dict Observation encoder configuration.
        @param action_dim int Dimensionality of the continuous action space.
        @param device str Torch device ('cpu' or 'cuda').
        """
        self.device = device

        # --- Initialise policy ---
        self.policy = SACPolicy(obs_config, action_dim, device=device)

        # --- Initialise replay buffer ---
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            device=torch.device(device),
            min_samples=min_samples
        )

        # --- Initialise trainer ---
        self.trainer = SACTrainer(self.policy)

    def act(self, obs: dict, deterministic: bool = False):
        """
        @brief Select an action using the current policy.

        @param obs dict Current observation batch.
        @param deterministic bool If True, use mean action (evaluation mode).
        @return tuple(torch.Tensor, torch.Tensor)
            - action [1, action_dim]
            - log_prob [1, 1] (None if deterministic)
        """
        return self.policy.act(obs, deterministic=deterministic)

    def store(self, obs: dict, action: torch.Tensor, reward, next_obs: dict, done: bool):
        """
        @brief Store a transition in the replay buffer.

        @param obs dict Observation at current timestep.
        @param action torch.Tensor Action taken.
        @param reward Reward received (scalar or vector).
        @param next_obs dict Observation at next timestep.
        @param done bool Episode termination flag.
        """
        self.buffer.store(obs, action, reward, next_obs, done)

    def update(self) -> dict | None:
        """
        @brief Perform a CVaR-SAC update if buffer has enough transitions.

        @return dict Loss metrics or None if buffer not ready.
        """
        if not self.buffer.ready:
            return None
        return self.trainer.update(self.buffer, batch_size=self.buffer.min_samples)