import os
import sys
import torch

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", ".."
    )))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "rl"
    )))

    from managers.utils.logger import Log
    from algorithms.ppo.rollout_buffer import RolloutBuffer
    from algorithms.ppo.ppo_trainer import PPOTrainer
    from algorithms.ppo.ppo_policy import PPOPolicy
except ImportError as e:
    Log.error(__file__, e)


class PPOAgent:
    """
    @class PPOAgent
    @brief High-level PPO agent coordinating policy, buffer, and training.

    This class orchestrates interaction between the environment and PPO
    components, including action selection, experience storage, and policy updates.

    @details
    The agent integrates:
      - `PPOPolicy` for action selection and evaluation.
      - `RolloutBuffer` for storing experiences.
      - `PPOTrainer` for performing the PPO update using stored experiences.
    """

    def __init__(self, obs_config, action_dim, device="cpu"):
        """
        @brief Initialise the PPO agent.

        @param obs_config dict
            Observation encoder configuration including shapes of camera, LiDAR,
            ego-state, and risk features.
        @param action_dim int
            Dimensionality of the continuous action space (e.g., [steer, throttle, brake]).
        @param device str
            Torch device to use ('cpu' or 'cuda') for policy and buffer tensors.
        """

        self.device = device

        # --- Initialise policy ---
        self.policy = PPOPolicy(obs_config, action_dim, device=device)

        # --- Initialise rollout buffer ---
        self.buffer = RolloutBuffer(
            buffer_size=2048,
            obs_shape=None,  # inferred from observations during storage
            action_dim=action_dim,
            device=torch.device(device)
        )

        # --- Initialise trainer ---
        self.trainer = PPOTrainer(self.policy)

    def act(self, obs):
        """
        @brief Select an action using the current policy.

        @param obs dict
            Current observation batch containing keys like 'camera', 'lidar',
            'ego_state', and 'risk_features'.

        @return tuple(torch.Tensor, torch.Tensor, torch.Tensor)
            - action: Tensor of shape [batch_size, action_dim]
            - log_prob: Tensor of log probabilities of actions [batch_size]
            - value: Tensor of value estimates [batch_size]
        """
        return self.policy.act(obs)

    def store(self, obs, action, log_prob, reward, done, value):
        """
        @brief Store a transition in the rollout buffer.

        @param obs dict
            Observation corresponding to this timestep.
        @param action torch.Tensor
            Action taken by the agent.
        @param log_prob torch.Tensor
            Log probability of the selected action.
        @param reward float
            Reward received at this timestep.
        @param done bool
            Episode termination flag.
        @param value torch.Tensor
            Estimated value of the observation.
        """
        self.buffer.store(obs, action, log_prob, reward, done, value)

    def update(self, last_obs, done):
        """
        @brief Perform a PPO policy update using experiences in the buffer.

        @param last_obs dict
            Observation at the last timestep to compute advantages.
        @param done bool
            Indicates whether the episode ended at the last observation.

        @details
        This method:
          - Computes returns and advantages from stored transitions.
          - Updates the policy using the PPO algorithm.
          - Clears the buffer after the update.
        """
        self.trainer.update(self.buffer, last_obs, done)
        self.buffer.clear()