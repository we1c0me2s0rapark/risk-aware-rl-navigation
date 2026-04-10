import torch

class RolloutBuffer:
    """
    @class RolloutBuffer
    @brief Stores trajectories and computes returns/advantages for PPO.

    @details
    This class collects observations, actions, log probabilities, rewards,
    done flags, and value estimates during rollout. It supports computing
    returns and advantages using Generalised Advantage Estimation (GAE)
    for policy optimisation.

    - Handles variable-length episodes.
    - Automatically normalises advantages for stable learning.
    - Detaches actions, log_probs, and values to avoid unwanted gradient tracking.
    """

    def __init__(self, buffer_size: int, obs_shape, action_dim: int, device):
        """
        @brief Initialise the rollout buffer.

        @param buffer_size int Maximum number of timesteps to store.
        @param obs_shape tuple or None Shape of observations (reserved for future preallocation).
        @param action_dim int Number of action dimensions.
        @param device torch.device or str Device to store tensors ('cpu' or 'cuda').
        """
        self.device = device
        self.ptr = 0
        self.max_size = buffer_size
        self._init_storage()

    def _init_storage(self):
        """@brief Initialise or reset storage lists for trajectories."""
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        """
        @brief Clear the buffer and reset the pointer.

        @details
        Typically called after a PPO update to start a new rollout.
        """
        self.ptr = 0
        self._init_storage()

    def store(self, obs, action: torch.Tensor, log_prob: torch.Tensor, reward: float, done: bool, value: torch.Tensor):
        """
        @brief Store a single transition in the buffer.

        @param obs dict Observation at the current timestep.
        @param action torch.Tensor Action taken.
        @param log_prob torch.Tensor Log probability of the action.
        @param reward float Reward received after taking the action.
        @param done bool Whether the episode terminated after this step.
        @param value torch.Tensor Value estimate from critic.

        @details
        - Detaches tensors to prevent gradients from leaking through the buffer.
        - Converts reward and done to device tensors.
        """
        self.obs.append(obs)
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.values.append(value.detach())
        self.dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))

    def compute_returns_advantages(self, gamma: float = 0.99, lam: float = 0.95, last_value=0, objective_weights=None):
        """
        @brief Compute returns and advantages for all timesteps in the buffer.

        @param gamma float Discount factor.
        @param lam float GAE lambda.
        @param last_value torch.Tensor or float Value estimate for the final timestep (bootstrap).
        @param objective_weights torch.Tensor Optional weighting per objective.

        @details
        - Computes Generalised Advantage Estimation (GAE) per objective.
        - Combines multiple objectives using learnable or provided weights.
        - Advantages are normalised for stable policy optimisation.
        """
        n_objectives = self.values[0].shape[-1]
        T = len(self.rewards)

        advantages = torch.zeros(T, n_objectives, device=self.device)
        gae = torch.zeros(n_objectives, device=self.device)
        next_value = last_value \
            if isinstance(last_value, torch.Tensor) \
            else torch.zeros(n_objectives, device=self.device)

        values = torch.stack(self.values)  # [T, n_objectives]

        # Compute GAE backwards
        for t in reversed(range(T)):
            done_mask = self.dones[t].float()
            delta = self.rewards[t] + gamma * next_value * (1 - done_mask) - values[t]
            gae = delta + gamma * lam * (1 - done_mask) * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values  # [T, n_objectives]

        # Apply objective weights
        if objective_weights is None:
            objective_weights = torch.ones(n_objectives, device=self.device) / n_objectives
        else:
            objective_weights = objective_weights.to(self.device)

        # Weighted sum of advantages per timestep, then normalise
        self.advantages = (advantages * objective_weights).sum(dim=-1)  # [T]
        self.returns = returns  # [T, n_objectives] - used for per-head critic loss
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)