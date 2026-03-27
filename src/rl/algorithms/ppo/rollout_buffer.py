import torch

class RolloutBuffer:
    """
    @class RolloutBuffer
    @brief Stores trajectories and computes returns/advantages for PPO.

    This class collects observations, actions, log probabilities, rewards,
    done flags, and value estimates during rollout. It supports computing
    returns and advantages using Generalised Advantage Estimation (GAE)
    for policy optimisation.

    @details
    - Handles variable-length episodes.
    - Automatically normalises advantages for stable learning.
    - Detaches actions, log_probs, and values to avoid unwanted gradient tracking.
    """

    def __init__(self, buffer_size, obs_shape, action_dim, device):
        """
        @brief Initialise the rollout buffer.

        @param buffer_size int Maximum number of timesteps to store.
        @param obs_shape tuple or None Shape of observations (unused here, for future preallocation).
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

        This is typically called after a PPO update to start a new rollout.
        """
        self.ptr = 0
        self._init_storage()

    def store(self, obs, action, log_prob, reward, done, value):
        """
        @brief Store a single transition in the buffer.

        @param obs dict Observation at current timestep.
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
        self.rewards.append(torch.tensor(reward, device=self.device))
        self.dones.append(torch.tensor(done, device=self.device))
        self.values.append(value.detach())

    def compute_returns_advantages(self, gamma=0.99, lam=0.95, last_value=0):
        """
        @brief Compute returns and advantages for the stored trajectory using GAE.

        @param gamma float Discount factor for future rewards.
        @param lam float Lambda parameter for GAE smoothing.
        @param last_value torch.Tensor or float Bootstrapped value for final timestep (0 if done).

        @details
        - Converts boolean done flags to float to allow arithmetic operations.
        - Computes advantages using:
          \f$ \delta_t = r_t + \gamma V_{t+1} (1 - done_t) - V_t \f$
          \f$ A_t = \delta_t + \gamma \lambda (1 - done_t) A_{t+1} \f$
        - Returns are computed as:
          \f$ R_t = A_t + V_t \f$
        - Advantages are normalised for numerical stability.
        """
        advantages = []
        gae = 0
        next_value = last_value

        values = torch.stack(self.values).squeeze(-1)

        for step in reversed(range(len(self.rewards))):
            # Ensure done flag is float to allow arithmetic
            done_mask = self.dones[step].float()
            delta = self.rewards[step] + gamma * next_value * (1.0 - done_mask) - self.values[step]
            gae = delta + gamma * lam * (1.0 - done_mask) * gae

            advantages.insert(0, gae)
            next_value = self.values[step]

        returns = [adv + val for adv, val in zip(advantages, values)]

        self.advantages = torch.stack(advantages).to(self.device)
        self.returns = torch.stack(returns).to(self.device)

        # normalise advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)