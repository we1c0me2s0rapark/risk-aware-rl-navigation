import torch
import torch.nn as nn
import torch.optim as optim

class PPOTrainer:
    """
    @class PPOTrainer
    @brief Trainer class implementing Proximal Policy Optimisation (PPO) updates.

    This class handles the PPO optimisation procedure for a given policy,
    including computing surrogate losses, clipping, and updating both
    actor and critic networks.

    @details
    - Supports mini-batch updates over multiple epochs.
    - Uses advantage normalisation internally via the rollout buffer.
    - Applies the PPO clipped objective to stabilise policy updates.
    """

    def __init__(
        self,
        agent,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=10,
        batch_size=64,
    ):
        """
        @brief Initialise the PPOTrainer.

        @param agent nn.Module
            Policy network or PPOAgent implementing `act()` and `evaluate()`.
        @param lr float
            Learning rate for the Adam optimiser.
        @param gamma float
            Discount factor for reward accumulation.
        @param lam float
            GAE lambda parameter for advantage estimation.
        @param clip_eps float
            PPO clipping epsilon for policy ratio.
        @param value_coef float
            Weight for the critic loss in the combined PPO loss.
        @param entropy_coef float
            Weight for the policy entropy bonus to encourage exploration.
        @param epochs int
            Number of epochs per update iteration.
        @param batch_size int
            Mini-batch size for stochastic gradient updates.
        """

        self.agent = agent
        self.optimiser = optim.Adam(agent.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def update(self, buffer, last_obs, done):
        """
        @brief Perform a PPO update using the rollout buffer.

        @param buffer RolloutBuffer
            Buffer storing collected trajectories including observations, actions,
            rewards, log probabilities, and value estimates.
        @param last_obs dict
            Observation at the last timestep to compute bootstrapped value for GAE.
        @param done bool
            Flag indicating if the episode terminated at the last observation.

        @details
        The update procedure:
          1. Compute returns and advantages using GAE.
          2. Split experiences into mini-batches and shuffle them.
          3. Compute the clipped surrogate loss for the actor.
          4. Compute the critic loss using mean squared error.
          5. Combine losses with entropy regularisation.
          6. Perform gradient descent step on the policy network.
        """
        with torch.no_grad():
            if done:
                last_value = 0
            else:
                last_value = self.agent.act(last_obs)[2].detach()

        buffer.compute_returns_advantages(self.gamma, self.lam, last_value)

        obs = buffer.obs
        actions = torch.stack(buffer.actions).to(self.agent.device)
        old_log_probs = torch.stack(buffer.log_probs).to(self.agent.device)
        returns = buffer.returns
        advantages = buffer.advantages

        dataset_size = len(actions)

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)

            for i in range(0, dataset_size, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]

                batch_obs = self.collate_obs([obs[j] for j in batch_idx])
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                log_probs, entropy, values = self.agent.evaluate(batch_obs, batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - values).pow(2).mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

    def collate_obs(self, obs_batch):
        """
        @brief Convert a batch of observation dictionaries into a batched tensor dict.

        @param obs_batch list of dict
            Each dict contains observation components (camera, lidar, ego_state, etc.).
        @return dict
            Batched tensors for each observation component, suitable for input to the policy network.
        """
        return {
            k: torch.stack([o[k] for o in obs_batch]).to(self.agent.device)
            for k in obs_batch[0]
        }