import torch
import torch.nn as nn
import torch.optim as optim

class PPOTrainer:
    """
    @brief Proximal Policy Optimisation (PPO) trainer supporting multi-objective rewards.

    @details
    Implements standard PPO updates with optional multi-objective learning.
    Each objective can have a learnable weight, allowing the critic to learn
    separate value functions per reward component.
    """
    def __init__(
        self,
        agent,
        n_objectives: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 10,
        batch_size: int = 64,
    ):
        """
        @brief Initialise PPO trainer.

        @param agent Policy-value network to train.
        @param n_objectives int Number of reward components.
        @param lr float Learning rate.
        @param gamma float Discount factor.
        @param lam float GAE lambda.
        @param clip_eps float PPO clipping epsilon.
        @param value_coef float Scaling factor for critic loss.
        @param entropy_coef float Scaling factor for entropy bonus.
        @param epochs int Number of epochs per update.
        @param batch_size int Mini-batch size for optimisation.
        """
        self.agent = agent
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

        # Learnable objective weights initialised equally
        self.objective_weights = nn.Parameter(
            torch.ones(n_objectives, device=self.agent.device) / n_objectives
        )

        self.optimiser = optim.Adam(
            list(agent.parameters()) + [self.objective_weights],
            lr=lr
        )

    def update(self, buffer, last_obs, done: bool):
        """
        @brief Perform PPO update using data from rollout buffer.

        @details
        Computes advantages and returns, then updates actor and critic
        using multiple epochs and mini-batches. Supports multi-objective
        value function with learnable weighting.

        @param buffer Rollout buffer containing observations, actions, log_probs, etc.
        @param last_obs Last observation from the environment.
        @param done bool Whether the episode terminated at the last step.
        """
        device = self.agent.device

        # ---- Bootstrap value (no gradient tracking) ----
        with torch.no_grad():
            if done:
                last_value = torch.zeros(self.n_objectives, device=device)
            else:
                last_value = self.agent.act(last_obs)[2].squeeze(0)

        # ---- Detach weights for advantage computation ----
        weights_detached = torch.softmax(self.objective_weights, dim=0).detach()

        buffer.compute_returns_advantages(
            self.gamma,
            self.lam,
            last_value,
            objective_weights=weights_detached
        )

        # ---- Detach rollout data to prevent gradient leakage ----
        obs = buffer.obs
        actions = torch.stack(buffer.actions).to(device).detach()
        old_log_probs = torch.stack(buffer.log_probs).to(device).detach()
        returns = buffer.returns.to(device).detach()        # [T, n_objectives]
        advantages = buffer.advantages.to(device).detach()  # [T]

        dataset_size = len(actions)

        # ---- PPO optimisation loop ----
        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size, device=device)

            for i in range(0, dataset_size, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]

                batch_obs = self.collate_obs([obs[j.item()] for j in batch_idx])
                batch_actions = actions[batch_idx]
                batch_old_lp = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # ---- Forward pass (fresh computation graph) ----
                log_probs, entropy, values = self.agent.evaluate(
                    batch_obs, batch_actions
                )  # values: [B, n_objectives]

                # ---- Learnable weights (with gradient) ----
                weights = torch.softmax(self.objective_weights, dim=0)

                # ---- PPO ratio ----
                ratio = torch.exp(log_probs - batch_old_lp)

                # ---- Actor loss with clipping ----
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # ---- Critic loss per objective ----
                critic_loss_per_head = (batch_returns - values).pow(2).mean(dim=0)
                critic_loss = (critic_loss_per_head * weights).sum()

                # ---- Total loss ----
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                # ---- Backpropagation ----
                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                self.optimiser.step()

    def collate_obs(self, obs_batch):
        """
        @brief Collate a batch of observations into tensors.

        @param obs_batch List of observations (dicts of tensors).
        @return dict Dictionary of stacked tensors for each observation key.
        """
        return {
            k: torch.stack([o[k] for o in obs_batch]).to(self.agent.device)
            for k in obs_batch[0]
        }