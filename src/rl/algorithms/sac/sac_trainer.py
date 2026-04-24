import torch
import torch.nn.functional as F
import numpy as np

class SACTrainer:
    """
    @class SACTrainer
    @brief Implements the SAC update rule with automatic entropy tuning.

    @details
    SAC optimises three objectives jointly:
        1. Critic:  minimise Bellman error for Q1 and Q2
        2. Actor:   maximise Q(s,a) - α * log π(a|s)
        3. Entropy: tune temperature α to maintain target entropy

    The decomposed reward (3 objectives) is summed into a scalar before
    the Bellman backup, allowing the multi-head environment reward to work
    with the standard SAC update without architectural changes.

    Target entropy is set to -action_dim as per the original SAC paper,
    which works well for continuous control tasks.
    """

    def __init__(
        self,
        policy,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float = None,
        reward_weights: list = None,
    ):
        """
        @brief Initialise the SAC trainer.

        @param policy SACPolicy The policy containing encoder, actor, and critic.
        @param lr_actor float Learning rate for actor network.
        @param lr_critic float Learning rate for critic networks.
        @param lr_alpha float Learning rate for entropy temperature.
        @param gamma float Discount factor.
        @param tau float Polyak averaging coefficient for target network.
        @param target_entropy float Target entropy. Defaults to -action_dim.
        @param reward_weights list[float] Weights for decomposed reward components.
                              Defaults to [1.0, 1.0, 1.0].
        """
        self.policy = policy
        self.gamma = gamma
        self.tau = tau
        self.device = policy.device

        # ---- Reward decomposition weights ----
        # Scalar reward = w · [r_navigation, r_safety, r_risk]
        if reward_weights is None:
            reward_weights = [1.0, 1.0, 1.0]
        self.reward_weights = torch.tensor(
            reward_weights, dtype=torch.float32, device=self.device
        )

        # ---- Target entropy ----
        # Default: -dim(A) as per SAC paper (Haarnoja et al. 2018)
        if target_entropy is None:
            target_entropy = -float(policy.action_dim)
        self.target_entropy = target_entropy

        # ---- Learnable log temperature ----
        # log α is optimised directly for numerical stability
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()

        # ---- Optimisers ----
        self.actor_optimiser = torch.optim.Adam(
            policy.actor.parameters(), lr=lr_actor
        )
        self.critic_optimiser = torch.optim.Adam(
            policy.critic.parameters(), lr=lr_critic
        )
        self.alpha_optimiser = torch.optim.Adam(
            [self.log_alpha], lr=lr_alpha
        )

    def _scalar_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        @brief Reduce decomposed reward vector to a scalar via weighted sum.

        @param rewards torch.Tensor Reward tensor of shape [B, n_objectives] or [B].
        @return torch.Tensor Scalar reward of shape [B].
        """
        if rewards.dim() == 1:
            return rewards # already scalar

        # [B, n_objectives] @ [n_objectives] → [B]
        return (rewards * self.reward_weights).sum(dim=-1)

    def update(self, buffer, batch_size: int = 256):
        """
        @brief Perform one SAC update step from a sampled minibatch.

        @details
        Update order:
            1. Sample batch from replay buffer
            2. Compute critic loss (Bellman backup with target networks)
            3. Compute actor loss (maximise Q - α * log π)
            4. Compute alpha loss (match entropy to target)
            5. Soft update target critic

        @param buffer ReplayBuffer Replay buffer to sample from.
        @param batch_size int Minibatch size.
        @return dict Losses for logging:
            - 'critic_loss', 'actor_loss', 'alpha_loss', 'alpha'
        """
        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

        # Reduce decomposed reward to scalar [B]
        rewards = self._scalar_reward(rewards)

        # Encode each obs once; encoder is frozen (not in any optimiser)
        with torch.no_grad():
            next_latent = self.policy._encode(next_obs)
            next_action, next_log_prob, _ = self.policy.actor.sample(next_latent)
            target_q1, target_q2 = self.policy.critic_target(next_latent, next_action)
            target_q = torch.min(target_q1, target_q2).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - dones) * (
                target_q - self.alpha * next_log_prob.squeeze(-1)
            )
            target_q = target_q.unsqueeze(-1)

        obs_latent = self.policy._encode(obs).detach()

        # ================================================================
        # 1. Critic update
        # ================================================================
        q1, q2 = self.policy.critic(obs_latent, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # ================================================================
        # 2. Actor update
        # ================================================================
        action_new, log_prob_new, _ = self.policy.actor.sample(obs_latent)
        q1_new = self.policy.critic.q1_forward(obs_latent, action_new)
        actor_loss = (self.alpha * log_prob_new - q1_new).mean()

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # ================================================================
        # 3. Alpha (entropy temperature) update
        # ================================================================
        alpha_loss = -(
            self.log_alpha * (log_prob_new.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        self.alpha = self.log_alpha.exp().item()

        # ================================================================
        # 4. Soft update target critic
        # ================================================================
        self.policy.soft_update_target(self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
        }