import os
import sys
import torch
import torch.nn.functional as F

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from rl.algorithms.sac.sac_trainer import SACTrainer
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

class CVaRSACTrainer(SACTrainer):
    """
    @class CVaRSACTrainer
    @brief CVaR-SAC trainer extending SACTrainer with quantile regression loss.

    @details
    Inherits the full SAC update structure from SACTrainer and overrides
    the critic and actor update steps to support distributional Q-learning
    and CVaR-based policy optimisation.

    Key differences from SACTrainer:

    1. Critic loss - quantile regression (Huber-based) instead of MSE:
          L = (1/N²) Σ_i Σ_j ρ_τi(r + γZ_target_j - Z_i)
       where ρ_τ(u) = |τ - 1(u < 0)| * HuberLoss(u)

    2. Bellman target - distributional backup:
          Z_target = r + γ(1-done)(min(Z1_target, Z2_target) - α*log_π)
       min is taken quantile-wise to prevent overestimation.

    3. Actor loss - CVaR_α(Z1) instead of E[Q1]:
          actor_loss = α*log_π - CVaR_α(Z1(s,a))
       This biases the policy toward actions that reduce worst-case outcomes,
       implementing risk-sensitive behaviour.

    Alpha (entropy temperature) update is inherited unchanged from SACTrainer.
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
        kappa: float = 1.0,
    ):
        """
        @brief Initialise CVaRSACTrainer.

        @details
        Calls SACTrainer.__init__() which sets up all three optimisers
        and the entropy temperature. Only adds kappa for the Huber loss.

        @param policy CVaRSACPolicy Policy with distributional critic.
        @param lr_actor float Learning rate for actor.
        @param lr_critic float Learning rate for distributional critic.
        @param lr_alpha float Learning rate for entropy temperature.
        @param gamma float Discount factor.
        @param tau float Polyak averaging coefficient for target network.
        @param target_entropy float Target entropy. Defaults to -action_dim.
        @param reward_weights list[float] Weights for decomposed reward components.
        @param kappa float Huber loss threshold for quantile regression (default 1.0).
        """
        super().__init__(
            policy=policy,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            gamma=gamma,
            tau=tau,
            target_entropy=target_entropy,
            reward_weights=reward_weights,
        )

        self.kappa = kappa
        self.n_quantiles = policy.n_quantiles

        # Midpoint quantile levels τ ∈ (0, 1) for QR loss
        # τ_i = (2i - 1) / (2N) for i = 1..N
        taus = torch.FloatTensor(
            [(2 * i - 1) / (2 * self.n_quantiles) for i in range(1, self.n_quantiles + 1)]
        ).to(policy.device)
        self.register_taus(taus)

    def register_taus(self, taus: torch.Tensor):
        """
        @brief Store quantile levels as a non-trainable buffer.

        @param taus torch.Tensor Quantile midpoints [N].
        """

        self.taus = taus

    def _quantile_regression_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        @brief Compute the quantile regression (Huber) loss.

        @details
        For each pair of predicted quantile z_i and target quantile z_j:
            δ_ij = z_j - z_i
            ρ_τi(δ) = |τ_i - 1(δ < 0)| * HuberLoss(δ, kappa)

        Loss is averaged over target quantiles and summed over predicted
        quantiles, then normalised by batch size.

        @param predicted torch.Tensor Predicted quantiles [B, N].
        @param target torch.Tensor Target quantiles [B, N] (detached).
        @return torch.Tensor Scalar quantile regression loss.
        """

        B, N = predicted.shape

        # Expand for pairwise comparison: [B, N, 1] vs [B, 1, N]
        pred_expanded = predicted.unsqueeze(2) # [B, N, 1]
        target_expanded = target.unsqueeze(1) # [B, 1, N]

        # Pairwise TD errors: δ_ij = target_j - predicted_i → [B, N, N]
        delta = target_expanded - pred_expanded

        # Huber loss component
        abs_delta = delta.abs()
        huber = torch.where(
            abs_delta <= self.kappa,
            0.5 * delta.pow(2),
            self.kappa * (abs_delta - 0.5 * self.kappa)
        ) # [B, N, N]

        # Asymmetric quantile weighting: |τ_i - 1(δ_ij < 0)|
        taus = self.taus.view(1, N, 1) # [1, N, 1] - broadcast over B and target dim
        indicator = (delta < 0).float()
        weight = (taus - indicator).abs() # [B, N, N]

        # Quantile regression loss: mean over target quantiles, sum over predicted
        loss = (weight * huber).mean(dim=2).sum(dim=1).mean()

        return loss

    def update(self, buffer, batch_size: int = 256) -> dict:
        """
        @brief Perform one CVaR-SAC update step.

        @details
        Update order (same as SACTrainer except steps 1 and 2):
            1. Sample batch from replay buffer
            2. Compute distributional Bellman target (quantile-wise min)
            3. Compute quantile regression critic loss
            4. Compute actor loss using CVaR_α(Z1) instead of E[Q1]
            5. Compute alpha loss (inherited, unchanged)
            6. Soft update target critic

        @param buffer ReplayBuffer Replay buffer to sample from.
        @param batch_size int Minibatch size.
        @return dict Loss metrics: critic_loss, actor_loss, alpha_loss, alpha, cvar_mean.
        """

        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

        # Reduce decomposed reward to scalar [B]
        rewards = self._scalar_reward(rewards)

        # ================================================================
        # 1. Distributional critic update
        # ================================================================
        with torch.no_grad():
            # Sample next action and log prob from current policy
            next_action, next_log_prob, _ = self.policy.actor.sample(
                self.policy._encode(next_obs)
            )

            # Get target quantile distributions
            target_z1, target_z2 = self.policy.get_target_q_values(next_obs, next_action)

            # Quantile-wise minimum to prevent overestimation [B, N]
            target_z = torch.min(target_z1, target_z2)

            # next_log_prob: [B, 1] → reshape to [B, 1] for clean broadcast against [B, N]
            entropy_term = (self.alpha * next_log_prob).expand_as(target_z) # [B, N]

            target_z = (
                rewards.unsqueeze(1) +
                self.gamma * (1.0 - dones.unsqueeze(1)) * (target_z - entropy_term)
            )

        # Current quantile predictions
        z1, z2 = self.policy.get_q_values(obs, actions)

        # Quantile regression loss for both networks
        critic_loss = (
            self._quantile_regression_loss(z1, target_z.detach()) +
            self._quantile_regression_loss(z2, target_z.detach())
        )

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # ================================================================
        # 2. CVaR actor update
        # ================================================================
        action_new, log_prob_new, latent = self.policy.evaluate(obs)

        # CVaR_α(Z1) - expected return in worst α fraction of outcomes
        cvar = self.policy.critic.cvar(latent, action_new, alpha=self.policy.cvar_alpha)

        # Actor loss: maximise CVaR - α*log_π (minimise negative CVaR + entropy cost)
        actor_loss = (self.alpha * log_prob_new - cvar).mean()

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        # ================================================================
        # 3. Alpha update (inherited from SACTrainer - unchanged)
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
            'actor_loss':  actor_loss.item(),
            'alpha_loss':  alpha_loss.item(),
            'alpha':       self.alpha,
            'cvar_mean':   cvar.mean().item(), # track mean CVaR for TensorBoard
        }
