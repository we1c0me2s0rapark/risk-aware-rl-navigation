import os
import sys
from torch.utils.tensorboard import SummaryWriter

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."
))

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from managers.utils.logger import Log
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

class TrainingLogger:
    """
    @class TrainingLogger
    @brief TensorBoard logging for PPO, SAC, and CVaR-SAC training runs.

    @details
    Wraps SummaryWriter to provide structured logging of training metrics.
    Supports PPO (rollout-based), SAC (step-based), and CVaR-SAC (step-based
    with additional CVaR distribution metrics).

    Logged metrics:

    Episode metrics (all algorithms, per episode):
        - reward/total, reward/navigation, reward/safety, reward/risk
        - reward/baseline
        - episode/collision, episode/goal_reached
        - episode/waypoint_completion
        - episode/ttc_min
        - safety/near_miss

    Rollout metrics (PPO only, per rollout):
        - loss/actor, loss/critic, loss/entropy, loss/total

    Step metrics (SAC, per step):
        - loss/actor, loss/critic, loss/alpha
        - sac/alpha, sac/buffer_size

    Step metrics (CVaR-SAC, extends SAC):
        - loss/actor, loss/critic, loss/alpha
        - sac/alpha, sac/buffer_size
        - cvar/mean        — mean CVaR across the batch
        - cvar/alpha       — the configured CVaR confidence level

    Usage:
        logger = TrainingLogger(log_dir="runs/ppo", algorithm="ppo")
        logger.log_episode(episode=1, reward=np.array([...]), info={...})
        logger.log_ppo_losses(rollout=5, actor_loss=0.1, critic_loss=0.3, entropy=0.01)
        logger.log_sac_losses(step=1000, losses={...})
        logger.close()
    """

    def __init__(self, log_dir: str, algorithm: str = "ppo"):
        """
        @brief Initialise the TensorBoard logger.

        @param log_dir str Directory to write TensorBoard event files.
        @param algorithm str Algorithm identifier ('ppo', 'sac', or 'cvar_sac').
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.algorithm = algorithm.lower()

        Log.info(__file__, f"TensorBoard logging to: {log_dir}")
        Log.info(__file__, f"Run: tensorboard --logdir={os.path.dirname(log_dir)}")

    def log_episode(
        self,
        episode: int,
        reward,
        info: dict,
        baseline_reward=None,
    ):
        """
        @brief Log per-episode metrics.

        @param episode int Episode index.
        @param reward np.ndarray Decomposed reward [nav, safety, risk].
        @param info dict Info dict from env.step().
        @param baseline_reward np.ndarray Optional baseline reward for comparison.
        """
        import numpy as np
        reward = np.asarray(reward)

        self.writer.add_scalar('reward/total',      float(reward.sum()),  episode)
        self.writer.add_scalar('reward/navigation', float(reward[0]),     episode)
        self.writer.add_scalar('reward/safety',     float(reward[1]),     episode)
        self.writer.add_scalar('reward/risk',       float(reward[2]),     episode)

        if baseline_reward is not None:
            baseline_reward = np.asarray(baseline_reward)
            self.writer.add_scalar('reward/baseline', float(baseline_reward.sum()), episode)

        self.writer.add_scalar('episode/collision',
            float(info.get('collision', False)), episode)
        self.writer.add_scalar('episode/goal_reached',
            float(info.get('goal_reached', False)), episode)

        # Route completion as percentage
        wp_idx = info.get('wp_idx', 0)
        wp_total = max(info.get('wp_total', 1), 1)
        self.writer.add_scalar('episode/waypoint_completion',
            wp_idx / wp_total, episode)

        # TTC min for near-miss tracking
        ttc_min = info.get('ttc_min', float('inf'))
        if ttc_min != float('inf'):
            self.writer.add_scalar('episode/ttc_min', ttc_min, episode)

    def log_ppo_losses(
        self,
        rollout: int,
        actor_loss: float,
        critic_loss: float,
        entropy: float,
        total_loss: float = None,
    ):
        """
        @brief Log PPO update losses per rollout.

        @param rollout int Rollout index.
        @param actor_loss float PPO surrogate actor loss.
        @param critic_loss float Critic value loss.
        @param entropy float Mean policy entropy.
        @param total_loss float Optional combined loss.
        """
        
        self.writer.add_scalar('loss/actor',   actor_loss,  rollout)
        self.writer.add_scalar('loss/critic',  critic_loss, rollout)
        self.writer.add_scalar('loss/entropy', entropy,     rollout)

        if total_loss is not None:
            self.writer.add_scalar('loss/total', total_loss, rollout)

    def log_sac_losses(self, step: int, losses: dict):
        """
        @brief Log SAC or CVaR-SAC update losses per step.

        @details
        Also handles CVaR-SAC losses — if 'cvar_mean' is present in the
        losses dict, it is logged under the cvar/ namespace automatically.

        @param step int Global step index.
        @param losses dict Keys: 'critic_loss', 'actor_loss', 'alpha_loss',
                           'alpha', and optionally 'cvar_mean' (CVaR-SAC only).
        """

        if losses is None:
            return

        self.writer.add_scalar('loss/critic', losses.get('critic_loss', 0.0), step)
        self.writer.add_scalar('loss/actor',  losses.get('actor_loss',  0.0), step)
        self.writer.add_scalar('loss/alpha',  losses.get('alpha_loss',  0.0), step)
        self.writer.add_scalar('sac/alpha',   losses.get('alpha',       0.0), step)

        # CVaR-SAC specific — logged automatically if present
        if 'cvar_mean' in losses:
            self.writer.add_scalar('cvar/mean', losses['cvar_mean'], step)

    def log_cvar_alpha(self, step: int, cvar_alpha: float):
        """
        @brief Log the CVaR confidence level (CVaR-SAC only).

        @details
        Logs the configured CVaR alpha periodically so it appears alongside
        the cvar/mean curve in TensorBoard for reference.

        @param step int Global step index.
        @param cvar_alpha float CVaR confidence level (e.g. 0.1).
        """

        self.writer.add_scalar('cvar/alpha', cvar_alpha, step)

    def log_buffer_size(self, step: int, size: int):
        """
        @brief Log replay buffer size (SAC and CVaR-SAC).

        @param step int Global step index.
        @param size int Current number of transitions in buffer.
        """

        self.writer.add_scalar('sac/buffer_size', size, step)

    def log_near_miss_rate(self, episode: int, near_miss: bool, ttc_threshold: float = 2.0):
        """
        @brief Log near-miss event based on TTC threshold.

        @param episode int Episode index.
        @param near_miss bool Whether TTC dropped below threshold this episode.
        @param ttc_threshold float TTC threshold in seconds defining a near-miss.
        """

        self.writer.add_scalar('safety/near_miss', float(near_miss), episode)

    def close(self):
        """@brief Flush and close the TensorBoard writer."""

        self.writer.flush()
        self.writer.close()