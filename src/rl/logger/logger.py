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
    Log.error(__file__, e)

class TrainingLogger:
    """
    @class TrainingLogger
    @brief TensorBoard logging for PPO and SAC training runs.

    @details
    Wraps SummaryWriter to provide structured logging of training metrics.
    Supports both PPO (rollout-based) and SAC (step-based) logging patterns.

    Logged metrics:

    Episode metrics (logged per episode):
        - reward/total
        - reward/navigation
        - reward/safety
        - reward/risk
        - reward/baseline
        - episode/steps
        - episode/collision
        - episode/goal_reached
        - episode/waypoint_completion

    Rollout metrics (PPO only, logged per rollout):
        - loss/actor
        - loss/critic
        - loss/entropy
        - loss/total

    Step metrics (SAC only, logged per update step):
        - loss/actor
        - loss/critic
        - loss/alpha
        - sac/alpha

    Usage:
        logger = TrainingLogger(log_dir="runs/ppo")
        logger.log_episode(episode=1, reward=np.array([...]), info={...})
        logger.log_ppo_losses(rollout=5, actor_loss=0.1, critic_loss=0.3, entropy=0.01)
        logger.log_sac_losses(step=1000, losses={...})
        logger.close()
    """

    def __init__(self, log_dir: str, algorithm: str = "ppo"):
        """
        @brief Initialise the TensorBoard logger.

        @param log_dir str Directory to write TensorBoard event files.
        @param algorithm str Algorithm identifier ('ppo' or 'sac') for log organisation.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.algorithm = algorithm.lower()

        Log.info(__file__, f"[TrainingLogger] TensorBoard logging to: {log_dir}")
        Log.info(__file__, f"[TrainingLogger] Run: tensorboard --logdir={os.path.dirname(log_dir)}")

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
        @brief Log SAC update losses per step.

        @param step int Global step index.
        @param losses dict Keys: 'critic_loss', 'actor_loss', 'alpha_loss', 'alpha'.
        """
        if losses is None:
            return

        self.writer.add_scalar('loss/critic',    losses.get('critic_loss', 0.0), step)
        self.writer.add_scalar('loss/actor',     losses.get('actor_loss',  0.0), step)
        self.writer.add_scalar('loss/alpha',     losses.get('alpha_loss',  0.0), step)
        self.writer.add_scalar('sac/alpha',      losses.get('alpha',       0.0), step)

    def log_buffer_size(self, step: int, size: int):
        """
        @brief Log replay buffer size (SAC).

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