import os
import sys
import tty
import termios
import numpy as np
import torch

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from env.gym_carla_env import CarlaEnv
    from managers.utils.logger import Log
    from rl.common.checkpoint import CheckpointManager
    from rl.common.normalisation import ObservationNormaliser
    from rl.common.preprocessing import preprocess_obs
    from rl.logger.logger import TrainingLogger
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")
    sys.exit(1)

class TrainingSession:
    """
    @class TrainingSession
    @brief Owns shared setup and teardown for all RL training scripts.

    @details
    Encapsulates the boilerplate common to PPO, SAC, and CVaR-SAC training:
    environment initialisation, device selection, observation configuration,
    observation normaliser, checkpoint manager, TensorBoard logger, and
    terminal settings. Algorithm-specific logic (agent construction and the
    training loop) remains in the individual train_*.py scripts.

    Usage:
        session = TrainingSession(algo="ppo")
        try:
            session.setup()
            # training loop using session.env, session.obs_config, etc.
        except Exception as e:
            Log.error(__file__, e)
        finally:
            session.close()
    """

    def __init__(self, algo: str):
        """
        @brief Initialise shared training infrastructure.

        @param algo str Algorithm identifier: 'ppo', 'sac', or 'cvar_sac'.
        """

        self.algo = algo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)

        self.env = CarlaEnv(run_tag=algo)

        cam = self.env.config['sensors']['camera']
        cam_res = cam['train_resolution']
        lidar = self.env.config['sensors']['lidar']
        lidar_res = lidar['train_resolution']
        render_root = self.env.config['render']['root']

        self.obs_config = dict(
            camera_shape=(cam['channels'], cam_res['y'], cam_res['x']),
            lidar_shape=(lidar['channels'], lidar_res['y'], lidar_res['x']),
            ego_state_dim=6 + self.env.config['risk']['waypoints_ahead'] * 3,
            latent_dim=256,
            hidden_dim=128,
            n_reward_components=3,
            risk_feature_dim=self.env.risk_module.feature_dim,
        )

        self.obs_normaliser = ObservationNormaliser(
            ego_state_dim=self.obs_config['ego_state_dim'],
            risk_feature_dim=self.obs_config['risk_feature_dim'],
        )

        checkpoint_dir = os.path.abspath(os.path.join(ws_root_path, render_root, "checkpoints"))
        self.checkpoint_manager = CheckpointManager(parent_dir=checkpoint_dir, ws_dir=algo)

        log_dir = os.path.abspath(os.path.join(ws_root_path, render_root, "runs", algo))
        self.logger = TrainingLogger(log_dir=log_dir, algorithm=algo)

    def setup(self):
        """@brief Enable non-blocking keyboard input for graceful termination."""
        tty.setcbreak(self._fd)

    def preprocess(self, obs: dict) -> dict:
        """@brief Preprocess a raw observation into batched torch tensors."""
        return preprocess_obs(obs, self.env.config, self.device)

    def load_checkpoint(self, agent) -> int:
        """
        @brief Load the latest checkpoint into the agent and normaliser.

        @return int Step or rollout count from the checkpoint, or 0 if none found.
        """
        return self.checkpoint_manager.load(agent, self.obs_normaliser)

    def save_checkpoint(self, agent, count: int):
        """@brief Save agent and normaliser state to a checkpoint."""
        self.checkpoint_manager.save(agent, count, self.obs_normaliser)

    def log_episode(
        self,
        episode: int,
        steps: int,
        episode_reward: np.ndarray,
        info: dict,
        baseline: np.ndarray = None,
    ):
        """
        @brief Log the result of a completed episode to console and TensorBoard.

        @param episode int Episode index.
        @param steps int Number of steps completed in the episode.
        @param episode_reward np.ndarray Decomposed reward array [nav, safety, risk].
        @param info dict Info dict returned by the final env.step().
        @param baseline np.ndarray Optional baseline reward vector (PPO only).
        """

        goal_reached = info.get('goal_reached', False)
        collision = info.get('collision', False)
        off_route = info.get('off_route', False)
        near_miss = info.get('ttc_min', float('inf')) < 2.0

        if goal_reached:
            status = f"🎯 Goal reached after {steps} steps"
        elif collision:
            status = f"💥 Collision after {steps} steps"
        elif off_route:
            status = f"🚧 Off-route after {steps} steps"
        else:
            status = f"⏱️  Timeout after {steps} steps"

        log_text = (
            f"{status}; "
            f"waypoints: {info['wp_idx']}/{max(info['wp_total'], 1)}; "
            f"total: {episode_reward.sum():.2f} "
            f"[ nav: {episode_reward[0]:.2f}, safety: {episode_reward[1]:.2f}, risk: {episode_reward[2]:.2f} ]"
        )
        Log.info(__file__, log_text)

        if baseline is not None:
            self.logger.log_episode(episode, episode_reward, info, baseline)
        else:
            self.logger.log_episode(episode, episode_reward, info)
        self.logger.log_near_miss_rate(episode, near_miss)

    def close(self):
        """@brief Restore terminal settings and release environment and logger resources."""
        
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        if self.logger is not None:
            self.logger.close()
        if self.env is not None:
            self.env.close()
