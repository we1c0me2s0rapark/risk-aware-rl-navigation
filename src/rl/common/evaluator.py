import os
import sys
import torch
import numpy as np
from datetime import datetime

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from managers.utils.logger import Log
    from rl.common.checkpoint import CheckpointManager
    from rl.common.normalisation import ObservationNormaliser
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")
    sys.exit(1)

SUPPORTED_ALGOS = ("ppo", "sac", "cvar_sac")

def _preprocess_obs(obs: dict, config: dict, device: torch.device) -> dict:
    """
    @brief Preprocess raw CARLA observations into batched torch tensors.

    @details
    Converts camera, LiDAR, ego state, and risk features from raw numpy
    arrays into normalised float32 tensors suitable for policy inference.
    LiDAR points are projected into a 2D bird's-eye-view occupancy grid.

    @param obs dict Raw observation dictionary from the CARLA environment.
    @param config dict Environment configuration specifying sensor properties.
    @param device torch.device Target device for the output tensors.
    @return dict Batched tensors with keys: camera, lidar, ego_state, risk_features.
    """

    cam_cfg = config['sensors']['camera']
    cam_res = cam_cfg['train_resolution']
    channels = cam_cfg['channels']
    camera = np.array(obs["camera"], dtype=np.float32).reshape(channels, cam_res['y'], cam_res['x'])

    lidar_cfg = config['sensors']['lidar']
    lidar_res = lidar_cfg['train_resolution']
    lidar_range = lidar_cfg['range']
    pts = np.array(obs["lidar"], dtype=np.float32).reshape(-1, 3)
    grid_h, grid_w = lidar_res['y'], lidar_res['x']
    lidar_bev = np.zeros((grid_h, grid_w), dtype=np.float32)
    x_px = (((pts[:, 0] / lidar_range) + 1) / 2 * (grid_w - 1)).astype(np.int32)
    y_px = (((pts[:, 1] / lidar_range) + 1) / 2 * (grid_h - 1)).astype(np.int32)
    mask = (x_px >= 0) & (x_px < grid_w) & (y_px >= 0) & (y_px < grid_h)
    lidar_bev[y_px[mask], x_px[mask]] = 1.0
    lidar = lidar_bev[np.newaxis, :, :]

    ego_state = np.array(obs["ego_state"], dtype=np.float32).flatten()
    risk_features = np.array(obs["risk_features"], dtype=np.float32).flatten()

    return {
        "camera": torch.tensor(camera, dtype=torch.float32, device=device).unsqueeze(0),
        "lidar": torch.tensor(lidar, dtype=torch.float32, device=device).unsqueeze(0),
        "ego_state": torch.tensor(ego_state, dtype=torch.float32, device=device).unsqueeze(0),
        "risk_features": torch.tensor(risk_features, dtype=torch.float32, device=device).unsqueeze(0),
    }

def _extract_camera_frame(obs: dict, config: dict) -> np.ndarray:
    """
    @brief Extract the raw camera frame as a uint8 HWC array for video writing.

    @param obs dict Raw observation dictionary from the CARLA environment.
    @param config dict Environment configuration specifying camera properties.
    @return np.ndarray Camera frame of shape (H, W, C) in uint8 format.
    """

    cam_cfg = config['sensors']['camera']
    cam_res = cam_cfg['train_resolution']
    channels = cam_cfg['channels']
    H, W = cam_res['y'], cam_res['x']
    frame = np.array(obs["camera"], dtype=np.float32).reshape(channels, H, W)
    return np.clip(np.transpose(frame, (1, 2, 0)), 0, 255).astype(np.uint8)

class PolicyEvaluator:
    """
    @class PolicyEvaluator
    @brief Evaluates a trained RL navigation policy over multiple episodes.

    @details
    Supports PPO, SAC, and CVaR-SAC. Loads the appropriate agent and
    checkpoint, runs N episodes, and reports the following metrics:
        - Collision rate
        - Near-miss rate (TTC < 2.0s)
        - Goal success rate
        - Mean waypoint completion
        - Mean total reward
        - Mean episode length

    Optionally records a demo video using OpenCV.

    Usage:
        evaluator = PolicyEvaluator(algo="ppo", obs_config=..., action_dim=3, device=device)
        evaluator.load_checkpoint(checkpoint_dir)
        results = evaluator.run(env, n_episodes=20, deterministic=True, record=True)
    """

    def __init__(
        self,
        algo: str,
        obs_config: dict,
        action_dim: int,
        device: torch.device,
    ):
        """
        @brief Initialise the evaluator and build the agent for the given algorithm.

        @param algo str Algorithm identifier, one of 'ppo', 'sac', or 'cvar_sac'.
        @param obs_config dict Observation encoder configuration dictionary.
        @param action_dim int Dimension of the continuous action space.
        @param device torch.device Torch device to run inference on.
        """

        if algo not in SUPPORTED_ALGOS:
            raise ValueError(f"algo must be one of {SUPPORTED_ALGOS}, got '{algo}'")

        self.algo = algo
        self.device = device
        self.agent = self._build_agent(algo, obs_config, action_dim, device)
        self.obs_normaliser = ObservationNormaliser(
            ego_state_dim =obs_config['ego_state_dim'],
            risk_feature_dim=obs_config['risk_feature_dim'],
        )

    def _build_agent(self, algo: str, obs_config: dict, action_dim: int, device: torch.device):
        """
        @brief Instantiate the correct agent class for the given algorithm.

        @param algo str Algorithm identifier.
        @param obs_config dict Observation encoder configuration dictionary.
        @param action_dim int Dimension of the continuous action space.
        @param device torch.device Torch device for the agent.
        @return Agent instance (PPOAgent, SACAgent, or CVaRSACAgent).
        """

        if algo == "ppo":
            from rl.algorithms.ppo.ppo_agent import PPOAgent
            return PPOAgent(obs_config, action_dim, device=device)
        elif algo == "sac":
            from rl.algorithms.sac.sac_agent import SACAgent
            return SACAgent(obs_config, action_dim, device=device)
        else: # cvar_sac
            from rl.algorithms.cvar_sac.cvar_sac_agent import CVaRSACAgent
            return CVaRSACAgent(obs_config, action_dim, device=device)

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """
        @brief Load the agent checkpoint from disc and set networks to eval mode.

        @param checkpoint_dir str Parent directory containing per-algorithm subdirectories.
        @return int Rollout count (PPO) or step count (SAC/CVaR-SAC), or 0 if no checkpoint found.
        """

        manager = CheckpointManager(parent_dir=checkpoint_dir, ws_dir=self.algo)
        count = manager.load(self.agent)

        self.agent.policy.encoder.eval()
        self.agent.policy.actor.eval()
        self.agent.policy.critic.eval()

        return count

    def _select_action(self, obs_tensor: dict, deterministic: bool) -> torch.Tensor:
        """
        @brief Select an action from the policy given a preprocessed observation.

        @details
        PPO does not expose a deterministic flag on its act() method, so the
        actor mean is used directly when deterministic=True. SAC and CVaR-SAC
        both support deterministic action selection natively via their act() method.

        @param obs_tensor dict Preprocessed, batched observation tensors.
        @param deterministic bool If True, return the mean action without sampling.
        @return torch.Tensor Selected action of shape [1, action_dim].
        """

        if self.algo == "ppo":
            if deterministic:
                with torch.no_grad():
                    latent = self.agent.policy._encode(obs_tensor)
                    mean, _ = self.agent.policy.actor(latent)
                return mean
            else:
                with torch.no_grad():
                    action, _, _ = self.agent.act(obs_tensor)
                return action
        else: # SAC and CVaR-SAC both support the deterministic flag
            with torch.no_grad():
                action, _ = self.agent.act(obs_tensor, deterministic=deterministic)
            return action

    def run(
        self,
        env,
        n_episodes: int = 20,
        deterministic: bool = False,
        record: bool = False,
        video_dir: str = None,
    ) -> dict:
        """
        @brief Run evaluation episodes and return aggregated metrics.

        @param env CarlaEnv Initialised CARLA environment instance.
        @param n_episodes int Number of episodes to evaluate over.
        @param deterministic bool Use the mean action if True, sample from the policy if False.
        @param record bool Save a demo video to disc (requires opencv-python).
        @param video_dir str Directory to write the video file. Required when record=True.
        @return dict Aggregated metrics with keys: collision_rate, near_miss_rate,
                     goal_rate, mean_completion, mean_reward, mean_episode_length.
        """

        video_writer = None

        if record:
            video_writer = self._init_video_writer(env.config, video_dir)

        collisions, near_misses, goals_reached = [], [], []
        wp_completions, total_rewards, ep_lengths = [], [], []

        for ep in range(n_episodes):
            obs = env.reset()
            ep_reward = np.zeros(3)
            steps = 0
            done = False

            while not done:
                self.obs_normaliser.update(obs)
                obs_tensor = _preprocess_obs(obs, env.config, self.device)
                obs_tensor = self.obs_normaliser.normalise(obs_tensor)

                action = self._select_action(obs_tensor, deterministic)
                action_np = action.detach().cpu().numpy().squeeze(0)

                if record and video_writer is not None:
                    import cv2
                    frame = _extract_camera_frame(obs, env.config)
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                next_obs, reward, done, info = env.step(action_np, log=False)
                env.render()

                ep_reward += reward
                steps += 1
                obs = next_obs

            collision = info.get('collision', False)
            goal_reached = info.get('goal_reached', False)
            near_miss = info.get('ttc_min', float('inf')) < 2.0
            wp_idx = info.get('wp_idx', 0)
            wp_total = max(info.get('wp_total', 1), 1)
            completion = wp_idx / wp_total

            collisions.append(collision)
            near_misses.append(near_miss)
            goals_reached.append(goal_reached)
            wp_completions.append(completion)
            total_rewards.append(ep_reward)
            ep_lengths.append(steps)

            status = "GOAL" if goal_reached else ("COLLISION" if collision else "TIMEOUT")
            Log.info(__file__,
                f"Ep {ep+1:>3}/{n_episodes} [{status:<9}] "
                f"steps: {steps:>4} | "
                f"completion: {completion:.0%} | "
                f"reward: {ep_reward.sum():.2f} "
                f"[nav {ep_reward[0]:.2f} | safety {ep_reward[1]:.2f} | risk {ep_reward[2]:.2f}]"
            )

        if video_writer is not None:
            video_writer.release()
            Log.info(__file__, "Video saved.")

        n = len(collisions)
        results = dict(
            collision_rate=sum(collisions) / n,
            near_miss_rate=sum(near_misses) / n,
            goal_rate=sum(goals_reached) / n,
            mean_completion=float(np.mean(wp_completions)),
            mean_reward=float(np.mean([r.sum() for r in total_rewards])),
            mean_episode_length =float(np.mean(ep_lengths)),
        )

        Log.info(__file__, (
            f"\n{'='*60}\n"
            f"  Evaluation Summary - {self.algo.upper()} ({n} episodes)\n"
            f"{'='*60}\n"
            f"  Collision rate   : {results['collision_rate']:.1%}\n"
            f"  Near-miss rate   : {results['near_miss_rate']:.1%}  (TTC < 2.0s)\n"
            f"  Goal success rate: {results['goal_rate']:.1%}\n"
            f"  Mean completion  : {results['mean_completion']:.1%}\n"
            f"  Mean total reward: {results['mean_reward']:.2f}\n"
            f"  Mean episode len : {results['mean_episode_length']:.1f} steps\n"
            f"{'='*60}"
        ))

        return results

    def _init_video_writer(self, config: dict, video_dir: str):
        """
        @brief Initialise an OpenCV VideoWriter for recording evaluation episodes.

        @param config dict Environment configuration specifying camera resolution.
        @param video_dir str Directory in which to save the output video file.
        @return cv2.VideoWriter instance, or None if initialisation fails.
        """

        try:
            import cv2
        except ImportError:
            Log.error(__file__, "opencv-python is required for video recording (pip install opencv-python)")
            return None

        if video_dir is None:
            Log.error(__file__, "video_dir must be provided when record=True")
            return None

        os.makedirs(video_dir, exist_ok=True)
        cam_res = config['sensors']['camera']['train_resolution']
        H, W = cam_res['y'], cam_res['x']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(video_dir, f"eval_{self.algo}_{timestamp}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
        Log.info(__file__, f"Recording video to: {path}")

        return writer