import os
import sys
import tty
import termios
import torch
import numpy as np

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    sys.path.append(os.path.abspath(os.path.join(ws_root_path)))
    sys.path.append(os.path.abspath(os.path.join(ws_root_path, "src")))

    from env.gym_carla_env import CarlaEnv
    from managers.utils.logger import Log
    from carla_client.utilities import is_q_pressed
    from rl.common.checkpoint import CheckpointManager
    from rl.common.normalisation import ObservationNormaliser
    from rl.logger.logger import TrainingLogger
    from rl.algorithms.sac.sac_agent import SACAgent
except ImportError as e:
    Log.error(__file__, e)

def preprocess_obs(obs, config, device):
    """
    @brief Preprocess raw CARLA environment observations for SAC input.

    @param obs dict Raw observation from CarlaEnv.
    @param config dict Environment configuration.
    @param device torch.device Target device for tensors.
    @return dict Batched tensors [1, ...] on the correct device.
    """
    try:
        # --- Camera ---
        cam_cfg = config['sensors']['camera']
        cam_res = cam_cfg['train_resolution']
        channels = cam_cfg['channels']
        camera = np.array(obs["camera"], dtype=np.float32).reshape(
            channels, cam_res['y'], cam_res['x']
        )
        camera = camera / 255.0  # normalise to [0, 1]

        # --- LiDAR BEV projection ---
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

        # --- Ego state & risk features ---
        ego_state = np.array(obs["ego_state"], dtype=np.float32).flatten()
        risk_features = np.array(obs["risk_features"], dtype=np.float32).flatten()

        return {
            "camera": torch.tensor(camera, dtype=torch.float32, device=device).unsqueeze(0),
            "lidar": torch.tensor(lidar, dtype=torch.float32, device=device).unsqueeze(0),
            "ego_state": torch.tensor(ego_state, dtype=torch.float32, device=device).unsqueeze(0),
            "risk_features": torch.tensor(risk_features, dtype=torch.float32, device=device).unsqueeze(0),
        }
    except Exception as e:
        Log.error(__file__, e)
        return None

def main():
    """
    @brief Main SAC training loop for CARLA environment.

    @details
    SAC is off-policy: transitions are stored in a replay buffer and
    sampled randomly for updates. Unlike PPO, updates happen every step
    once the buffer has enough transitions (>= learning_starts).

    Key differences from train_ppo.py:
        - ReplayBuffer instead of RolloutBuffer (no clearing after episode)
        - Update every step, not every N steps
        - Deterministic evaluation via agent.act(deterministic=True)
        - Soft target network updates inside SACTrainer
    """
    # Enable non-blocking keyboard input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Training state
    step_count = 0
    total_reward = np.zeros(3)
    done = False

    # Hyperparameters
    action_dim = 3              # [steer, throttle, brake]
    learning_starts = 1_000     # warm-up steps with random actions before training begins
    save_every = 1_000          # steps between checkpoint saves
    log_every = 1_000           # steps between periodic log outputs
    batch_size = 256            # minibatch size for SAC updates
    buffer_capacity = 500_000   # maximum transitions stored in replay buffer
    total_steps = 1_000_000

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialise environment ---
    env = CarlaEnv(run_tag="sac")

    # --- Initialise logger to None for safe cleanup in finally block ---
    logger = None

    try:
        # Enable non-blocking keyboard input
        tty.setcbreak(fd)

        # --- Camera and LiDAR configuration from environment ---
        cam_config = env.config['sensors']['camera']
        cam_res = cam_config['train_resolution']
        cam_channels = cam_config['channels']

        lidar_config = env.config['sensors']['lidar']
        lidar_res = lidar_config['train_resolution']
        lidar_channels = lidar_config['channels']

        # Root directory for saving outputs
        render_root = env.config['render']['root']

        # --- Observation configuration ---
        obs_config = dict(
            camera_shape=(cam_channels, cam_res['y'], cam_res['x']),
            lidar_shape=(lidar_channels, lidar_res['y'], lidar_res['x']),
            ego_state_dim=16, # 6 base + 10 waypoint values
            latent_dim=256,
            hidden_dim=128,
            n_reward_components=3,
            risk_feature_dim=env.risk_module.feature_dim,
        )

        # --- Initialise SAC agent ---
        agent = SACAgent(obs_config, action_dim, buffer_capacity=buffer_capacity, min_samples=batch_size, device=device)

        # --- Observation normaliser ---
        obs_normaliser = ObservationNormaliser(
            ego_state_dim=obs_config['ego_state_dim'],
            risk_feature_dim=obs_config['risk_feature_dim'],
        )

        checkpoint_dir = os.path.abspath(os.path.join(
            ws_root_path, render_root, "checkpoints"
        ))
        checkpoint_manager = CheckpointManager(parent_dir=checkpoint_dir, ws_dir="sac")
        step_count = checkpoint_manager.load(agent)  # returns 0 if no checkpoint
 
        # --- TensorBoard logger ---
        log_dir = os.path.abspath(os.path.join(
            ws_root_path, render_root, "runs", "sac"
        ))
        logger = TrainingLogger(log_dir=log_dir, algorithm="sac")

        # --- Main training loop ---
        obs = env.reset()

        # Episode-level tracking for logging
        episode = 0
        episode_reward = np.zeros(3)
        episode_steps = 0
        episode_collisions = 0

        # Log training start
        Log.info(__file__, f"Starting SAC training on {device}")
        Log.info(__file__, f"Warming up for {learning_starts} steps...")

        while step_count < total_steps:
            # Check for user quit input
            if is_q_pressed():
                raise RuntimeError("\n'q' pressed - Bye! 👋\n")
 
            # Update normaliser stats from raw obs, then preprocess and normalise
            obs_normaliser.update(obs)
            obs_tensor = preprocess_obs(obs, env.config, device)
            obs_tensor = obs_normaliser.normalise(obs_tensor)

            # --- Action selection ---
            if step_count < learning_starts:
                # Random exploration during warm-up
                action_np = env.action_space.sample()
                action = torch.tensor(action_np, dtype=torch.float32, device=device)
            else:
                action, _ = agent.act(obs_tensor)
                action_np = action.detach().cpu().numpy().squeeze(0)

            # --- Environment step ---
            next_obs, reward, done, info = env.step(action_np, log=False)
            env.render()

            # Accumulate rewards for logging
            total_reward += reward
            episode_reward += reward
            episode_steps += 1
            episode_collisions += int(info.get('collision', False))
 
            # Normalise next_obs before storing in replay buffer
            obs_normaliser.update(next_obs)
            next_obs_tensor = preprocess_obs(next_obs, env.config, device)
            next_obs_tensor = obs_normaliser.normalise(next_obs_tensor)

            # --- Store transition (unbatched) ---
            next_obs_tensor = preprocess_obs(next_obs, env.config, device)
            agent.store(
                {k: v.squeeze(0) for k, v in obs_tensor.items()},
                action.squeeze(0) if isinstance(action, torch.Tensor) else torch.tensor(action_np, device=device),
                reward,
                {k: v.squeeze(0) for k, v in next_obs_tensor.items()},
                done
            )

            obs = next_obs
            step_count += 1

            # --- SAC update ---
            if step_count >= learning_starts:
                losses = agent.update()
                logger.log_sac_losses(step_count, losses)

            # --- Episode reset ---
            if done:
                goal_reached = info.get('goal_reached', False)
                collision = info.get('collision', False)
                near_miss = info.get('ttc_min', float('inf')) < 2.0

                log_text = ""
                if goal_reached: log_text += f"🎯 Goal reached after {episode_steps} steps; "
                elif collision:   log_text += f"💥 Collision after {episode_steps} steps; "
                else:             log_text += f"⏱️  Timeout after {episode_steps} steps; "

                log_text += f"waypoints: {info['wp_idx']}/{max(info['wp_total'], 1)}; "
                log_text += f"total: {episode_reward.sum():.2f} "
                log_text += f"[ nav: {episode_reward[0]:.2f}, safety: {episode_reward[1]:.2f}, risk: {episode_reward[2]:.2f} ]"

                Log.info(__file__, log_text)
 
                # TensorBoard episode logging
                logger.log_episode(episode, episode_reward, info)
                logger.log_near_miss_rate(episode, near_miss)

                # Reset episode tracking
                obs = env.reset()
                episode += 1
                episode_reward = np.zeros(3)
                episode_steps = 0
                episode_collisions = 0

            # --- Periodic logging ---
            if step_count % log_every == 0 and step_count >= learning_starts:
                logger.log_buffer_size(step_count, len(agent.buffer))
                Log.info(__file__,
                    f"[Step {step_count:07d}] "
                    f"Buffer: {len(agent.buffer)} | "
                    f"Alpha: {agent.trainer.alpha:.4f}"
                )

            # --- Checkpointing ---
            if step_count % save_every == 0:
                checkpoint_manager.save(agent, step_count)
                Log.info(__file__, f"Checkpoint saved at step {step_count}")

    except Exception as e:
        Log.error(__file__, e)

    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        Log.info(__file__,
            f"\n🏳️  Training stopped at step {step_count}; rewards - "
            f"total: {total_reward.sum():.2f}\n"
            f"\t- nav:    {total_reward[0]:.2f}\n"
            f"\t- safety: {total_reward[1]:.2f}\n"
            f"\t- risk:   {total_reward[2]:.2f}\n"
        )
 
        # --- Close logger safely ---
        if logger is not None:
            logger.close()

        # --- Close environment safely ---
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
