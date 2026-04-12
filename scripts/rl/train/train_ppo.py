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
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path
    )))
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from env.gym_carla_env import CarlaEnv
    from managers.utils.logger import Log
    from carla_client.utilities import is_q_pressed
    from rl.common.checkpoint import CheckpointManager
    from rl.common.normalisation import ObservationNormaliser
    from rl.logger.logger import TrainingLogger
    from rl.algorithms.ppo.ppo_agent import PPOAgent
except ImportError as e:
    Log.error(__file__, e)


def preprocess_obs(obs, config, device):
    """
    @brief Preprocess raw CARLA environment observations for PPO input.

    @param[in] obs Dictionary of raw sensor and state observations.
    @param[in] config Configuration dictionary specifying sensor properties.
    @param[in] device Torch device for tensors.

    @return Dictionary of batched torch tensors suitable for policy input:
        - 'camera': [1, channels, height, width]
        - 'lidar': [1, 1, height, width]
        - 'ego_state': [1, ego_state_dim]
        - 'risk_features': [1, risk_feature_dim]
    """
    try:
        # --- Camera ---
        cam_cfg = config['sensors']['camera']
        cam_res = cam_cfg['train_resolution']
        channels = cam_cfg['channels']
        camera = np.array(obs["camera"], dtype=np.float32).reshape(channels, cam_res['y'], cam_res['x'])

        # --- LiDAR ---
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

        # --- Convert to batched torch tensors ---
        processed = {
            "camera": torch.tensor(camera, dtype=torch.float32, device=device).unsqueeze(0),
            "lidar": torch.tensor(lidar, dtype=torch.float32, device=device).unsqueeze(0),
            "ego_state": torch.tensor(ego_state, dtype=torch.float32, device=device).unsqueeze(0),
            "risk_features": torch.tensor(risk_features, dtype=torch.float32, device=device).unsqueeze(0),
        }
        return processed
    except Exception as e:
        Log.error(__file__, e)
        return None


def main():
    """
    @brief PPO training loop for CARLA environment using fixed-size rollouts.

    @details
    - Collects a fixed number of steps per rollout (e.g. 2048).
    - Resets the environment when episodes terminate mid-rollout.
    - Updates the PPO policy after each rollout using the last observation
      and the correct 'done' flag for bootstrapping.
    """
    # Enable non-blocking keyboard input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    # Tracking variables
    step_count = 0      # steps within current episode (resets on done)
    rollout_step = 0    # steps within current rollout (resets after update)
    rollout_count = 0   # number of completed rollouts (for checkpointing)
    episode = 0         # total episode count

    # Cumulative rewards for logging
    total_reward = np.zeros(3)
    total_baseline = np.zeros(3)
    done = False

    # PPO hyperparameters
    action_dim = 3 # [steer, throttle, brake]
    rollout_size = 2048 # fixed rollout size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialise environment ---
    env = CarlaEnv(run_tag="ppo")

    # --- Initialise logger to None for safe cleanup in finally block ---
    logger = None

    try:
        # Enable non-blocking keyboard input
        tty.setcbreak(fd)

        # Camera and LiDAR configuration from environment config
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

        # --- Initialise PPO agent ---
        agent = PPOAgent(obs_config, action_dim, device=device)
 
        # --- Observation normaliser ---
        obs_normaliser = ObservationNormaliser(
            ego_state_dim=obs_config['ego_state_dim'],
            risk_feature_dim=obs_config['risk_feature_dim'],
        )

        # --- Initialise checkpoint manager ---
        checkpoint_dir = os.path.abspath(os.path.join(
            ws_root_path, render_root, "checkpoints"
        ))
        checkpoint_manager = CheckpointManager(parent_dir=checkpoint_dir, ws_dir="ppo")
        rollout_count = checkpoint_manager.load(agent) # returns 0 if no checkpoint

        # --- TensorBoard logger ---
        log_dir = os.path.abspath(os.path.join(
            ws_root_path, render_root, "runs", "ppo"
        ))
        logger = TrainingLogger(log_dir=log_dir, algorithm="ppo")

        # --- Main training loop ---
        obs = env.reset()

        while True:
            # Check for user quit input
            if is_q_pressed():
                raise RuntimeError("\n'q' pressed - Bye! 👋\n")
 
            # Update normaliser stats from raw obs, then preprocess and normalise
            obs_normaliser.update(obs)
            obs_tensor = preprocess_obs(obs, env.config, device)
            obs_tensor = obs_normaliser.normalise(obs_tensor)

            # Select action from policy
            with torch.no_grad():
                action, log_prob, value = agent.act(obs_tensor)

            # Convert to numpy for Gym
            action_np = action.detach().cpu().numpy().squeeze(0)

            # Step environment
            next_obs, reward, done, info = env.step(action_np, log=False)
            env.render()

            # Accumulate rewards for logging
            total_reward += reward
            total_baseline += np.array(info.get('baseline_reward', [0.0, 0.0, 0.0]))

    #         Log.info(__file__, f"""RISK
    # Step {step_count:03d} | Done: {done} | Min TTC: {info.get('ttc_min', float('inf')):.2f}s""")
    #         Log.info(__file__, f"""REWARD COMPARISON
    # Baseline: {info['baseline_reward']:.2f} | Risk-aware: {info['risk_reward']:.2f} | Collision: {info['collision']}""")

            # Store in agent buffer
            agent.store(
                obs={k: v.squeeze(0) for k, v in obs_tensor.items()},
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                rewards=torch.tensor(reward, dtype=torch.float32),
                done=done,
                value=value.squeeze(0), # now [n_objectives]
            )

            # Prepare for next step
            obs = next_obs

            step_count += 1
            rollout_step += 1

            # Reset environment if episode terminates mid-rollout
            if done:
                goal_reached = info.get('goal_reached', False)
                collision = info.get('collision', False)
                near_miss = info.get('ttc_min', float('inf')) < 2.0

                log_text = ""

                if goal_reached: log_text += f"🎯 Goal reached after {step_count} steps; "
                elif collision: log_text += f"💥 Collision after {step_count} steps; "
                else: log_text += f"⏱️  Timeout after {step_count} steps; "

                log_text += f"completion: {info['wp_idx']}/{max(info['wp_total'], 1)}; "
                log_text += f"total: {total_reward.sum():.2f} "
                log_text += f"[ "
                log_text += f"nav: {total_reward[0]:.2f}, "
                log_text += f"safety: {total_reward[1]:.2f}, "
                log_text += f"risk: {total_reward[2]:.2f}"
                log_text += f" ]"

                Log.info(__file__, log_text)
 
                # TensorBoard logging
                logger.log_episode(episode, total_reward, info, total_baseline)
                logger.log_near_miss_rate(episode, near_miss)

                # Reset episode tracking
                episode += 1
                total_reward = np.zeros(3)      # reset per episode
                total_baseline = np.zeros(3)    # reset per episode
                step_count = 0
                obs = env.reset()

            # --- Update PPO policy after fixed rollout ---
            if rollout_step >= rollout_size:
                Log.info(__file__, 
                    f"🏳️  Rollout update after {rollout_step} steps; "
                    f"nav: {total_reward[0]:.2f} | "
                    f"safety: {total_reward[1]:.2f} | "
                    f"risk: {total_reward[2]:.2f}"
                )

                # Use last observation and done flag for bootstrapping value estimates
                last_obs_tensor = preprocess_obs(obs, env.config, device)
                last_obs_tensor = obs_normaliser.normalise(last_obs_tensor)
                final_done = done # last step's done flag

                # PPO update step
                agent.update(last_obs=last_obs_tensor, done=final_done)
                rollout_count += 1
                rollout_step = 0
                checkpoint_manager.save(agent, rollout_count)

    except Exception as e:
        Log.error(__file__, e)

    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        Log.info(__file__, 
            f"\n🏳️  Training stopped at rollout {rollout_count}, step {rollout_step}; rewards - "
            f"baseline: {total_baseline.sum():.2f}\n"
            f"total: {total_reward.sum():.2f}\n"
            f"\t- nav {total_reward[0]:.2f}\n"
            f"\t- safety {total_reward[1]:.2f}\n"
            f"\t- risk {total_reward[2]:.2f}\n"
        )

        # --- Close logger safely ---
        if logger is not None:
            logger.close()
        
        # --- Close environment safely ---
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()