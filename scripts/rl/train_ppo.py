import os
import sys
import tty
import termios
import torch
import numpy as np

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src"
    )))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "env"
    )))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "rl"
    )))

    from gym_carla_env import CarlaEnv
    from managers.utils.logger import Log
    from carla_client.utilities import is_q_pressed
    from algorithms.ppo.ppo_agent import PPOAgent
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

    # Preserve terminal configuration for later restoration
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    step_count = 0
    total_reward = 0.0
    total_baseline = 0.0
    done = False

    action_dim = 3 # [steer, throttle, brake]
    rollout_size = 2048 # fixed rollout size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialise environment ---
    env = CarlaEnv()

    try:
        # Enable non-blocking keyboard input
        tty.setcbreak(fd)

        cam_config = env.config['sensors']['camera']
        cam_res = cam_config['train_resolution']
        cam_channels = cam_config['channels']

        lidar_config = env.config['sensors']['lidar']
        lidar_res = lidar_config['train_resolution']
        lidar_channels = lidar_config['channels']

        # --- Observation configuration ---
        obs_config = dict(
            camera_shape=(cam_channels, cam_res['y'], cam_res['x']),
            lidar_shape=(lidar_channels, lidar_res['y'], lidar_res['x']),
            ego_state_dim=6,
            risk_feature_dim=env.risk_module.feature_dim,
        )

        # --- Initialise PPO agent ---
        agent = PPOAgent(obs_config, action_dim, device=device)

        obs = env.reset()

        while True:
            if is_q_pressed():
                raise RuntimeError("\n'q' pressed - Bye! 👋\n")

            obs_tensor = preprocess_obs(obs, env.config, device)

            # Select action from policy
            with torch.no_grad():
                action, log_prob, value = agent.act(obs_tensor)

            # Convert to numpy for Gym
            action_np = action.detach().cpu().numpy().squeeze(0)

            # Step environment
            next_obs, reward, done, info = env.step(action_np, log=False)

            env.render(save=False)

            total_reward += reward
            total_baseline += info.get('baseline_reward', 0.0)

    #         Log.info(__file__, f"""RISK
    # Step {step_count:03d} | Done: {done} | Min TTC: {info.get('ttc_min', float('inf')):.2f}s""")
    #         Log.info(__file__, f"""REWARD COMPARISON
    # Baseline: {info['baseline_reward']:.2f} | Risk-aware: {info['risk_reward']:.2f} | Collision: {info['collision']}""")


            # Store in agent buffer
            agent.store(
                {k: v.squeeze(0) for k, v in obs_tensor.items()},
                action.squeeze(0),
                log_prob.squeeze(0),
                reward,
                done,
                value.squeeze(0)
            )

            obs = next_obs
            step_count += 1

            # Reset environment if episode terminates mid-rollout
            if done:
                Log.info(__file__, 
                    f"🏳️  Rollout update completed after {step_count} steps; "
                    f"baseline reward: {total_baseline:.2f}, "
                    f"total reward: {total_reward:.2f}"
                )
                obs = env.reset()

            # --- Update PPO policy after fixed rollout ---
            if step_count >= rollout_size:
                Log.info(__file__, 
                    f"🏳️  Rollout update completed after {step_count} steps; "
                    f"baseline reward: {total_baseline:.2f}, "
                    f"total reward: {total_reward:.2f}"
                )
                last_obs_tensor = preprocess_obs(obs, env.config, device)
                final_done = done  # last step's done flag
                agent.update(last_obs=last_obs_tensor, done=final_done)
                step_count = 0

    except Exception as e:
        Log.error(__file__, e)

    finally:
        # Restore terminal configuration
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        Log.info(__file__, 
            f"🏳️  Episode finished after {step_count} steps; "
            f"baseline reward: {total_baseline:.2f}, "
            f"total reward: {total_reward:.2f}"
        )
        
        # --- Close environment safely ---
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()