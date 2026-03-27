import os
import sys
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
    from algorithms.ppo.ppo_policy import PPOPolicy
    from algorithms.ppo.ppo_trainer import PPOTrainer
    from algorithms.ppo.rollout_buffer import RolloutBuffer
except ImportError as e:
    Log.error(__file__, e)

def preprocess_obs(obs, config):
    """
    @brief Preprocesses raw observations from CARLA environment for PPO input.

    @param[in] obs Dictionary containing raw sensor and state observations.
    @param[in] config Configuration dictionary specifying sensor properties.

    @return Dictionary of batched torch tensors ready for policy input:
        - 'camera': [1, channels, height, width] tensor
        - 'lidar': [1, 1, height, width] tensor
        - 'ego_state': [1, ego_state_dim] tensor
        - 'risk_features': [1, risk_feature_dim] tensor
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Camera processing ---
        cam_cfg = config['sensors']['camera']
        cam_res = cam_cfg['train_resolution']
        channels = cam_cfg['channels']
        camera = np.array(obs["camera"], dtype=np.float32).reshape(channels, cam_res['y'], cam_res['x'])

        # --- LiDAR processing ---
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

        # --- Ego vehicle state and risk features ---
        ego_state = np.array(obs["ego_state"], dtype=np.float32).flatten()
        risk_features = np.array(obs["risk_features"], dtype=np.float32).flatten()

        # Returns batched tensors [1, ...] suitable for policy.act()
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
    @brief Main training loop for PPO policy in CARLA environment.

    @details
    - Initialises the CARLA environment, PPO policy, PPO trainer, and rollout buffer.
    - Collects rollouts for a fixed number of steps per episode.
    - Updates the PPO policy after each rollout.
    - Resets the environment when an episode terminates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialise CARLA Gym environment ---
    env = CarlaEnv()

    # --- Define action space dimension ---
    action_dim = 3 # [steer, throttle, brake]

    # --- Observation configuration for PPO ---
    obs_config = dict(
        camera_shape=(3, 84, 84),
        lidar_shape=(1, 64, 64),
        ego_state_dim=6,
        risk_feature_dim=55,
    )

    # --- Initialise PPO policy and trainer ---
    policy = PPOPolicy(obs_config, action_dim, device)
    trainer = PPOTrainer(policy)

    # --- Initialise rollout buffer ---
    buffer = RolloutBuffer(
        buffer_size=2048,
        obs_shape=None,
        action_dim=action_dim,
        device=device
    )

    # --- Reset environment for initial observation ---
    obs = env.reset()

    # --- Main training loop ---
    for episode in range(10):
        for step in range(20):
            # Preprocess observation → batched [1, ...] tensors for policy.act()
            obs_tensor = preprocess_obs(obs, env.config)

            # Select action, log probability, and value from policy
            action, log_prob, value = policy.act(obs_tensor)

            # Convert action to numpy array for Gym environment
            # squeeze (1, 3) → (3,) so _apply_control can unpack [steer, throttle, brake]
            action_np = action.detach().cpu().numpy().squeeze(0)

            # Step environment
            next_obs, reward, done, _ = env.step(action_np)

            # Store unbatched transitions in rollout buffer
            # squeeze batch dim [1, ...] → [...] to prevent extra leading dimension
            buffer.store(
                {k: v.squeeze(0) for k, v in obs_tensor.items()},
                action.squeeze(0),
                log_prob.squeeze(0),
                reward,
                done,
                value.squeeze(0)
            )

            obs = next_obs

            # Reset environment if episode terminates
            if done:
                obs = env.reset()

        # --- Update PPO policy after collecting rollout ---
        # Preprocess last obs so PPO trainer can compute the bootstrap value
        last_obs_tensor = preprocess_obs(obs, env.config)
        trainer.update(buffer, last_obs=last_obs_tensor, done=False)

        buffer.clear()
        Log.check(__file__, f"Episode {episode} completed")

    # --- Close environment and release resources ---
    env.close()

if __name__ == "__main__":
    main()