import torch
import numpy as np

from src.env.gym_carla_env import CarlaEnv
from src.rl.algorithms.ppo.ppo_agent import PPOAgent
from src.rl.algorithms.ppo.ppo_trainer import PPOTrainer
from src.rl.algorithms.ppo.rollout_buffer import RolloutBuffer


def preprocess_obs(obs):
    # ⚠️ TEMP: adapt based on the encoder input format
    return obs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CarlaEnv()

    action_dim = env.action_space.shape[0]

    obs_config = dict(
        camera_shape=(3, 84, 84),
        lidar_shape=(1, 64, 64),
        ego_state_dim=6,
        risk_feature_dim=3,
    )

    agent = PPOAgent(obs_config, action_dim, device)
    trainer = PPOTrainer(agent)

    buffer = RolloutBuffer(buffer_size=2048, obs_shape=None, action_dim=action_dim, device=device)

    obs = env.reset()

    for episode in range(1000):
        for step in range(2048):
            obs_tensor = preprocess_obs(obs)

            action, log_prob, value = agent.get_action(obs_tensor)

            action_np = action.detach().cpu().numpy()

            next_obs, reward, done, _ = env.step(action_np)

            buffer.store(obs_tensor, action, log_prob, reward, done, value)

            obs = next_obs

            if done:
                obs = env.reset()

        trainer.update(buffer)
        buffer.clear()

        print(f"Episode {episode} completed")

    env.close()

if __name__ == "__main__":
    main()