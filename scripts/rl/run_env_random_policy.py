import os
import sys
import gym
import time
import numpy as np
from datetime import datetime

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src"
    )))

    from env import CarlaEnv
except ImportError as e:
    print(f"[{__name__}] Error: {e}")


def main():
    """
    @brief Entry point for testing the CARLA Gym environment.

    Spawns the environment, runs a single episode using random
    actions, and prints step-wise information including reward
    and collision count.

    The environment is properly closed at the end to clean up
    all sensors and actors.
    """

    env = None
    total_reward = 0.0
    step_count = 0

    frame_idx = 0

    try:
        # --- Create CARLA environment ---
        env = CarlaEnv()

        # --- Reset environment ---
        obs = env.reset()
        print("Initial observation shape:", obs.shape)

        done = False
        while not done:
            # Random action test: [steer, throttle, brake]
            action = env.action_space.sample()

            # Step simulation
            obs, reward, done, info = env.step(action)

            env.render(save=False)

            total_reward += reward
            step_count += 1

            print(
                f"Step {step_count:03d} | Reward: {reward:.2f} | Done: {done} | "
                f"Collision count: {info['collision']}"
            )

            # Slow down for human observation
            time.sleep(0.05)

    except Exception as e:
        print(f"[{__name__}] Error: {e}")

    finally:
        print(
            f"Episode finished after {step_count} steps, "
            f"total reward: {total_reward:.2f}"
        )

        # Ensure environment and sensors are cleaned up
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()