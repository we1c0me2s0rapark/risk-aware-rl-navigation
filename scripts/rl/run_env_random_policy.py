import os
import sys
import time

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src"
    )))

    from env import CarlaEnv
    from managers.utils.logger import Log
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")


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
    total_baseline = 0.0
    step_count = 0

    frame_idx = 0

    try:
        # --- Create CARLA environment ---
        env = CarlaEnv()

        # --- Reset environment ---
        obs = env.reset()

        done = False
        while not done:
            # Random action test: [steer, throttle, brake]
            action = env.action_space.sample()

            # Step simulation
            obs, reward, done, info = env.step(action)

            env.render()

            total_reward += reward
            total_baseline += info.get('baseline_reward', 0.0)

            step_count += 1

            Log.info(__file__, f"""RISK
    Step {step_count:03d} | Done: {done} | Min TTC: {info.get('ttc_min', float('inf')):.2f}s""")
            Log.info(__file__, f"""REWARD COMPARISON
    Baseline: {info['baseline_reward']:.2f} | Risk-aware: {info['risk_reward']:.2f} | Collision: {info['collision']}""")

            # Slow down for human observation
            time.sleep(0.05)

    except Exception as e:
        Log.error(__file__, e)

    finally:
        Log.info(__file__, 
            f"🏳️  Episode finished after {step_count} steps; "
            f"baseline reward: {total_baseline:.2f}, "
            f"total reward: {total_reward:.2f}"
        )

        # Ensure environment and sensors are cleaned up
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()