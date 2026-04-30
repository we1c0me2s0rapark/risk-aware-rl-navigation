import os
import sys
import torch
import numpy as np

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
))

try:
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from managers.utils.logger import Log
    from carla_client.utilities import is_q_pressed
    from rl.common.training_session import TrainingSession
    from rl.algorithms.ppo.ppo_agent import PPOAgent
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")


def main():
    """
    @brief PPO training loop for CARLA environment using fixed-size rollouts.

    @details
    Collects a fixed number of steps per rollout (rollout_size from config),
    resets the environment when episodes terminate mid-rollout, and updates
    the PPO policy after each complete rollout. Training stops when
    total_steps // rollout_size rollouts have been completed.
    """

    action_dim = 3
    rollout_count = 0
    rollout_step = 0

    session = TrainingSession(algo="ppo")

    seed = session.env.config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rollout_size = session.env.config['ppo']['rollout_size']
    max_rollouts = session.env.config['training']['total_steps'] // rollout_size

    agent = PPOAgent(session.obs_config, action_dim, device=session.device)
    rollout_count = session.load_checkpoint(agent)

    episode = 0
    step_count = 0
    episode_reward = np.zeros(3)
    episode_baseline = np.zeros(3)
    total_reward = np.zeros(3)
    done = False
    obs = session.env.reset()

    try:
        session.setup()
        Log.info(__file__, f"Starting PPO training on {session.device} | max rollouts: {max_rollouts}")

        while True:
            if is_q_pressed():
                raise RuntimeError("'q' pressed - Bye! 👋")

            session.obs_normaliser.update(obs)
            obs_tensor = session.preprocess(obs)
            obs_tensor = session.obs_normaliser.normalise(obs_tensor)

            with torch.no_grad():
                action, log_prob, value = agent.act(obs_tensor)

            action_np = action.detach().cpu().numpy().squeeze(0)
            next_obs, reward, done, info = session.env.step(action_np, log=False)
            session.env.render()

            episode_reward += reward
            episode_baseline += np.array(info.get('baseline_reward', [0.0, 0.0, 0.0]))
            total_reward += reward
            step_count += 1
            rollout_step += 1

            agent.store(
                obs={k: v.squeeze(0) for k, v in obs_tensor.items()},
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                rewards=torch.tensor(reward, dtype=torch.float32),
                done=done,
                value=value.squeeze(0),
            )

            obs = next_obs

            if done:
                session.log_episode(episode, step_count, episode_reward, info, episode_baseline)
                episode += 1
                step_count = 0
                episode_reward = np.zeros(3)
                episode_baseline = np.zeros(3)
                obs = session.env.reset()

            if rollout_step >= rollout_size:
                Log.info(__file__,
                    f"🏳️  Rollout update after {rollout_step} steps; "
                    f"nav: {episode_reward[0]:.2f} | "
                    f"safety: {episode_reward[1]:.2f} | "
                    f"risk: {episode_reward[2]:.2f}"
                )
                last_obs_tensor = session.preprocess(obs)
                last_obs_tensor = session.obs_normaliser.normalise(last_obs_tensor)
                agent.update(last_obs=last_obs_tensor, done=done)
                rollout_count += 1
                rollout_step = 0
                session.save_checkpoint(agent, rollout_count)

                if rollout_count >= max_rollouts:
                    Log.info(__file__, f"Reached max_rollouts ({max_rollouts}). Stopping training.")
                    break

    except Exception as e:
        Log.error(__file__, e)

    finally:
        total_steps = rollout_count * rollout_size + rollout_step
        avg = total_reward / max(total_steps, 1)
        Log.info(__file__,
            f"\n🏳️  Training stopped at rollout {rollout_count} (max: {max_rollouts}), step {rollout_step}; "
            f"avg/step — max: {avg.max():.4f} med: {np.median(avg):.4f} min: {avg.min():.4f}\n"
            f"\t[ nav: {avg[0]:.4f}, safety: {avg[1]:.4f}, risk: {avg[2]:.4f} ]\n"
        )
        session.close()

if __name__ == "__main__":
    main()
