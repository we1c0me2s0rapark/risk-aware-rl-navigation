import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict

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
    from rl.algorithms.cvar_sac.cvar_sac_agent import CVaRSACAgent
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

def main():
    """
    @brief CVaR-SAC training loop for CARLA environment.

    @details
    Structurally identical to train_sac.py. Key differences:
        - Uses CVaRSACAgent with a distributional critic (n_quantiles quantiles).
        - Actor is optimised using CVaR_alpha(Z) instead of E[Q].
        - Logs an additional CVaR alpha reference point to TensorBoard.

    The distributional critic learns the full return distribution Z(s,a),
    and CVaR focuses the policy on reducing worst-case outcomes.
    """

    action_dim = 3
    step_count = 0
    total_reward = np.zeros(3)

    session = TrainingSession(algo="cvar_sac")

    seed = session.env.config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cvar_cfg = session.env.config['cvar_sac']
    total_steps = session.env.config['training']['total_steps']
    batch_size = cvar_cfg['batch_size']
    buffer_capacity = cvar_cfg['buffer_capacity']
    learning_starts = cvar_cfg['learning_starts']
    save_every = cvar_cfg['save_every']
    log_every = cvar_cfg['log_every']
    n_quantiles = cvar_cfg['n_quantiles']
    cvar_alpha = cvar_cfg['cvar_alpha']
    update_every = cvar_cfg.get('update_every', 1)

    agent = CVaRSACAgent(
        obs_config=session.obs_config,
        action_dim=action_dim,
        n_quantiles=n_quantiles,
        cvar_alpha=cvar_alpha,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        device=session.device,
    )
    step_count = session.load_checkpoint(agent)

    episode = 0
    episode_reward = np.zeros(3)
    episode_steps = 0
    obs = session.env.reset()

    # Timing accumulators: seconds spent in each section per log_every window
    timings = defaultdict(float)

    try:
        session.setup()
        Log.info(__file__, f"Starting CVaR-SAC training on {session.device}")
        Log.info(__file__, f"CVaR alpha: {cvar_alpha} | Quantiles: {n_quantiles}")
        Log.info(__file__, f"Warming up for {learning_starts} steps...")

        while step_count < total_steps:
            if is_q_pressed():
                raise RuntimeError("'q' pressed - Bye! 👋")

            t0 = time.perf_counter()
            session.obs_normaliser.update(obs)
            obs_tensor = session.preprocess(obs)
            obs_tensor = session.obs_normaliser.normalise(obs_tensor)
            timings['preprocess'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if step_count < learning_starts:
                action_np = session.env.action_space.sample()
                action = torch.tensor(action_np, dtype=torch.float32, device=session.device)
            else:
                action, _ = agent.act(obs_tensor)
                action_np = action.detach().cpu().numpy().squeeze(0)
            timings['act'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_obs, reward, done, info = session.env.step(action_np, log=False)
            session.env.render()
            timings['env_step'] += time.perf_counter() - t0

            total_reward += reward
            episode_reward += reward
            episode_steps += 1

            t0 = time.perf_counter()
            session.obs_normaliser.update(next_obs)
            next_obs_tensor = session.preprocess(next_obs)
            next_obs_tensor = session.obs_normaliser.normalise(next_obs_tensor)
            timings['preprocess'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            agent.store(
                {k: v.squeeze(0) for k, v in obs_tensor.items()},
                action.squeeze(0) if isinstance(action, torch.Tensor) else torch.tensor(action_np, device=session.device),
                reward,
                {k: v.squeeze(0) for k, v in next_obs_tensor.items()},
                done,
            )
            timings['store'] += time.perf_counter() - t0

            obs = next_obs
            step_count += 1

            if step_count >= learning_starts and step_count % update_every == 0:
                t0 = time.perf_counter()
                losses = agent.update()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings['update'] += time.perf_counter() - t0
                session.logger.log_sac_losses(step_count, losses)

                if step_count % log_every == 0:
                    session.logger.log_cvar_alpha(step_count, cvar_alpha)

            if done:
                session.log_episode(episode, episode_steps, episode_reward, info)
                episode += 1
                episode_reward = np.zeros(3)
                episode_steps = 0
                obs = session.env.reset()

            if step_count % log_every == 0 and step_count >= learning_starts:
                session.logger.log_buffer_size(step_count, len(agent.buffer))

                total_t = sum(timings.values()) or 1e-9
                Log.info(__file__,
                    f"[Step {step_count:07d}] "
                    f"Buffer: {len(agent.buffer)} | "
                    f"Alpha: {agent.trainer.alpha:.4f} | "
                    f"step={timings['env_step']*1000/log_every:.1f}ms "
                    f"pre={timings['preprocess']*1000/log_every:.1f}ms "
                    f"act={timings['act']*1000/log_every:.1f}ms "
                    f"store={timings['store']*1000/log_every:.1f}ms "
                    f"update={timings['update']*1000/log_every:.1f}ms "
                    f"[{', '.join(f'{k}:{v/total_t*100:.0f}%' for k, v in sorted(timings.items(), key=lambda x: -x[1]))}]"
                )
                timings = defaultdict(float)

            if step_count % save_every == 0:
                session.save_checkpoint(agent, step_count)
                Log.info(__file__, f"Checkpoint saved at step {step_count}")

    except Exception as e:
        Log.error(__file__, e)

    finally:
        avg = total_reward / max(step_count, 1)
        Log.info(__file__,
            f"\n🏳️  Training stopped at step {step_count}; "
            f"avg/step — max: {avg.max():.4f} med: {np.median(avg):.4f} min: {avg.min():.4f}\n"
            f"\t[ nav: {avg[0]:.4f}, safety: {avg[1]:.4f}, risk: {avg[2]:.4f} ]\n"
        )
        session.close()

if __name__ == "__main__":
    main()