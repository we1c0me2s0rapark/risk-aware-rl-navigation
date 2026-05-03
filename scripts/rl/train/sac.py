import os
import sys
import time
import torch
import torch.nn.functional as F
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
    from rl.algorithms.sac.sac_agent import SACAgent
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

def _encode_controller_action(steer: float, throttle: float, brake: float) -> np.ndarray:
    """Encode controller outputs to 2-D combined action space [steer, combined_tb].

    combined_tb = throttle - brake ∈ [-1, 1]: positive = throttle, negative = brake.
    Normal driving (throttle≈0.3, brake=0) maps to combined≈0.3 — well inside tanh
    interior, avoiding the saturation region that SAC entropy fights against.
    """
    s        = float(np.clip(steer,            -1.0, 1.0))
    combined = float(np.clip(throttle - brake, -1.0, 1.0))
    return np.array([s, combined], dtype=np.float32)

def _run_bc_pretraining(agent, bc_steps: int, bc_lr: float, batch_size: int, log_file: str):
    """Pretrain actor + encoder via behaviour cloning on buffered controller demos."""
    if len(agent.buffer) < batch_size:
        Log.info(log_file, f"Buffer too small for BC ({len(agent.buffer)} < {batch_size}). Skipping.")
        return
    bc_params = list(agent.policy.encoder.parameters()) + list(agent.policy.actor.parameters())
    bc_opt = torch.optim.Adam(bc_params, lr=float(bc_lr))

    Log.info(log_file, f"Starting BC pre-training for {bc_steps} steps...")
    for i in range(bc_steps):
        obs_b, act_b, _, _, _ = agent.buffer.sample(batch_size)
        latent_b = agent.policy._encode(obs_b)
        mean_b, _ = agent.policy.actor(latent_b)   # pre-squash mean
        loss = F.mse_loss(torch.tanh(mean_b), act_b)
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()
        if (i + 1) % 500 == 0:
            Log.info(log_file, f"BC [{i+1}/{bc_steps}] loss={loss.item():.4f}")

    Log.info(log_file, "BC pre-training complete — switching to SAC updates.")

def main():
    """
    @brief SAC training loop for CARLA environment.

    @details
    SAC is off-policy: transitions are stored in a replay buffer and
    sampled randomly for updates. Updates happen every update_every steps
    once the buffer has enough transitions (>= learning_starts).

    Warmup phase runs the pure-pursuit waypoint controller as an expert to
    populate the replay buffer with quality demonstrations. A short BC
    pre-training phase then initialises the actor before SAC updates begin.
    """

    action_dim = 2
    step_count = 0
    total_reward = np.zeros(3)

    session = TrainingSession(algo="sac")

    seed = session.env.config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sac_cfg = session.env.config['sac']
    total_steps = session.env.config['training']['total_steps']
    batch_size = sac_cfg['batch_size']
    buffer_capacity = sac_cfg['buffer_capacity']
    learning_starts = sac_cfg['learning_starts']
    bc_steps = sac_cfg.get('bc_steps', 2000)
    bc_lr = sac_cfg.get('bc_lr', 1e-4)
    save_every = sac_cfg['save_every']
    log_every = sac_cfg['log_every']
    update_every = sac_cfg.get('update_every', 1)

    agent = SACAgent(
        session.obs_config, action_dim,
        buffer_capacity=buffer_capacity,
        min_samples=batch_size,
        device=session.device,
    )
    step_count = session.load_checkpoint(agent)

    episode = 0
    episode_reward = np.zeros(3)
    episode_steps = 0
    obs = session.env.reset()

    timings = defaultdict(float)
    bc_done = step_count >= learning_starts  # skip BC when resuming past warmup

    try:
        session.setup()
        Log.info(__file__, f"Starting SAC training on {session.device}")
        Log.info(__file__, f"Collecting {learning_starts} controller demos, then BC for {bc_steps} steps...")

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
                # Run waypoint controller as expert; encode to tanh-space for buffer
                ctrl = session.env._compute_controller_action()
                action_np = _encode_controller_action(*ctrl)
                action = torch.tensor(action_np, dtype=torch.float32, device=session.device)
            else:
                if not bc_done:
                    _run_bc_pretraining(agent, bc_steps, bc_lr, batch_size, __file__)
                    bc_done = True
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

            if step_count >= learning_starts and bc_done and step_count % update_every == 0:
                t0 = time.perf_counter()
                losses = agent.update()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings['update'] += time.perf_counter() - t0
                if losses:
                    session.logger.log_sac_losses(step_count, losses)

            if done:
                session.log_episode(episode, episode_steps, episode_reward, info)
                episode += 1
                episode_reward = np.zeros(3)
                episode_steps = 0
                obs = session.env.reset()

            if step_count % log_every == 0 and step_count >= learning_starts and bc_done:
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