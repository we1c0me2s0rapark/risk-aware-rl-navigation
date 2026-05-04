import os
import sys
import tty
import termios
import argparse
import torch

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."
))

try:
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))

    from env.gym_carla_env import CarlaEnv
    from managers.utils.logger import Log
    from rl.common.evaluator import PolicyEvaluator
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")
    sys.exit(1)

def main():
    """
    @brief Entry point for evaluating a trained RL navigation policy.

    @details
    Parses command-line arguments, initialises the CARLA environment and
    PolicyEvaluator, loads the appropriate checkpoint, then runs evaluation
    episodes and prints a summary of the results.

    Supported algorithms: ppo, sac, cvar_sac.

    Example usage:
        python evaluate_policy.py --algo ppo --episodes 20
        python evaluate_policy.py --algo cvar_sac --deterministic --record
    """

    parser = argparse.ArgumentParser(description="Evaluate a trained RL navigation policy")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "cvar_sac"],
        help="Algorithm whose checkpoint to evaluate"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the mean action instead of sampling from the policy"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record a demo video to disc (requires opencv-python)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    env = CarlaEnv(run_tag=f"{args.algo}_eval")

    try:
        tty.setcbreak(fd)
        cam_config = env.config['sensors']['camera']
        cam_res = cam_config['train_resolution']
        lidar_config = env.config['sensors']['lidar']
        lidar_res = lidar_config['train_resolution']
        render_root = env.config['render']['root']

        obs_config = dict(
            camera_shape=(cam_config['channels'], cam_res['y'], cam_res['x']),
            lidar_shape=(lidar_config['channels'], lidar_res['y'], lidar_res['x']),
            ego_state_dim=6 + env.config['risk']['waypoints_ahead'] * 3 + 8,
            latent_dim=256,
            hidden_dim=128,
            n_reward_components=3,
            risk_feature_dim=env.risk_module.feature_dim,
        )
        # Note: obs_config is kept here since evaluate_policy does not use TrainingSession.

        evaluator = PolicyEvaluator(
            algo=args.algo,
            obs_config=obs_config,
            action_dim=2,
            device=device,
        )

        checkpoint_dir = os.path.abspath(os.path.join(
            ws_root_path, render_root, "checkpoints"
        ))
        count = evaluator.load_checkpoint(checkpoint_dir)
        if count == 0:
            Log.info(__file__, "Warning: no checkpoint found - evaluating an untrained policy.")

        video_dir = os.path.abspath(
            os.path.join(ws_root_path, render_root, "videos", args.algo)
        ) if args.record else None

        evaluator.run(
            env=env,
            n_episodes=args.episodes,
            deterministic=args.deterministic,
            record=args.record,
            video_dir=video_dir,
        )

    except Exception as e:
        Log.error(__file__, e)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
