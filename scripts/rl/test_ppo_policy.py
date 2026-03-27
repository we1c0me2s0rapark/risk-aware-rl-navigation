import sys
import os
import torch

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "rl"
    )))

    from algorithms.ppo.ppo_agent import PPOPolicy
except ImportError as e:
    print(f"[{__name__}] Error: {e}")

def test_ppo_policy():
    device = "cpu"

    # Fake observation config (match encoder input)
    obs_config = {
        "camera_shape": (3, 84, 84),
        "lidar_shape": (1, 64, 64),
        "ego_state_dim": 6,
        "risk_feature_dim": 3,
        "latent_dim": 256,
        "use_lidar": True,
        "use_risk": True
    }

    action_dim = 3  # steer, throttle, brake

    # Instantiate PPOPolicy
    policy = PPOPolicy(obs_config, action_dim, device=device)
    policy.eval()  # set to evaluation mode

    # Create dummy inputs
    batch_size = 2
    dummy_obs = {
        "camera": torch.randn(batch_size, *obs_config["camera_shape"]).to(device),
        "lidar": torch.randn(batch_size, *obs_config["lidar_shape"]).to(device),
        "ego_state": torch.randn(batch_size, obs_config["ego_state_dim"]).to(device),
        "risk_features": torch.randn(batch_size, obs_config["risk_feature_dim"]).to(device)
    }

    # Test act()
    action, log_prob, value = policy.act(dummy_obs)
    print("act() outputs:")
    print("Action shape:", action.shape)       # should be [batch_size, 3]
    print("Log prob shape:", log_prob.shape)   # should be [batch_size]
    print("Value shape:", value.shape)         # should be [batch_size, 1] or [batch_size]

    # Test evaluate()
    log_prob2, entropy, value2 = policy.evaluate(dummy_obs, action)
    print("\nevaluate() outputs:")
    print("Log prob shape:", log_prob2.shape)
    print("Entropy shape:", entropy.shape)
    print("Value shape:", value2.shape)

if __name__ == "__main__":
    test_ppo_policy()