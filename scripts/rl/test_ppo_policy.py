import sys
import os
import torch

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src"
    )))
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "rl"
    )))

    from managers.utils.logger import Log
    from algorithms.ppo.ppo_policy import PPOPolicy
except ImportError as e:
    Log.error(__file__, e)

def test_ppo_policy():
    """
    @brief Test the PPOPolicy class with synthetic observations.

    This function performs a basic sanity check of the PPO policy by:
    - Initialising the policy with a dummy observation configuration.
    - Generating a batch of fake observations.
    - Testing the `act()` method for action selection.
    - Testing the `evaluate()` method for log-probability, entropy, and value outputs.

    @note The function uses CPU by default; change `device` to 'cuda' to test on GPU.
    """

    device = "cpu"

    # --- Dummy observation configuration ---
    obs_config = {
        "camera_shape": (3, 84, 84),
        "lidar_shape": (1, 64, 64),
        "ego_state_dim": 6,
        "risk_feature_dim": 3,
        "latent_dim": 256,
        "hidden_dim": 128,
        "use_lidar": True,
        "use_risk": True
    }

    action_dim = 3 # steer, throttle, brake

    # --- Instantiate PPOPolicy ---
    policy = PPOPolicy(obs_config, action_dim, device=device)
    policy.eval() # set to evaluation mode

    # --- Create dummy batch of observations ---
    batch_size = 2
    dummy_obs = {
        "camera": torch.randn(batch_size, *obs_config["camera_shape"]).to(device),
        "lidar": torch.randn(batch_size, *obs_config["lidar_shape"]).to(device),
        "ego_state": torch.randn(batch_size, obs_config["ego_state_dim"]).to(device),
        "risk_features": torch.randn(batch_size, obs_config["risk_feature_dim"]).to(device)
    }

    # --- Test act() method ---
    action, log_prob, value = policy.act(dummy_obs)

    # --- Test evaluate() method ---
    log_prob2, entropy, value2 = policy.evaluate(dummy_obs, action)

    Log.info(__file__, f"""
    act() outputs:
        Action shape: {action.shape}
        Log prob shape: {log_prob.shape}
        Value shape: {value.shape}

    evaluate() outputs:
        Log prob shape: {log_prob2.shape}
        Entropy shape: {entropy.shape}
        Value shape: {value2.shape}""")

if __name__ == "__main__":
    """
    @brief Entry point for the test script.

    Executes `test_ppo_policy()` if the script is run directly.
    """
    test_ppo_policy()