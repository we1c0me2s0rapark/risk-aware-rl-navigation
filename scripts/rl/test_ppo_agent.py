import os
import sys
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
    from algorithms.ppo.ppo_agent import PPOAgent
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

def test_ppo_agent():
    """
    @brief Test the PPOAgent class with dummy data.

    This function performs a full sanity check of the PPO agent by:
    - Initialising the agent with a dummy observation configuration.
    - Generating a batch of synthetic observations.
    - Running `act()` to select actions.
    - Storing transitions in the rollout buffer.
    - Performing a PPO update and clearing the buffer.

    @note Uses CPU by default; change `device` to 'cuda' to test on GPU.
    """

    device = "cpu" # Change to "cuda" if GPU is available

    # --- Dummy observation configuration ---
    obs_config = {
        "camera_shape": (3, 84, 84),
        "lidar_shape": (1, 64, 64),
        "ego_state_dim": 6,
        "risk_feature_dim": 3,
        "latent_dim": 256,
        "use_lidar": True,
        "use_risk": True
    }

    action_dim = 3  # e.g., steer, throttle, brake

    # --- Instantiate PPOAgent ---
    agent = PPOAgent(obs_config, action_dim, device=device)
    agent.policy.eval() # Set policy to evaluation mode

    # --- Create dummy batch of observations ---
    batch_size = 4
    dummy_obs = {
        "camera": torch.randn(batch_size, *obs_config["camera_shape"]).to(device),
        "lidar": torch.randn(batch_size, *obs_config["lidar_shape"]).to(device),
        "ego_state": torch.randn(batch_size, obs_config["ego_state_dim"]).to(device),
        "risk_features": torch.randn(batch_size, obs_config["risk_feature_dim"]).to(device)
    }

    # --- Test act() method ---
    action, log_prob, value = agent.act(dummy_obs)
    log_text = f"""TEST act()
    Action shape: {action.shape}
    Log prob shape: {log_prob.shape}
    Value shape: {value.shape}"""


    # --- Store transitions in buffer ---
    log_text += "\nStoring transitions in buffer"
    for i in range(batch_size):
        obs_i = {k: v[i] for k, v in dummy_obs.items()}
        agent.store(
            obs_i, 
            action[i], 
            log_prob[i], 
            reward=1.0, 
            done=False, 
            value=value[i]
        )
        
    log_text += f"Buffer length: {len(agent.buffer.actions)}"

    # --- Perform PPO update ---
    log_text += "\nPerforming update"

    agent.update(last_obs=dummy_obs, done=False)

    log_text += "Update complete. Buffer should be cleared."
    log_text += f"Buffer length after clear: {len(agent.buffer.actions)}"

    Log.info(__file__, log_text)


if __name__ == "__main__":
    """
    @brief Entry point for the test script.

    Executes `test_ppo_agent()` if the script is run directly.
    """
    test_ppo_agent()