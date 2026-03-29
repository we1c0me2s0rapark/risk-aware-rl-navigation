import os
import torch
from managers.utils.logger import Log

class CheckpointManager:
    """
    @class CheckpointManager
    @brief Handles saving and loading of PPO agent checkpoints.

    This class manages the persistence of the agent's neural network weights
    and optimiser state, allowing training to be paused and resumed from the
    last saved checkpoint.
    """

    def __init__(self, checkpoint_dir: str, filename: str = "ppo_checkpoint.pth"):
        """
        @brief Initialise the checkpoint manager.

        Ensures that the checkpoint directory exists.

        @param checkpoint_dir Directory in which to store checkpoint files.
        @param filename Name of the checkpoint file (default: "ppo_checkpoint.pth").
        """

        self.path = os.path.join(checkpoint_dir, filename)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, agent, rollout_count: int):
        """
        @brief Save the agent's state to disk.

        Persists the policy, critic and encoder network weights, together
        with the optimiser state. Also stores the number of completed rollouts
        to allow training to resume.

        @param agent The PPO agent whose state is to be saved.
        @param rollout_count Number of completed rollouts.
        """

        torch.save({
            'rollout_count': rollout_count,
            'encoder':       agent.policy.encoder.state_dict(),
            'actor':         agent.policy.actor.state_dict(),
            'critic':        agent.policy.critic.state_dict(),
            'optimiser':     agent.trainer.optimiser.state_dict(),
        }, self.path)
        Log.info(__file__, f"💾 Checkpoint saved at rollout {rollout_count} → {self.path}")

    def load(self, agent) -> int:
        """
        @brief Load the agent's state from disk if a checkpoint exists.

        Restores the encoder, actor, critic and optimiser states from the
        checkpoint file. Returns the rollout count to resume training from.
        If no checkpoint is found, returns 0.

        @param agent The PPO agent into which the state will be loaded.
        @return Number of completed rollouts at the time of saving, or 0 if no checkpoint exists.
        """
        
        if not os.path.exists(self.path):
            Log.info(__file__, "No checkpoint found - starting from scratch.")
            return 0

        ckpt = torch.load(self.path, map_location=agent.device)
        agent.policy.encoder.load_state_dict(ckpt['encoder'])
        agent.policy.actor.load_state_dict(ckpt['actor'])
        agent.policy.critic.load_state_dict(ckpt['critic'])
        agent.trainer.optimiser.load_state_dict(ckpt['optimiser'])

        rollout_count = ckpt.get('rollout_count', 0)
        Log.info(__file__, f"✅ Checkpoint loaded - resuming from rollout {rollout_count}.")
        return rollout_count