import os
import torch
from managers.utils.logger import Log

class CheckpointManager:
    """
    @class CheckpointManager
    @brief Manages saving and loading of PPO agent checkpoints.

    Persists policy weights and optimiser state across training sessions,
    allowing training to resume from the last saved rollout.
    """

    def __init__(self, checkpoint_dir: str, filename: str = "ppo_checkpoint.pth"):
        """
        @brief Initialise the CheckpointManager.

        @param checkpoint_dir str Directory in which to store checkpoint files.
        @param filename str Name of the checkpoint file.
        """
        self.path = os.path.join(checkpoint_dir, filename)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, agent, rollout_count: int):
        """
        @brief Save agent state to disk.

        @param agent PPOAgent The agent whose state is to be saved.
        @param rollout_count int Number of completed rollouts, stored for resumption.
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
        @brief Load agent state from disk if a checkpoint exists.

        @param agent PPOAgent The agent into which state will be loaded.
        @return int Number of completed rollouts at the time of saving, or 0 if no checkpoint found.
        """
        if not os.path.exists(self.path):
            Log.info(__file__, "No checkpoint found — starting from scratch.")
            return 0

        ckpt = torch.load(self.path, map_location=agent.device)
        agent.policy.encoder.load_state_dict(ckpt['encoder'])
        agent.policy.actor.load_state_dict(ckpt['actor'])
        agent.policy.critic.load_state_dict(ckpt['critic'])
        agent.trainer.optimiser.load_state_dict(ckpt['optimiser'])

        rollout_count = ckpt.get('rollout_count', 0)
        Log.info(__file__, f"✅ Checkpoint loaded - resuming from rollout {rollout_count}.")
        return rollout_count