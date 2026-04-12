import os
import torch
from managers.utils.logger import Log

class CheckpointManager:
    """
    @class CheckpointManager
    @brief Handles saving and loading of agent checkpoints for PPO and SAC.

    @details
    Supports both PPO (single optimiser) and SAC (three separate optimisers)
    by detecting the agent type at save/load time. The checkpoint format
    differs between the two:

    PPO checkpoint keys:
        rollout_count, encoder, actor, critic, optimiser

    SAC checkpoint keys:
        step_count, encoder, actor, critic,
        actor_optimiser, critic_optimiser, alpha_optimiser, log_alpha
    """

    def __init__(self, parent_dir: str, ws_dir: str, filename: str = "checkpoint.pth"):
        """
        @brief Initialise the checkpoint manager.

        @param parent_dir str Root directory for all checkpoints.
        @param ws_dir str Subdirectory identifying the algorithm (e.g. 'ppo', 'sac').
        @param filename str Checkpoint filename.
        """
        os.makedirs(parent_dir, exist_ok=True)

        ws_path = os.path.join(parent_dir, ws_dir)
        os.makedirs(ws_path, exist_ok=True)

        self.path = os.path.join(ws_path, filename)

    def _is_sac(self, agent) -> bool:
        """
        @brief Detect whether the agent is a SAC agent.

        @details
        Inferred by checking for 'actor_optimiser' on the trainer.
        This assumes PPO trainers never define this attribute.

        @param agent Agent instance to inspect.
        @return bool True if SAC, False if PPO.
        """
        return hasattr(agent.trainer, 'actor_optimiser')

    def save(self, agent, count: int):
        """
        @brief Save the agent's state to disk.

        @param agent PPOAgent or SACAgent instance.
        @param count int Rollout count (PPO) or step count (SAC).
        """
        if self._is_sac(agent):
            torch.save({
                'step_count':       count,
                'encoder':          agent.policy.encoder.state_dict(),
                'actor':            agent.policy.actor.state_dict(),
                'critic':           agent.policy.critic.state_dict(),
                'actor_optimiser':  agent.trainer.actor_optimiser.state_dict(),
                'critic_optimiser': agent.trainer.critic_optimiser.state_dict(),
                'alpha_optimiser':  agent.trainer.alpha_optimiser.state_dict(),
                'log_alpha':        agent.trainer.log_alpha,
            }, self.path)
            Log.info(__file__, f"💾 SAC checkpoint saved at step {count} → {self.path}")
        else:
            torch.save({
                'rollout_count': count,
                'encoder':       agent.policy.encoder.state_dict(),
                'actor':         agent.policy.actor.state_dict(),
                'critic':        agent.policy.critic.state_dict(),
                'optimiser':     agent.trainer.optimiser.state_dict(),
            }, self.path)
            Log.info(__file__, f"💾 PPO checkpoint saved at rollout {count} → {self.path}")

    def load(self, agent) -> int:
        """
        @brief Load the agent's state from disk if a checkpoint exists.

        @param agent PPOAgent or SACAgent instance.
        @return int Rollout count (PPO) or step count (SAC), or 0 if no checkpoint.
        """
        if not os.path.exists(self.path):
            Log.info(__file__, f"No checkpoint found at {self.path} - starting from scratch.")
            return 0

        ckpt = torch.load(self.path, map_location=agent.device)
        agent.policy.encoder.load_state_dict(ckpt['encoder'])
        agent.policy.actor.load_state_dict(ckpt['actor'])
        agent.policy.critic.load_state_dict(ckpt['critic'])

        if self._is_sac(agent):
            agent.trainer.actor_optimiser.load_state_dict(ckpt['actor_optimiser'])
            agent.trainer.critic_optimiser.load_state_dict(ckpt['critic_optimiser'])
            agent.trainer.alpha_optimiser.load_state_dict(ckpt['alpha_optimiser'])
            agent.trainer.log_alpha = ckpt['log_alpha']
            agent.trainer.alpha = agent.trainer.log_alpha.exp().item()
            count = ckpt.get('step_count', 0)
            Log.info(__file__, f"✅ SAC checkpoint loaded - resuming from step {count}.")
        else:
            agent.trainer.optimiser.load_state_dict(ckpt['optimiser'])
            count = ckpt.get('rollout_count', 0)
            Log.info(__file__, f"✅ PPO checkpoint loaded - resuming from rollout {count}.")

        return count