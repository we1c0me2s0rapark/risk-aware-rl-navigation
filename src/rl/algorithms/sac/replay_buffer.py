import torch
import numpy as np
import random

class ReplayBuffer:
    """
    @class ReplayBuffer
    @brief Off-policy experience replay buffer for SAC.

    @details
    Stores transitions (obs, action, reward, next_obs, done) and samples
    random minibatches for off-policy training. Unlike PPO's RolloutBuffer,
    this buffer is not cleared after each update - transitions are retained
    and overwritten in a FIFO manner once capacity is reached.

    Observations are stored as dictionaries of tensors to support the
    multimodal observation structure (camera, lidar, ego_state, risk_features).
    """

    def __init__(self, capacity: int, device: torch.device, min_samples: int):
        """
        @brief Initialise the replay buffer.

        @param capacity int Maximum number of transitions to store.
        @param device torch.device Device to move sampled tensors to.
        @param min_samples int Minimum transitions required before sampling is allowed.
        """
        self.capacity = capacity
        self.device = device
        self.min_samples = min_samples
        self.buffer = [None] * capacity
        self._write_idx = 0
        self._size = 0

    def store(
        self,
        obs: dict,
        action: torch.Tensor,
        reward: np.ndarray,
        next_obs: dict,
        done: bool
    ):
        """
        @brief Store a single transition in the buffer.

        @details
        Observations are stored as CPU tensors to minimise GPU memory usage.
        They are moved to the target device at sample time.

        @param obs dict Unbatched observation dict with tensors of shape [...].
        @param action torch.Tensor Action tensor of shape [action_dim].
        @param reward np.ndarray Reward vector of shape [n_objectives] or scalar.
        @param next_obs dict Unbatched next observation dict.
        @param done bool Episode termination flag.
        """
        # Store everything on CPU to conserve GPU memory
        obs_cpu = {k: v.detach().cpu() for k, v in obs.items()}
        next_obs_cpu = {k: v.detach().cpu() for k, v in next_obs.items()}
        action_cpu = action.detach().cpu()
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(float(done), dtype=torch.float32)

        self.buffer[self._write_idx] = (obs_cpu, action_cpu, reward_tensor, next_obs_cpu, done_tensor)
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        @brief Sample a random minibatch of transitions.

        @details
        Samples uniformly at random from the buffer and stacks each field
        into a batched tensor on the target device.

        @param batch_size int Number of transitions to sample.
        @return tuple (obs, actions, rewards, next_obs, dones) where:
            - obs: dict of batched tensors [B, ...]
            - actions: torch.Tensor [B, action_dim]
            - rewards: torch.Tensor [B, n_objectives] or [B]
            - next_obs: dict of batched tensors [B, ...]
            - dones: torch.Tensor [B]
        """
        indices = random.sample(range(self._size), batch_size)
        batch = [self.buffer[i] for i in indices]
        obs_list, actions, rewards, next_obs_list, dones = zip(*batch)

        # Stack observations dict-of-tensors → dict of batched tensors
        obs = {
            k: torch.stack([o[k] for o in obs_list]).to(self.device)
            for k in obs_list[0]
        }
        next_obs = {
            k: torch.stack([o[k] for o in next_obs_list]).to(self.device)
            for k in next_obs_list[0]
        }

        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.stack(dones).to(self.device)

        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        """
        @brief Return the current number of stored transitions.

        @return int Number of transitions in the buffer.
        """
        return self._size

    @property
    def ready(self) -> bool:
        """
        @brief Check whether the buffer has enough transitions to sample.
        
        @return bool True if buffer contains at least min_samples transitions.
        """
        return self._size >= self.min_samples