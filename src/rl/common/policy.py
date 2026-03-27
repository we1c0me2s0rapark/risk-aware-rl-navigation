import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Policy(nn.Module, ABC):
    """
    @brief Abstract policy interface for RL algorithms.
    """

    @abstractmethod
    def act(self, obs):
        """
        @brief Sample action from policy.

        @param obs Input observation
        @return action, log_prob, value
        """
        pass

    @abstractmethod
    def evaluate(self, obs, action):
        """
        @brief Evaluate given action.

        @param obs Observation
        @param action Action tensor
        @return log_prob, entropy, value
        """
        pass