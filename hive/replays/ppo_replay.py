import os
import pickle

import numpy as np

from hive.replays.circular_replay import CircularReplayBuffer
from hive.utils.utils import create_folder, seeder

class PPOReplayBuffer(CircularReplayBuffer):
    def __init__(self, capacity: int = 10000, stack_size: int = 1, n_step: int = 1, gamma: float = 0.99, gae_lambda = 0.95, observation_shape=..., observation_dtype=..., action_shape=..., action_dtype=..., reward_shape=..., reward_dtype=..., extra_storage_types=None, num_players_sharing_buffer: int = None):
        super().__init__(capacity, stack_size, n_step, gamma, observation_shape, observation_dtype, action_shape, action_dtype, reward_shape, reward_dtype, extra_storage_types, num_players_sharing_buffer)
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._gae_lambda = gae_lambda
    
    # Taken from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_shared.py
    def compute_advantages(self, next_value, next_done):
        lastgaelam = 0
        for t in reversed(range(self._capacity)):
            if t == self._capacity - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self._storage["done"][t + 1]
                nextvalues = self._storage["value"][t + 1]
            delta = self._storage["reward"][t] + self._gamma * nextvalues * nextnonterminal - self._storage["value"][t]
            self._storage["advantage"] = lastgaelam = delta + self._gamma * self._gae_lambda * nextnonterminal * lastgaelam
            self._storage["return"] = self._storage["advantage"] + self._storage["value"][t]
            
    
    def reset(self):
        self._create_storage(self._capacity, self._specs)
        self._episode_start = True
        self._cursor = 0
        self._num_added = 0    