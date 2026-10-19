"""Shared CPU/CUDA MLP, seeding and host replay storage.

Replay owns its NumPy generator. Training and evaluation torch generators belong
to the runner and must be saved separately for exact continuation.
"""
import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(input_size, output_size, hidden):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, output_size))


def tanh_log_jacobian(raw):
    return 2 * (math.log(2.) - raw - F.softplus(-2 * raw))


class Replay:
    def __init__(self, observation_size, action_size, capacity=20000, seed=1):
        if any(not isinstance(v, int) or v < 1 for v in
               (observation_size, action_size, capacity)):
            raise ValueError("Replay dimensions and capacity must be positive integers")
        self.capacity = int(capacity)
        self.observation = np.zeros((capacity, observation_size), np.float32)
        self.next_observation = np.zeros_like(self.observation)
        self.action = np.zeros((capacity, action_size), np.float32)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.terminated = np.zeros((capacity, 1), np.float32)
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, terminated):
        p = self.position
        self.observation[p] = np.asarray(observation, np.float32).reshape(-1)
        self.action[p] = np.asarray(action, np.float32).reshape(-1)
        self.reward[p, 0] = reward
        self.next_observation[p] = np.asarray(next_observation, np.float32).reshape(-1)
        self.terminated[p, 0] = terminated
        self.position = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, observation, action, reward, next_observation, terminated):
        """Insert chronological vector-environment transitions in one host copy.

        This preserves exactly the ring layout of repeated ``add`` operations,
        including batches longer than capacity and wraparound at any position.
        """
        obs = np.asarray(observation, np.float32)
        act = np.asarray(action, np.float32)
        next_obs = np.asarray(next_observation, np.float32)
        rew = np.asarray(reward, np.float32).reshape(-1, 1)
        done = np.asarray(terminated, np.float32).reshape(-1, 1)
        count = len(obs)
        expected = ((count, self.observation.shape[1]),
                    (count, self.action.shape[1]), (count, 1),
                    (count, self.observation.shape[1]), (count, 1))
        arrays = (obs, act, rew, next_obs, done)
        if any(array.shape != shape for array, shape in zip(arrays, expected)):
            raise ValueError("Replay batch shapes differ")
        if not np.isin(done, (0., 1.)).all():
            raise ValueError("Replay terminal flags must be binary")
        if count == 0:
            return
        # Positions skipped by trimming are still counted in ring chronology.
        skip = max(0, count - self.capacity)
        indices = (self.position + np.arange(skip, count)) % self.capacity
        for destination, source in zip((self.observation, self.action, self.reward,
                                       self.next_observation, self.terminated), arrays):
            destination[indices] = source[skip:]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size, device):
        if self.size < 1 or batch_size < 1:
            raise ValueError("Cannot sample an empty replay")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(array[indices], device=device)
                     for array in (self.observation, self.action, self.reward,
                                   self.next_observation, self.terminated))

    def state_dict(self):
        return {name: copy.deepcopy(getattr(self, name)) for name in
                ("capacity", "observation", "next_observation", "action", "reward",
                 "terminated", "position", "size")} | {
                     "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay shape or capacity differs")
        if not (0 <= state["position"] < self.capacity and 0 <= state["size"] <= self.capacity):
            raise ValueError("Invalid replay indices")
        for name in ("observation", "next_observation", "action", "reward", "terminated"):
            if getattr(self, name).shape != state[name].shape:
                raise ValueError("Replay shape or capacity differs")
        for name, value in state.items():
            if name != "rng":
                setattr(self, name, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])



def observation_tensor(observation, observation_size, device):
    array = np.asarray(observation, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != observation_size:
        raise ValueError("Batched observation shape differs")
    return torch.as_tensor(array, device=device)
