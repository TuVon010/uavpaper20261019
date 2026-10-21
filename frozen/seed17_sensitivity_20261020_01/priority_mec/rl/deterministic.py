"""Deterministic actor-critic learners, with behavior noise owned by the runner.

TD3: https://arxiv.org/abs/1802.09477 (double Q, delayed updates and target noise).
DDPG: https://arxiv.org/abs/1509.02971 (one Q and deterministic policy gradient).
The environment receives bounded requested actions; its safety projection is
part of the transition and is never substituted into the replay action.
"""
import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import mlp, observation_tensor

LEARNER_VERSION = 1


@dataclass(frozen=True)
class DeterministicConfig:
    algorithm: str = "td3"
    hidden: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 1.0
    tau: float = .005
    batch_size: int = 256
    replay_capacity: int = 20000
    target_noise: float = .2
    noise_clip: float = .5
    policy_delay: int = 2

    def __post_init__(self):
        if self.algorithm not in {"td3", "ddpg"}:
            raise ValueError("algorithm must be td3 or ddpg")
        if any(not isinstance(v, int) or v < 1 for v in
               (self.hidden, self.batch_size, self.replay_capacity, self.policy_delay)):
            raise ValueError("Sizes and policy delay must be positive integers")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must be at least batch size")
        if not all(math.isfinite(v) for v in (self.actor_lr, self.critic_lr, self.gamma,
                                              self.tau, self.target_noise, self.noise_clip)):
            raise ValueError("Configuration values must be finite")
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ValueError("Learning rates must be positive")
        if not (0 <= self.gamma <= 1 and 0 < self.tau <= 1):
            raise ValueError("Invalid discount or target update rate")
        if self.target_noise < 0 or self.noise_clip < 0:
            raise ValueError("Target noise and clipping radius must be nonnegative")
        if self.algorithm == "ddpg":
            object.__setattr__(self, "policy_delay", 1)


class DeterministicActor(nn.Module):
    def __init__(self, observation_size, action_size, hidden):
        super().__init__()
        self.network = mlp(observation_size, action_size, hidden)

    def forward(self, observation):
        return torch.tanh(self.network(observation))


class DeterministicLearner:
    def __init__(self, observation_size, action_size, cfg=None, device="cpu"):
        self.cfg = cfg or DeterministicConfig()
        self.observation_size, self.action_size = int(observation_size), int(action_size)
        if min(self.observation_size, self.action_size) < 1:
            raise ValueError("Observation and action sizes must be positive")
        self.device = torch.device(device)
        self.actor = DeterministicActor(self.observation_size, self.action_size,
                                        self.cfg.hidden).to(self.device)
        self.q1 = mlp(self.observation_size + self.action_size, 1, self.cfg.hidden).to(self.device)
        self.q2 = (mlp(self.observation_size + self.action_size, 1, self.cfg.hidden).to(self.device)
                   if self.cfg.algorithm == "td3" else None)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = (copy.deepcopy(self.q2).requires_grad_(False)
                          if self.q2 is not None else None)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        q_parameters = list(self.q1.parameters())
        if self.q2 is not None:
            q_parameters += list(self.q2.parameters())
        self.q_optimizer = torch.optim.Adam(q_parameters, lr=self.cfg.critic_lr)
        self.updates = 0

    def act(self, observation, deterministic=True, generator=None):
        """The actor is deterministic; the runner supplies behavior noise.

        Common-interface deterministic/generator arguments do not consume RNG.
        Target policy smoothing is used in TD3 Bellman targets only.
        """
        return self.act_batch(np.asarray(observation, np.float32).reshape(1, -1),
                              deterministic, generator)[0]

    def act_batch(self, observations, deterministic=True, generator=None):
        tensor = observation_tensor(observations, self.observation_size, self.device)
        with torch.no_grad():
            return self.actor(tensor).cpu().numpy()

    @torch.no_grad()
    def _bellman_target(self, reward, next_observation, terminal):
        next_action = self.target_actor(next_observation)
        if self.cfg.algorithm == "td3":
            # Clip noise before adding it, then bound the resulting action.
            noise = (torch.randn_like(next_action) * self.cfg.target_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (next_action + noise).clamp(-1., 1.)
        target_input = torch.cat((next_observation, next_action), dim=-1)
        next_q = self.target_q1(target_input)
        if self.target_q2 is not None:
            next_q = torch.minimum(next_q, self.target_q2(target_input))
        return reward + self.cfg.gamma * (1. - terminal) * next_q

    @torch.no_grad()
    def _update_targets(self):
        pairs = [(self.actor, self.target_actor), (self.q1, self.target_q1)]
        if self.q2 is not None:
            pairs.append((self.q2, self.target_q2))
        for source, target in pairs:
            for p, target_p in zip(source.parameters(), target.parameters()):
                target_p.lerp_(p, self.cfg.tau)

    def update(self, replay, update_actor=True):
        obs, action, reward, next_obs, terminal = replay.sample(self.cfg.batch_size, self.device)
        # Critic-only updates leave no stale actor gradients.
        self.actor_optimizer.zero_grad(set_to_none=True)
        target = self._bellman_target(reward, next_obs, terminal)
        value_input = torch.cat((obs, action), dim=-1)
        q1 = self.q1(value_input)
        q2 = self.q2(value_input) if self.q2 is not None else None
        q_loss = F.mse_loss(q1, target)
        if q2 is not None:
            q_loss = q_loss + F.mse_loss(q2, target)
        if not torch.isfinite(q_loss):
            raise FloatingPointError("Non-finite deterministic critic loss")
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        actor_updated = bool(update_actor and (self.updates + 1) % self.cfg.policy_delay == 0)
        metrics = {"q_loss": float(q_loss.detach()), "q1_mean": float(q1.detach().mean()),
                   "target_mean": float(target.mean()), "target_abs_max": float(target.abs().max()),
                   "reward_mean": float(reward.mean()), "actor_updated": int(actor_updated)}
        if q2 is not None:
            metrics["q2_mean"] = float(q2.detach().mean())
        if actor_updated:
            self.q1.requires_grad_(False)
            try:
                new_action = self.actor(obs)
                actor_loss = -self.q1(torch.cat((obs, new_action), dim=-1)).mean()
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Non-finite deterministic actor loss")
                actor_loss.backward()
                self.actor_optimizer.step()
            finally:
                self.q1.requires_grad_(True)
            metrics.update(actor_loss=float(actor_loss.detach()),
                           mean_action_abs=float(new_action.detach().abs().mean()),
                           action_saturation=float((new_action.detach().abs() > .98).float().mean()))
            # All TD3 targets update with its delayed actor; DDPG delay is one.
            self._update_targets()
        self.updates += 1
        return metrics

    def state_dict(self):
        state = {"version": LEARNER_VERSION, "config": asdict(self.cfg),
                 "observation_size": self.observation_size, "action_size": self.action_size,
                 "actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                 "target_actor": self.target_actor.state_dict(),
                 "target_q1": self.target_q1.state_dict(),
                 "q2": self.q2.state_dict() if self.q2 is not None else None,
                 "target_q2": self.target_q2.state_dict() if self.target_q2 is not None else None,
                 "actor_optimizer": self.actor_optimizer.state_dict(),
                 "q_optimizer": self.q_optimizer.state_dict(), "updates": self.updates}
        return copy.deepcopy(state)

    def load_state_dict(self, state):
        if state["version"] != LEARNER_VERSION or state["config"] != asdict(self.cfg):
            raise ValueError("Deterministic learner version or configuration differs")
        if (state["observation_size"], state["action_size"]) != (self.observation_size, self.action_size):
            raise ValueError("Deterministic observation/action size differs")
        if (state["q2"] is None) != (self.q2 is None) or (
                state["target_q2"] is None) != (self.target_q2 is None):
            raise ValueError("Deterministic critic architecture differs")
        for name in ("actor", "q1", "target_actor", "target_q1", "actor_optimizer", "q_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        if self.q2 is not None:
            self.q2.load_state_dict(state["q2"])
            self.target_q2.load_state_dict(state["target_q2"])
        self.updates = int(state["updates"])
