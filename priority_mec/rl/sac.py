"""Plain Soft Actor-Critic: flat MLP, twin Q and learned temperature.

Replay contains bounded requested actions. Routing, resources and safety
projection are environment transitions, never substitutes for replay actions.
"""
import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import mlp, tanh_log_jacobian, observation_tensor

LEARNER_VERSION = 1


@dataclass(frozen=True)
class SACConfig:
    hidden: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-4
    gamma: float = 1.0
    tau: float = .005
    initial_alpha: float = .01
    batch_size: int = 256
    replay_capacity: int = 20000

    def __post_init__(self):
        if (any(not isinstance(value, int) or value < 1 for value in
                (self.hidden, self.batch_size, self.replay_capacity))
                or self.replay_capacity < self.batch_size):
            raise ValueError("Invalid hidden, batch size or replay capacity")
        if not all(math.isfinite(value) for value in (self.actor_lr, self.critic_lr,
                self.alpha_lr, self.gamma, self.tau, self.initial_alpha)):
            raise ValueError("SAC configuration must be finite")
        if not (0 <= self.gamma <= 1 and 0 < self.tau <= 1):
            raise ValueError("Invalid SAC gamma or tau")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.initial_alpha) <= 0:
            raise ValueError("SAC learning rates and initial alpha must be positive")


class Actor(nn.Module):
    def __init__(self, observation_size, action_size, hidden=128):
        super().__init__()
        self.action_size = action_size
        self.network = mlp(observation_size, 2 * action_size, hidden)

    def forward(self, observation):
        mean, log_std = self.network(observation).chunk(2, dim=-1)
        return mean, log_std.clamp(-20., 2.)

    def sample(self, observation, deterministic=False, generator=None,
               return_diagnostics=False):
        mean, log_std = self(observation)
        if deterministic:
            raw = mean
        else:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                                generator=generator)
            raw = mean + log_std.exp() * noise
        log_prob = (-.5 * (((raw - mean) * (-log_std).exp()).square()
                          + 2 * log_std + math.log(2 * math.pi))
                    - tanh_log_jacobian(raw)).sum(-1, keepdim=True)
        action = torch.tanh(raw)
        if return_diagnostics:
            # Reuse this forward pass; logging must not draw another action or
            # consume an additional training/validation RNG value.
            diagnostics = {"latent_std_mean": log_std.detach().exp().mean(),
                           "mean_action_abs": torch.tanh(mean.detach()).abs().mean()}
            return action, log_prob, diagnostics
        return action, log_prob


class SAC:
    def __init__(self, observation_size, action_size, cfg=None, device="cpu"):
        self.cfg = cfg or SACConfig()
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        if min(self.observation_size, self.action_size) < 1:
            raise ValueError("Observation and action sizes must be positive")
        self.device = torch.device(device)
        self.actor = Actor(observation_size, action_size, self.cfg.hidden).to(self.device)
        self.q1 = mlp(observation_size + action_size, 1, self.cfg.hidden).to(self.device)
        self.q2 = mlp(observation_size + action_size, 1, self.cfg.hidden).to(self.device)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                            lr=self.cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(self.cfg.initial_alpha), device=self.device,
                                      requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
        self.target_entropy = -float(action_size)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.detach().exp()

    def act(self, observation, deterministic=True, generator=None):
        return self.act_batch(np.asarray(observation, np.float32).reshape(1, -1),
                              deterministic, generator)[0]

    def act_batch(self, observations, deterministic=True, generator=None):
        tensor = observation_tensor(observations, self.observation_size, self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(tensor, deterministic, generator)
        return action.cpu().numpy()

    @torch.no_grad()
    def _bellman_target(self, reward, next_obs, terminal):
        next_action, next_log_prob = self.actor.sample(next_obs)
        next_input = torch.cat((next_obs, next_action), -1)
        next_q = torch.minimum(self.target_q1(next_input), self.target_q2(next_input))
        return reward + self.cfg.gamma * (1 - terminal) * (
            next_q - self.alpha * next_log_prob)

    def update(self, replay, update_actor=True):
        obs, action, reward, next_obs, terminal = replay.sample(self.cfg.batch_size, self.device)
        self.actor_optimizer.zero_grad(set_to_none=True)
        target = self._bellman_target(reward, next_obs, terminal)
        value_input = torch.cat((obs, action), -1)
        q1, q2 = self.q1(value_input), self.q2(value_input)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(q_loss):
            raise FloatingPointError("Non-finite SAC critic loss")
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        metrics = {"q_loss": float(q_loss.detach()), "q1_mean": float(q1.detach().mean()),
                   "q2_mean": float(q2.detach().mean()), "target_mean": float(target.mean()),
                   "target_abs_max": float(target.abs().max()), "reward_mean": float(reward.mean()),
                   "actor_updated": int(update_actor)}
        if update_actor:
            self.q1.requires_grad_(False)
            self.q2.requires_grad_(False)
            new_action, log_prob, diagnostics = self.actor.sample(obs, return_diagnostics=True)
            new_input = torch.cat((obs, new_action), -1)
            actor_q = torch.minimum(self.q1(new_input), self.q2(new_input))
            actor_alpha = self.alpha
            actor_loss = (actor_alpha * log_prob - actor_q).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Non-finite SAC actor loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.q1.requires_grad_(True)
            self.q2.requires_grad_(True)
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            metrics.update(actor_loss=float(actor_loss.detach()),
                           alpha_loss=float(alpha_loss.detach()),
                           log_prob_mean=float(log_prob.detach().mean()),
                           latent_std_mean=float(diagnostics["latent_std_mean"]),
                           mean_action_abs=float(diagnostics["mean_action_abs"]),
                           entropy_bonus_mean=float(-actor_alpha * log_prob.detach().mean()),
                           action_saturation=float((new_action.detach().abs() > .98).float().mean()))
        with torch.no_grad():
            for source, target_network in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for source_parameter, target_parameter in zip(source.parameters(), target_network.parameters()):
                    target_parameter.lerp_(source_parameter, self.cfg.tau)
        self.updates += 1
        metrics["alpha"] = float(self.alpha)
        return metrics

    def state_dict(self):
        return copy.deepcopy({"version": LEARNER_VERSION, "config": asdict(self.cfg),
                "observation_size": self.observation_size, "action_size": self.action_size,
                "actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(), "updates": self.updates})

    def load_state_dict(self, state):
        if state["version"] != LEARNER_VERSION or state["config"] != asdict(self.cfg):
            raise ValueError("SAC version or configuration differs")
        if (state["observation_size"], state["action_size"]) != (self.observation_size, self.action_size):
            raise ValueError("SAC observation/action size differs")
        for name in ("actor", "q1", "q2", "target_q1", "target_q2",
                     "actor_optimizer", "q_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.updates = state["updates"]
