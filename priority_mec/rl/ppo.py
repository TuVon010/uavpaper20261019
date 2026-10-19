"""PPO-Clip with a tanh Gaussian policy and finite-episode GAE.

The replayed/requested action is bounded before the environment projects it.
Its *unsquashed* Gaussian sample is also retained: log probabilities can then
be recomputed exactly even when float32 tanh has rounded an action to +/-1.
The default critic uses ordinary MSE; optional value clipping is a separately
declared implementation choice rather than an unlabelled PPO2 modification.
"""
import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import observation_tensor

LEARNER_VERSION = 6


@dataclass(frozen=True)
class PPOConfig:
    hidden: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = .95
    clip_ratio: float = .2
    value_clip: float | None = None
    entropy_coef: float = 0.0
    value_coef: float = .5
    max_grad_norm: float = .5
    epochs: int = 10
    critic_epochs: int | None = None
    minibatch_size: int = 128
    target_kl: float | None = .02
    rollout_steps: int = 512
    variant: str = 'baseline'
    popart_beta: float = .01
    popart_min_std: float = .1
    motion_memory: bool = False
    ar_rho: float = 0.
    known_delay_index: int | None = None
    continuation_critic: bool = False
    continuation_mode: str = 'known'
    ar_adaptive: bool = False
    ar_rho_max: float = .9
    schedule_steps: int = 0
    noise_final_scale: float = 1.
    noise_anneal_start: float = .1
    noise_anneal_end: float = .7
    actor_lr_final_scale: float = 1.

    def __post_init__(self):
        if any(not isinstance(value, int) or value < 1 for value in
               (self.hidden, self.epochs, self.minibatch_size, self.rollout_steps)):
            raise ValueError("PPO dimensions and update counts must be positive")
        if self.critic_epochs is not None and (not isinstance(self.critic_epochs, int)
                                               or self.critic_epochs < 1):
            raise ValueError('PPO critic epochs must be positive or None')
        values = (self.actor_lr, self.critic_lr, self.gamma, self.gae_lambda,
                  self.clip_ratio, self.entropy_coef, self.value_coef, self.max_grad_norm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PPO configuration must be finite")
        if min(self.actor_lr, self.critic_lr, self.value_coef, self.max_grad_norm) <= 0:
            raise ValueError("PPO learning rates, value coefficient and gradient bound must be positive")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("PPO gamma and GAE lambda must be in [0,1]")
        if not 0 < self.clip_ratio < 1 or self.entropy_coef < 0:
            raise ValueError("Invalid PPO clip ratio or entropy coefficient")
        if self.value_clip is not None and self.value_clip <= 0:
            raise ValueError("PPO value clip must be positive or None")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("PPO target KL must be positive or None")
        if self.variant not in {'baseline', 'value_decomp', 'value_decomp_popart'}:
            raise ValueError('Unknown PPO architecture variant')
        if not (math.isfinite(self.popart_beta) and 0 < self.popart_beta <= 1
                and math.isfinite(self.popart_min_std) and self.popart_min_std > 0):
            raise ValueError('Invalid PopArt statistics settings')
        if not isinstance(self.motion_memory, bool) or not isinstance(self.continuation_critic, bool):
            raise ValueError('Motion-memory and continuation switches must be Boolean')
        if not (math.isfinite(self.ar_rho) and 0 <= self.ar_rho < 1):
            raise ValueError('Autoregressive rho must be finite and in [0,1)')
        if self.ar_rho and not self.motion_memory:
            raise ValueError('Autoregressive motion requires observable previous latent motion')
        if not isinstance(self.ar_adaptive, bool):
            raise ValueError('Adaptive autoregression switch must be Boolean')
        if not (math.isfinite(self.ar_rho_max) and 0 < self.ar_rho_max < 1):
            raise ValueError('Adaptive rho maximum must be finite and in (0,1)')
        if self.ar_adaptive and (not self.motion_memory or not 0 < self.ar_rho < self.ar_rho_max):
            raise ValueError('Adaptive autoregression requires visible memory and 0 < rho < rho_max')
        if (not isinstance(self.schedule_steps, int) or isinstance(self.schedule_steps, bool)
                or self.schedule_steps < 0):
            raise ValueError('Schedule steps must be a nonnegative integer')
        if not all(math.isfinite(value) and value > 0 for value in
                   (self.noise_final_scale, self.actor_lr_final_scale)):
            raise ValueError('Final noise and actor learning-rate scales must be positive and finite')
        if not (math.isfinite(self.noise_anneal_start) and math.isfinite(self.noise_anneal_end)
                and 0 <= self.noise_anneal_start < self.noise_anneal_end <= 1):
            raise ValueError('Noise annealing interval must satisfy 0 <= start < end <= 1')
        if (self.known_delay_index is not None
                and (not isinstance(self.known_delay_index, int)
                     or isinstance(self.known_delay_index, bool) or self.known_delay_index < 0)):
            raise ValueError('Known delay index must be a nonnegative integer or None')
        if self.continuation_mode not in {'known', 'learned'}:
            raise ValueError('Continuation mode must be known or learned')
        if self.continuation_critic:
            if self.variant != 'value_decomp_popart':
                raise ValueError('Continuation critic requires PopArt components')
            if self.continuation_mode == 'known' and self.known_delay_index is None:
                raise ValueError('Known continuation requires observed current delay')
            if self.continuation_mode == 'learned' and self.known_delay_index is not None:
                raise ValueError('Learned continuation forbids an observed current-delay feature')
        elif self.continuation_mode != 'known':
            raise ValueError('A learned continuation mode requires continuation_critic')


def compute_gae(reward, value, next_value, terminated, gamma=1., gae_lambda=.95,
                stream_id=None):
    """Return GAE and lambda-returns, stopping recursion at every true terminal.

    next_value belongs to *that transition's* next observation, not a shifted
    array that might accidentally bootstrap from a reset episode. A nonterminal
    end of rollout bootstraps its value but does not invent a future advantage.
    """
    arrays = [np.asarray(x, np.float64).reshape(-1)
              for x in (reward, value, next_value, terminated)]
    if len({len(x) for x in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("GAE arrays must have the same nonzero length")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Non-finite GAE input")
    reward, value, next_value, terminated = arrays
    if not np.isin(terminated, (0., 1.)).all():
        raise ValueError("GAE terminal flags must be binary")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("Invalid GAE discount or lambda")
    if stream_id is not None:
        streams = np.asarray(stream_id)
        if (streams.shape != reward.shape or not np.issubdtype(streams.dtype, np.integer)):
            raise ValueError("GAE stream_id must be an integer per transition")
        advantages = np.empty_like(reward, dtype=np.float32)
        returns = np.empty_like(reward, dtype=np.float32)
        for stream in np.unique(streams):
            indices = np.flatnonzero(streams == stream)
            a, r = compute_gae(reward[indices], value[indices], next_value[indices],
                               terminated[indices], gamma, gae_lambda)
            advantages[indices], returns[indices] = a, r
        return advantages, returns
    advantage = np.empty_like(reward)
    following_advantage = 0.
    for index in range(len(reward) - 1, -1, -1):
        continuation = 1. - terminated[index]
        delta = reward[index] + gamma * continuation * next_value[index] - value[index]
        following_advantage = delta + gamma * gae_lambda * continuation * following_advantage
        advantage[index] = following_advantage
    return advantage.astype(np.float32), (advantage + value).astype(np.float32)


def clipped_policy_loss(new_log_prob, old_log_prob, advantage, clip_ratio):
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1. - clip_ratio, 1. + clip_ratio) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def tanh_log_jacobian(raw):
    return 2. * (math.log(2.) - raw - F.softplus(-2. * raw))


def _mlp(input_size, output_size, hidden):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, output_size))


class Actor(nn.Module):
    def __init__(self, observation_size, action_size, hidden, motion_memory=False,
                 ar_rho=0., memory_start_index=None, ar_adaptive=False, ar_rho_max=.9):
        super().__init__()
        if not (math.isfinite(ar_rho) and 0 <= ar_rho < 1):
            raise ValueError('Autoregressive rho must be finite and in [0,1)')
        if ar_rho and not motion_memory:
            raise ValueError('Autoregressive motion requires visible motion memory')
        if motion_memory and (action_size < 6 or memory_start_index is None
                              or not 0 <= memory_start_index <= observation_size - 6):
            raise ValueError('Motion memory must identify six observation features and actions')
        if not isinstance(ar_adaptive, bool):
            raise ValueError('Adaptive autoregression switch must be Boolean')
        if not (math.isfinite(ar_rho_max) and 0 < ar_rho_max < 1):
            raise ValueError('Adaptive rho maximum must be finite and in (0,1)')
        if ar_adaptive and (not motion_memory or not 0 < ar_rho < ar_rho_max):
            raise ValueError('Adaptive autoregression requires visible memory and 0 < rho < rho_max')
        self.motion_memory = bool(motion_memory)
        self.ar_rho = float(ar_rho)
        self.memory_start_index = memory_start_index
        self.ar_adaptive = ar_adaptive
        self.ar_rho_max = float(ar_rho_max)
        self.noise_scale = 1.
        self.network = _mlp(observation_size, action_size, hidden)
        self.log_std = nn.Parameter(torch.full((action_size,), -.5))
        if self.ar_adaptive:
            # Direct zero creation consumes no RNG, keeping the base network
            # and all subsequent initialization identical to the fixed policy.
            self.rho_weight = nn.Parameter(torch.zeros(3, observation_size))
            self.rho_bias = nn.Parameter(torch.full((3,), math.log(ar_rho / (ar_rho_max - ar_rho))))

    def set_noise_scale(self, scale):
        if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Policy noise scale must be positive and finite')
        self.noise_scale = float(scale)

    def motion_rho(self, observation):
        """Observed-state rho for the three two-dimensional UAV actions."""
        if self.ar_adaptive:
            return self.ar_rho_max * torch.sigmoid(F.linear(observation, self.rho_weight, self.rho_bias))
        return observation.new_full((*observation.shape[:-1], 3), self.ar_rho)

    def forward(self, observation):
        mean = self.network(observation)
        # Bounds prevent numerical overflow without changing normal operating
        # behavior; the same bound is used during rollout and every update.
        log_std = self.log_std.clamp(-20., 2.).expand_as(mean)
        if self.ar_adaptive:
            rho = self.motion_rho(observation).repeat_interleave(2, dim=-1)
            previous = observation[..., self.memory_start_index:self.memory_start_index + 6]
            motion_mean = rho * previous + (1. - rho) * mean[..., :6]
            mean = torch.cat((motion_mean, mean[..., 6:]), dim=-1)
            motion_log_std = log_std[..., :6] + .5 * torch.log1p(-rho.square())
            log_std = torch.cat((motion_log_std, log_std[..., 6:]), dim=-1)
        elif self.ar_rho:
            # This is the actual conditional Normal. Memory is observed data;
            # rollout and PPO likelihood recomputation use the same forward.
            previous = observation[..., self.memory_start_index:self.memory_start_index + 6]
            motion_mean = self.ar_rho * previous + (1. - self.ar_rho) * mean[..., :6]
            mean = torch.cat((motion_mean, mean[..., 6:]), dim=-1)
            motion_log_std = log_std[..., :6] + .5 * math.log1p(-self.ar_rho ** 2)
            log_std = torch.cat((motion_log_std, log_std[..., 6:]), dim=-1)
        if self.noise_scale != 1.:
            log_std = log_std + math.log(self.noise_scale)
        return mean, log_std

    @staticmethod
    def log_prob(raw, mean, log_std):
        normal_log_prob = -.5 * (((raw - mean) * (-log_std).exp()).square()
                                 + 2. * log_std + math.log(2. * math.pi))
        return (normal_log_prob - tanh_log_jacobian(raw)).sum(-1)

    def sample(self, observation, deterministic=False, generator=None):
        mean, log_std = self(observation)
        if deterministic:
            raw = mean
        else:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                                generator=generator)
            raw = mean + log_std.exp() * noise
        return torch.tanh(raw), self.log_prob(raw, mean, log_std), raw


class PPO:
    def __init__(self, observation_size, action_size, cfg=None, device="cpu", physics=None):
        self.cfg = cfg or PPOConfig()
        self.observation_size, self.action_size = int(observation_size), int(action_size)
        if min(self.observation_size, self.action_size) < 1:
            raise ValueError("Observation and action sizes must be positive")
        self.device = torch.device(device)
        self.physics = copy.deepcopy(physics)
        if self.cfg.known_delay_index is not None and self.cfg.known_delay_index >= self.observation_size:
            raise ValueError('Known delay index exceeds the observation size')
        memory_start = self.observation_size - 6 - int(self.cfg.known_delay_index is not None)
        if self.cfg.motion_memory and (self.cfg.known_delay_index is not None
                                       and self.cfg.known_delay_index != self.observation_size - 1):
            raise ValueError('Known delay must follow the six appended motion-memory features')
        self.actor = Actor(observation_size, action_size, self.cfg.hidden,
                           self.cfg.motion_memory, self.cfg.ar_rho, memory_start,
                           self.cfg.ar_adaptive, self.cfg.ar_rho_max).to(self.device)
        if self.cfg.variant in {'value_decomp', 'value_decomp_popart'}:
            from .value_components import (RewardDecomposedCritic, PopArtRewardDecomposedCritic,
                                           PopArtContinuationCritic, PopArtLearnedContinuationCritic)
            if self.cfg.variant == 'value_decomp_popart':
                if self.cfg.continuation_critic:
                    horizon = (physics.get('episode_frames') if isinstance(physics, dict)
                               else getattr(physics, 'episode_frames', None))
                    if self.cfg.continuation_mode == 'learned':
                        time_index = self.observation_size - 1 - (6 if self.cfg.motion_memory else 0)
                        self.critic = PopArtLearnedContinuationCritic(
                            observation_size, self.cfg.hidden, self.cfg.popart_beta,
                            self.cfg.popart_min_std, time_index, horizon).to(self.device)
                    else:
                        time_index = self.cfg.known_delay_index - 1 - (6 if self.cfg.motion_memory else 0)
                        self.critic = PopArtContinuationCritic(
                            observation_size, self.cfg.hidden, self.cfg.popart_beta,
                            self.cfg.popart_min_std, self.cfg.known_delay_index,
                            time_index, horizon).to(self.device)
                else:
                    self.critic = PopArtRewardDecomposedCritic(
                        observation_size, self.cfg.hidden, self.cfg.popart_beta,
                        self.cfg.popart_min_std).to(self.device)
            else:
                self.critic = RewardDecomposedCritic(observation_size, self.cfg.hidden).to(self.device)
        else:
            self.critic = _mlp(observation_size, 1, self.cfg.hidden).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.updates = 0
        self.schedule_step = 0

    def set_training_progress(self, step):
        """Set schedules at a rollout boundary, using the configured fixed horizon.

        The runner owns unfinished rollouts and must keep this setting fixed
        until their likelihoods have been consumed by ``update``.
        """
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError('Training progress must be a nonnegative integer')
        self.schedule_step = step
        if self.cfg.schedule_steps:
            progress = min(step / self.cfg.schedule_steps, 1.)
            anneal = min(max((progress - self.cfg.noise_anneal_start)
                             / (self.cfg.noise_anneal_end - self.cfg.noise_anneal_start), 0.), 1.)
            noise_scale = (1. if anneal == 0. else self.cfg.noise_final_scale if anneal == 1.
                           else math.exp(anneal * math.log(self.cfg.noise_final_scale)))
            lr_scale = (self.cfg.actor_lr_final_scale if progress == 1.
                        else 1. + progress * (self.cfg.actor_lr_final_scale - 1.))
        else:
            noise_scale = lr_scale = 1.
        self.actor.set_noise_scale(noise_scale)
        for group in self.actor_optimizer.param_groups:
            group['lr'] = self.cfg.actor_lr * lr_scale

    def policy_diagnostics(self, observations=None):
        metrics = {'active_noise_scale': getattr(self.actor, 'noise_scale', 1.),
                   'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
                   'schedule_step': self.schedule_step}
        if observations is not None:
            obs = observation_tensor(observations, self.observation_size, self.device)
            with torch.no_grad():
                effective_std = self.actor(obs)[1].exp()
                base_std = (self.actor.log_std.clamp(-20., 2.).exp()
                            if hasattr(self.actor, 'log_std') else effective_std)
                for label, std in (('base_std', base_std), ('effective_std', effective_std)):
                    metrics.update({label+'_min': float(std.min()),
                                    label+'_mean': float(std.double().mean()),
                                    label+'_max': float(std.max())})
                if self.cfg.ar_adaptive:
                    rho = self.actor.motion_rho(obs)
                    metrics.update(rho_min=float(rho.min()), rho_mean=float(rho.double().mean()),
                                   rho_max=float(rho.max()),
                                   rho_ge_085_fraction=float((rho >= .85).double().mean()))
                    for index in range(3):
                        metrics['rho_uav_'+str(index + 1)+'_mean'] = float(rho[:, index].double().mean())
        return metrics

    def _observation(self, observation):
        return observation_tensor(np.asarray(observation, np.float32).reshape(1, -1),
                                  self.observation_size, self.device)

    def act(self, observation, deterministic=True, generator=None):
        return self.act_batch(np.asarray(observation, np.float32).reshape(1, -1),
                              deterministic, generator)[0]

    def act_batch(self, observations, deterministic=True, generator=None, return_raw=False):
        obs = observation_tensor(observations, self.observation_size, self.device)
        with torch.no_grad():
            action, _, raw = self.actor.sample(obs, deterministic, generator)
        if return_raw:
            return action.cpu().numpy(), raw.cpu().numpy()
        return action.cpu().numpy()

    def act_train(self, observation):
        action, log_prob, value, raw = self.act_train_batch(
            np.asarray(observation, np.float32).reshape(1, -1))
        return action[0], float(log_prob[0]), float(value[0]), raw[0]

    def act_train_batch(self, observations, generator=None, return_components=False):
        obs = observation_tensor(observations, self.observation_size, self.device)
        with torch.no_grad():
            action, log_prob, raw = self.actor.sample(obs, generator=generator)
            if self.cfg.variant in {'value_decomp', 'value_decomp_popart'}:
                components = self.critic.forward_components(obs)
                value = components.sum(-1)
            else:
                value = self.critic(obs).squeeze(-1)
        tensors = (action, log_prob, value, raw)
        if return_components:
            if self.cfg.variant not in {'value_decomp', 'value_decomp_popart'}:
                raise ValueError('Component values require value_decomp')
            tensors += (components,)
        return tuple(tensor.cpu().numpy() for tensor in tensors)

    def value_components_batch(self, observations):
        if self.cfg.variant not in {'value_decomp', 'value_decomp_popart'}:
            raise ValueError('Component values require value_decomp')
        obs = observation_tensor(observations, self.observation_size, self.device)
        with torch.no_grad():
            return self.critic.forward_components(obs).cpu().numpy()

    def value(self, observation):
        return float(self.value_batch(np.asarray(observation, np.float32).reshape(1, -1))[0])

    def value_batch(self, observations):
        obs = observation_tensor(observations, self.observation_size, self.device)
        with torch.no_grad():
            return self.critic(obs).squeeze(-1).cpu().numpy()

    def update(self, rollout):
        if "obs" not in rollout and "observation" in rollout:
            rollout = dict(rollout, obs=rollout["observation"])
        names = ("obs", "action", "raw_action", "log_prob", "value", "reward",
                 "terminated", "next_value")
        arrays = {name: np.asarray(rollout[name], np.float32) for name in names}
        count = len(arrays["reward"])
        if not count or any(len(array) != count for array in arrays.values()):
            raise ValueError("PPO rollout lengths differ or rollout is empty")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Non-finite PPO rollout")
        if arrays["obs"].shape != (count, self.observation_size):
            raise ValueError("PPO observation shape differs")
        for name in ("action", "raw_action"):
            if arrays[name].shape != (count, self.action_size):
                raise ValueError("PPO action shape differs")
        if not np.allclose(np.tanh(arrays["raw_action"]), arrays["action"], atol=1e-6, rtol=1e-6):
            raise ValueError("PPO rollout must contain requested tanh(raw_action), not projected action")
        advantage_np, returns_np = compute_gae(
            arrays["reward"], arrays["value"], arrays["next_value"], arrays["terminated"],
            self.cfg.gamma, self.cfg.gae_lambda, rollout.get("stream_id"))
        component_arrays = None
        if self.cfg.variant in {'value_decomp', 'value_decomp_popart'}:
            from .value_components import compute_component_gae
            component_arrays = {name: np.asarray(rollout[name], np.float32) for name in
                                ('reward_components', 'value_components', 'next_value_components')}
            for name, array in component_arrays.items():
                if array.shape != (count, 2) or not np.isfinite(array).all():
                    raise ValueError('Invalid two-component rollout: '+name)
            for base, name in [('reward', 'reward_components'), ('value', 'value_components'),
                               ('next_value', 'next_value_components')]:
                if not np.allclose(arrays[base], component_arrays[name].sum(-1), atol=2e-5, rtol=2e-5):
                    raise ValueError('Rollout component sum differs: '+base)
            component_gae = compute_component_gae(
                component_arrays['reward_components'], component_arrays['value_components'],
                component_arrays['next_value_components'], arrays['terminated'],
                self.cfg.gamma, self.cfg.gae_lambda, rollout.get('stream_id'))
            advantage_np, returns_np = component_gae.advantages, component_gae.returns
        to_tensor = lambda array: torch.as_tensor(array, device=self.device)
        obs, raw = to_tensor(arrays["obs"]), to_tensor(arrays["raw_action"])
        old_log_prob = to_tensor(arrays["log_prob"].reshape(-1))
        old_value = to_tensor(arrays["value"].reshape(-1))
        returns, advantage = to_tensor(returns_np), to_tensor(advantage_np)
        if component_arrays is not None:
            component_returns = to_tensor(component_gae.component_returns)
            old_components = to_tensor(component_arrays['value_components'])
            regression_returns, old_regression = component_returns, old_components
        learned_continuation = (self.cfg.continuation_critic
                                and self.cfg.continuation_mode == 'learned')
        component_scale = None
        if self.cfg.variant == 'value_decomp_popart':
            # Targets and rollout values above are in original reward units.
            # Statistics updates preserve current physical-unit predictions;
            # only the critic regression loss is rescaled per component.
            if learned_continuation:
                immediate_delay = to_tensor(component_arrays['reward_components'][:, 0])
                regression_returns = self.critic.regression_targets(component_returns, immediate_delay)
                # The internal W/U split is not part of public rollout values.
                # Freeze its pre-update predictions for optional value clipping;
                # PopArt preserves them when transporting the affine head below.
                with torch.no_grad():
                    old_regression = self.critic.regression_components(obs).detach()
                self.critic.update_statistics(component_returns, self.critic_optimizer,
                                              observations=obs,
                                              immediate_delay_rewards=immediate_delay)
            else:
                self.critic.update_statistics(component_returns, self.critic_optimizer,
                                              observations=obs)
            component_scale = self.critic.target_std.to(component_returns.dtype)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        actor_weighted = {key: 0. for key in ("policy_loss", "approx_kl", "clip_fraction",
                                            "latent_entropy", "actor_grad_norm")}
        critic_weighted = {key: 0. for key in ('value_loss', 'critic_grad_norm')}
        samples_updated = minibatches_updated = epochs_completed = 0
        critic_samples_updated = critic_minibatches_updated = critic_epochs_completed = 0
        stopped_early = False
        bounded_entropy_estimate = 0.
        for _ in range(self.cfg.epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, self.cfg.minibatch_size):
                index = permutation[start:start + self.cfg.minibatch_size]
                mean, log_std = self.actor(obs[index])
                new_log_prob = self.actor.log_prob(raw[index], mean, log_std)
                log_ratio = new_log_prob - old_log_prob[index]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1.) - log_ratio).mean()
                if not torch.isfinite(approx_kl):
                    raise FloatingPointError("Non-finite PPO probability ratio")
                if self.cfg.target_kl is not None and float(approx_kl.detach()) > 1.5 * self.cfg.target_kl:
                    stopped_early = True
                    break
                policy_loss = clipped_policy_loss(new_log_prob, old_log_prob[index],
                                                  advantage[index], self.cfg.clip_ratio)
                latent_entropy = (log_std + .5 * math.log(2. * math.pi * math.e)).sum(-1).mean()
                if self.cfg.entropy_coef:
                    # The latent Normal entropy is not the bounded policy's
                    # entropy. Only a declared nonzero entropy coefficient draws
                    # this reparameterized Monte Carlo entropy sample.
                    entropy_raw = mean + log_std.exp() * torch.randn_like(mean)
                    entropy = -self.actor.log_prob(entropy_raw, mean, log_std).mean()
                    actor_loss = policy_loss - self.cfg.entropy_coef * entropy
                    bounded_entropy_estimate += float(entropy.detach()) * len(index)
                else:
                    actor_loss = policy_loss
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                diagnostics = {"policy_loss": policy_loss,
                               "approx_kl": approx_kl,
                               "clip_fraction": ((ratio - 1.).abs() > self.cfg.clip_ratio).float().mean(),
                               "latent_entropy": latent_entropy,
                               "actor_grad_norm": actor_grad}
                for key, metric in diagnostics.items():
                    actor_weighted[key] += float(metric.detach()) * len(index)
                samples_updated += len(index)
                minibatches_updated += 1
            if stopped_early:
                break
            epochs_completed += 1
        # The value estimator is independent of the actor network. Its fixed
        # fit budget must not be truncated by an actor KL stopping condition.
        # This schedule is common to original PPO and all architecture variants.
        critic_epochs = self.cfg.epochs if self.cfg.critic_epochs is None else self.cfg.critic_epochs
        for _ in range(critic_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, self.cfg.minibatch_size):
                index = permutation[start:start + self.cfg.minibatch_size]
                if component_arrays is not None:
                    from .value_components import component_value_loss
                    predicted_components = (self.critic.regression_components(obs[index])
                                            if learned_continuation
                                            else self.critic.forward_components(obs[index]))
                    value_loss = component_value_loss(
                        predicted_components, regression_returns[index],
                        old_regression[index], self.cfg.value_clip, component_scale)
                else:
                    value = self.critic(obs[index]).squeeze(-1)
                    squared_error = (value - returns[index]).square()
                    if self.cfg.value_clip is not None:
                        clipped_value = old_value[index] + (value - old_value[index]).clamp(
                            -self.cfg.value_clip, self.cfg.value_clip)
                        squared_error = torch.maximum(squared_error, (clipped_value - returns[index]).square())
                    value_loss = .5 * squared_error.mean()
                if not torch.isfinite(value_loss):
                    raise FloatingPointError('Non-finite PPO critic loss')
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.cfg.value_coef * value_loss).backward()
                critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()
                critic_weighted['value_loss'] += float(value_loss.detach()) * len(index)
                critic_weighted['critic_grad_norm'] += float(critic_grad.detach()) * len(index)
                critic_samples_updated += len(index)
                critic_minibatches_updated += 1
            critic_epochs_completed += 1
        with torch.no_grad():
            predicted = self.critic(obs).squeeze(-1)
            return_variance = returns.var(unbiased=False)
            explained_variance = (1. - (returns - predicted).var(unbiased=False)
                                  / return_variance) if float(return_variance) > 1e-12 else torch.tensor(0.)
            value_rmse = (returns - predicted).square().mean().sqrt()
            mean, log_std = self.actor(obs)
            final_log_ratio = self.actor.log_prob(raw, mean, log_std) - old_log_prob
            final_kl = ((final_log_ratio.exp() - 1.) - final_log_ratio).mean()
        self.updates += 1
        metrics = {key: value / max(samples_updated, 1) for key, value in actor_weighted.items()}
        metrics.update({key: value / max(critic_samples_updated, 1)
                        for key, value in critic_weighted.items()})
        metrics.update({"rollout_size": count, "minibatches_updated": minibatches_updated,
                        "epochs_completed": epochs_completed, "early_stopped": int(stopped_early),
                        "actor_minibatches_updated": minibatches_updated,
                        "actor_epochs_completed": epochs_completed,
                        "critic_minibatches_updated": critic_minibatches_updated,
                        "critic_epochs_completed": critic_epochs_completed,
                        "approx_kl_final": float(final_kl),
                        "value_explained_variance_gae": float(explained_variance),
                        "value_rmse_gae": float(value_rmse), "return_mean_gae": float(returns.mean()),
                        "advantage_mean_raw": float(advantage_np.mean()),
                        "advantage_std_raw": float(advantage_np.std()),
                        "latent_std_mean": float(log_std.exp().mean()),
                        "mean_action_abs": float(torch.tanh(mean).abs().mean()),
                        "action_saturation": float((to_tensor(arrays["action"]).abs() > .98).float().mean()),
                        "bounded_entropy_bonus": self.cfg.entropy_coef * bounded_entropy_estimate / max(samples_updated, 1)})
        if component_arrays is not None:
            with torch.no_grad():
                error = self.critic.forward_components(obs) - component_returns
                for i, name in enumerate(('delay', 'penalty')):
                    metrics['value_rmse_gae_'+name] = float(error[:, i].square().mean().sqrt())
                    metrics['return_mean_gae_'+name] = float(component_returns[:, i].mean())
                    metrics['return_std_gae_'+name] = float(component_returns[:, i].std(unbiased=False))
                    if component_scale is not None and not learned_continuation:
                        metrics['popart_mean_'+name] = float(self.critic.target_mean[i])
                        metrics['popart_std_'+name] = float(component_scale[i])
                        metrics['value_rmse_normalized_'+name] = float(
                            error[:, i].square().mean().sqrt() / component_scale[i])
                if learned_continuation:
                    regression_error = self.critic.regression_components(obs) - regression_returns
                    for i, name in enumerate(('current_delay', 'future_delay', 'penalty')):
                        rmse = regression_error[:, i].square().mean().sqrt()
                        metrics['popart_mean_'+name] = float(self.critic.target_mean[i])
                        metrics['popart_std_'+name] = float(component_scale[i])
                        metrics['value_rmse_gae_'+name] = float(rmse)
                        metrics['value_rmse_normalized_'+name] = float(rmse / component_scale[i])
                        metrics['return_mean_gae_'+name] = float(regression_returns[:, i].mean())
        metrics.update(self.policy_diagnostics(arrays['obs']))
        return metrics

    def state_dict(self):
        # Global training RNG and any unfinished rollout belong to the runner's
        # checkpoint; no hidden generator or rollout is owned by this learner.
        return copy.deepcopy({"version": LEARNER_VERSION, "algorithm": "ppo", "config": asdict(self.cfg),
                "observation_size": self.observation_size, "action_size": self.action_size, "physics": self.physics,
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "updates": self.updates,
                "schedule_step": self.schedule_step,
                "noise_scale": getattr(self.actor, 'noise_scale', 1.),
                "actor_lr": self.actor_optimizer.param_groups[0]['lr']})

    def load_state_dict(self, state):
        if (state["version"] != LEARNER_VERSION or state.get("algorithm") != "ppo"
                or state["config"] != asdict(self.cfg)):
            raise ValueError("PPO version, algorithm or configuration differs")
        if (state["observation_size"], state["action_size"]) != (self.observation_size, self.action_size):
            raise ValueError("PPO observation/action size differs")
        if state.get('physics') != self.physics:
            raise ValueError('PPO physical feature configuration differs')
        for name in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.updates = int(state["updates"])
        schedule_step, noise_scale, actor_lr = (state['schedule_step'], state['noise_scale'], state['actor_lr'])
        if (not isinstance(schedule_step, int) or isinstance(schedule_step, bool) or schedule_step < 0
                or not math.isfinite(actor_lr) or actor_lr <= 0):
            raise ValueError('Invalid PPO active schedule state')
        if any(group['lr'] != actor_lr for group in self.actor_optimizer.param_groups):
            raise ValueError('PPO active actor learning rate differs from optimizer state')
        self.actor.set_noise_scale(noise_scale)
        self.schedule_step = schedule_step
