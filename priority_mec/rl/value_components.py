"""Evaluate the same policy's delay and all existing penalties separately.

Neither the environment reward nor its coefficients are changed. Component
advantages are added in original units, then normalized once by PPO.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn


class RewardDecomposedCritic(nn.Module):
    def __init__(self, observation_size, hidden=128, head_count=2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(observation_size, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.heads = nn.Linear(hidden, head_count)

    def forward_components(self, observation):
        return self.heads(self.trunk(observation))

    def forward(self, observation):
        return self.forward_components(observation).sum(-1, keepdim=True)


class PopArtRewardDecomposedCritic(RewardDecomposedCritic):
    """Two normalized targets with output-preserving physical-unit values.

    EMA moments are debiased, and their change is compensated in the final
    affine layer. PPO still receives *physical-unit* component values. Head
    Adam moments are transported by the same row scale (squared for variance),
    matching the change of normalized-loss gradients; this is not a claim that
    Adam or the shared trunk is invariant to target normalization.
    """
    def __init__(self, observation_size, hidden=128, beta=.01, min_std=.1,
                 head_count=2):
        super().__init__(observation_size, hidden, head_count)
        if not (math.isfinite(beta) and 0 < beta <= 1
                and math.isfinite(min_std) and min_std > 0):
            raise ValueError('Invalid PopArt statistics settings')
        self.beta, self.min_std = float(beta), float(min_std)
        self.register_buffer('target_mean', torch.zeros(head_count, dtype=torch.float64))
        self.register_buffer('target_second', torch.zeros(head_count, dtype=torch.float64))
        self.register_buffer('target_std', torch.ones(head_count, dtype=torch.float64))
        self.register_buffer('ema_mean', torch.zeros(head_count, dtype=torch.float64))
        self.register_buffer('ema_second', torch.zeros(head_count, dtype=torch.float64))
        self.register_buffer('ema_weight', torch.zeros((), dtype=torch.float64))
        self.register_buffer('statistics_updates', torch.zeros((), dtype=torch.int64))

    def forward_components(self, observation):
        normalized = self.heads(self.trunk(observation))
        return (normalized * self.target_std.to(normalized.dtype)
                + self.target_mean.to(normalized.dtype))

    @torch.no_grad()
    def update_statistics(self, targets, optimizer=None, observations=None):
        if targets.ndim != 2 or targets.shape[1] != self.heads.out_features or not len(targets):
            raise ValueError('PopArt targets must match the nonempty regression heads')
        if not torch.isfinite(targets).all():
            raise ValueError('Non-finite PopArt targets')
        data = targets.detach().to(device=self.target_mean.device, dtype=torch.float64)
        old_mean, old_std = self.target_mean.clone(), self.target_std.clone()
        keep = 1. - self.beta
        self.ema_mean.mul_(keep).add_(self.beta * data.mean(0))
        self.ema_second.mul_(keep).add_(self.beta * data.square().mean(0))
        self.ema_weight.mul_(keep).add_(self.beta)
        self.target_mean.copy_(self.ema_mean / self.ema_weight)
        self.target_second.copy_(self.ema_second / self.ema_weight)
        variance = (self.target_second - self.target_mean.square()).clamp_min(self.min_std ** 2)
        self.target_std.copy_(variance.sqrt())
        ratio = (old_std / self.target_std).to(self.heads.weight.dtype)
        self.heads.weight.mul_(ratio[:, None])
        self.heads.bias.copy_((old_std.to(self.heads.bias.dtype) * self.heads.bias
                              + (old_mean - self.target_mean).to(self.heads.bias.dtype))
                             / self.target_std.to(self.heads.bias.dtype))
        if optimizer is not None:
            for parameter, scale in ((self.heads.weight, ratio[:, None]),
                                     (self.heads.bias, ratio)):
                state = optimizer.state.get(parameter, {})
                for name, power in (('exp_avg', 1), ('exp_avg_sq', 2),
                                    ('max_exp_avg_sq', 2)):
                    if name in state:
                        state[name].mul_(scale ** power)
        self.statistics_updates.add_(1)


class PopArtContinuationCritic(PopArtRewardDecomposedCritic):
    """Known current delay plus learned future delay; penalties stay untouched.

    The two normalized head targets are (U_delay, V_penalty). Public values,
    GAE and actor advantages remain in original reward units, with
    V_delay(s) = -J(s) + U_delay(s). At the last service slot U is exactly zero.
    """
    def __init__(self, observation_size, hidden=128, beta=.01, min_std=.1,
                 known_delay_index=None, time_index=None, episode_frames=None):
        if (not isinstance(known_delay_index, int) or not isinstance(time_index, int)
                or not 0 <= time_index < observation_size
                or not 0 <= known_delay_index < observation_size
                or time_index == known_delay_index):
            raise ValueError('Continuation critic needs distinct valid cost and time indices')
        if not isinstance(episode_frames, int) or episode_frames < 1:
            raise ValueError('Continuation critic requires the actual positive episode horizon')
        super().__init__(observation_size, hidden, beta, min_std)
        self.known_delay_index = known_delay_index
        self.time_index = time_index
        self.episode_frames = episode_frames

    def continuation_components(self, observation):
        values = super().forward_components(observation)
        active = (observation[..., self.time_index] >
                  1. / self.episode_frames + 2. * torch.finfo(observation.dtype).eps)
        return torch.stack((values[..., 0] * active.to(values.dtype), values[..., 1]), -1)

    def forward_components(self, observation):
        continuation = self.continuation_components(observation)
        return torch.stack((continuation[..., 0] - observation[..., self.known_delay_index],
                            continuation[..., 1]), -1)

    @torch.no_grad()
    def update_statistics(self, targets, optimizer=None, observations=None):
        if observations is None or observations.shape[0] != targets.shape[0]:
            raise ValueError('Continuation statistics need their matching observations')
        future_targets = targets.detach().clone()
        future_targets[..., 0] += observations[..., self.known_delay_index].to(targets.dtype)
        return super().update_statistics(future_targets, optimizer)


class PopArtLearnedContinuationCritic(PopArtRewardDecomposedCritic):
    """State-only estimates of immediate delay, future delay, and penalties.

    A pure controller chooses resources in its action, so its realized current
    reward r_d = -J_t is unavailable before that action. The internal heads are
    W(s) = E[r_d | s], U(s) = E[sum_{k>t} gamma^(k-t) r_d,k | s], and V_p(s).
    PPO sees only the physical-unit pair [W + U, V_p]. These are state baselines;
    neither the sampled action nor its resulting delay enters their forward.

    Given the ordinary delay lambda-return R_d, the three regression targets
    are [r_d, R_d - r_d, R_p]. Thus W and U are identified separately while
    their targets sum to the same delay lambda-return used by component GAE.
    PopArt moments and affine compensation use these *transformed* targets.
    The future-delay head is masked exactly to zero at the final service slot.
    """
    def __init__(self, observation_size, hidden=128, beta=.01, min_std=.1,
                 time_index=None, episode_frames=None):
        if (not isinstance(time_index, int) or isinstance(time_index, bool)
                or not 0 <= time_index < observation_size):
            raise ValueError('Learned continuation needs a valid observed time index')
        if (not isinstance(episode_frames, int) or isinstance(episode_frames, bool)
                or episode_frames < 1):
            raise ValueError('Learned continuation requires the actual positive episode horizon')
        super().__init__(observation_size, hidden, beta, min_std, head_count=3)
        self.time_index = time_index
        self.episode_frames = episode_frames

    def regression_components(self, observation):
        """Return physical-unit [W, U, V_p], with no action/cost argument."""
        values = super().forward_components(observation)
        active = (observation[..., self.time_index] >
                  1. / self.episode_frames + 2. * torch.finfo(observation.dtype).eps)
        return torch.stack((values[..., 0], values[..., 1] * active.to(values.dtype),
                            values[..., 2]), -1)

    def forward_components(self, observation):
        values = self.regression_components(observation)
        return torch.stack((values[..., 0] + values[..., 1], values[..., 2]), -1)

    @staticmethod
    def regression_targets(component_returns, immediate_delay_rewards):
        if (component_returns.ndim != 2 or component_returns.shape[1] != 2
                or immediate_delay_rewards.shape != component_returns.shape[:1]):
            raise ValueError('Learned continuation targets need N by two returns and N rewards')
        if (not torch.isfinite(component_returns).all()
                or not torch.isfinite(immediate_delay_rewards).all()):
            raise ValueError('Non-finite learned continuation targets')
        return torch.stack((immediate_delay_rewards,
                            component_returns[:, 0] - immediate_delay_rewards,
                            component_returns[:, 1]), -1)

    @torch.no_grad()
    def update_statistics(self, targets, optimizer=None, observations=None,
                          immediate_delay_rewards=None):
        if immediate_delay_rewards is None:
            raise ValueError('Learned continuation statistics need observed immediate delay rewards')
        transformed = self.regression_targets(targets, immediate_delay_rewards)
        return super().update_statistics(transformed, optimizer)


def reward_components(infos, rewards):
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if len(infos) != len(rewards):
        raise ValueError('Reward/info lengths differ')
    delay = -np.asarray([i['normalized_weighted_delay'] for i in infos], np.float64)
    components = np.column_stack((delay, rewards-delay))
    if not np.isfinite(components).all():
        raise ValueError('Non-finite reward components')
    return components


@dataclass
class ComponentGAE:
    advantages: np.ndarray
    returns: np.ndarray
    component_advantages: np.ndarray
    component_returns: np.ndarray


def compute_component_gae(rewards, values, next_values, terminated, gamma=1.,
                          gae_lambda=.95, stream_id=None):
    from .ppo import compute_gae
    arrays = [np.asarray(x, np.float64) for x in (rewards, values, next_values)]
    if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 2 or arrays[0].shape[1] != 2:
        raise ValueError('Expected equal N by two component arrays')
    pairs = [compute_gae(arrays[0][:, k], arrays[1][:, k], arrays[2][:, k],
                         terminated, gamma, gae_lambda, stream_id) for k in range(2)]
    advantage = np.column_stack([p[0] for p in pairs])
    returns = np.column_stack([p[1] for p in pairs])
    return ComponentGAE(advantage.sum(-1), returns.sum(-1), advantage, returns)


def component_value_loss(predicted, targets, old_values=None, value_clip=None,
                         scale=None):
    squared = (predicted-targets).square()
    if value_clip is not None:
        if old_values is None:
            raise ValueError('Clipped component loss requires old values')
        clipped = old_values+(predicted-old_values).clamp(-value_clip, value_clip)
        squared = torch.maximum(squared, (clipped-targets).square())
    if scale is not None:
        if (scale.shape != (predicted.shape[-1],) or not torch.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError('Component scale must be finite, positive, and match the heads')
        squared = squared / scale.square()
    return .5*squared.mean()
