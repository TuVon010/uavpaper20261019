"""Small, checkpointable episode summaries for vector training.

Each environment keeps only sums, final energy counters, and decision timings.
CSV I/O happens once when an episode terminates; a checkpoint also retains any
unfinished episode so resuming does not lose its early frames.
"""
from dataclasses import asdict, dataclass, field
import numpy as np


def _task_count(info, default=None):
    """Read the task count from metadata or a complete per-task vector."""
    count = info.get('num_users')
    if count is None:
        for key in ('association', 'delay_s', 'deadline_s', 'data_bit'):
            if key in info:
                count = len(info[key])
                break
    if count is None:
        count = default
    if count is None or isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('A positive task count is required for weighted-delay sums')
    return int(count)


def normalized_delay(info, num_users=None):
    for key in ('normalized_weighted_delay', 'weighted_delay_mean', 'relative_delay'):
        if key in info:
            return float(info[key])
    return float(info['weighted_delay_sum']) / _task_count(info, num_users)


def delay_reward_cost(info):
    """Positive delay term in reward units; physical J is kept separately."""
    if 'delay_reward_cost' in info:
        return float(info['delay_reward_cost'])
    return float(info.get('delay_cost_weight', 1.))*normalized_delay(info)


def service_cutoff_protocol(cfg):
    """Explicit execution rule, distinct from the unchanged strict energy metric."""
    return dict(enabled=bool(getattr(cfg, 'uav_service_cutoff_enabled', False)),
        threshold_j=float(cfg.energy_budget_j),
        test='slot-start cumulative propulsion plus compute energy >= threshold',
        activation='a slot-end threshold crossing disables compute from the next slot',
        disabled_scope='UAV computation and task admission only',
        propulsion_continues=True, energy_penalty_unchanged=True,
        energy_feasibility='all final per-UAV total energies <= original budget')


@dataclass
class ServiceCutoffAccumulator:
    """Checkpointable common accounting for training and evaluation episodes.

    First-slot arrays are one-based and use None when the event never occurred.
    Threshold reaching and actual slot-start service closure are separate events:
    a threshold reached on the final slot need not create a closed-service slot.
    """
    frames: int = 0
    enabled: bool = False
    disabled_any_frames: int = 0
    per_uav_disabled_frames: list = field(default_factory=list)
    first_disabled_slot: list = field(default_factory=list)
    first_threshold_reached_slot: list = field(default_factory=list)
    local_fraction_sum: float = 0.
    uav_fraction_sum: float = 0.
    satellite_fraction_sum: float = 0.

    def add(self, info, slot=None):
        energy = np.asarray(info.get('cumulative_energy_j', []), dtype=float)
        count = len(energy)
        if not count:
            count = len(info.get('uav_service_disabled', info.get('uav_service_available', [])))
        if not self.frames:
            self.per_uav_disabled_frames = [0] * count
            self.first_disabled_slot = [None] * count
            self.first_threshold_reached_slot = [None] * count
        elif len(self.per_uav_disabled_frames) != count:
            raise ValueError('UAV count changed inside an episode service log')
        slot = self.frames + 1 if slot is None else int(slot)
        if slot != self.frames + 1:
            raise ValueError('Service log requires contiguous one-based episode slots')
        self.enabled |= bool(info.get('service_cutoff_enabled', False))
        if 'uav_service_disabled' in info:
            disabled = np.asarray(info['uav_service_disabled'], dtype=bool)
        else:
            disabled = ~np.asarray(info.get('uav_service_available', [True] * count), dtype=bool)
        if disabled.shape != (count,):
            raise ValueError('Invalid service-disabled UAV mask in episode log')
        self.disabled_any_frames += bool(np.any(disabled))
        threshold = info.get('service_cutoff_threshold_j')
        reached = (energy >= float(threshold)) if threshold is not None else np.zeros(count, dtype=bool)
        for j in range(count):
            self.per_uav_disabled_frames[j] += int(disabled[j])
            if disabled[j] and self.first_disabled_slot[j] is None:
                self.first_disabled_slot[j] = slot
            if reached[j] and self.first_threshold_reached_slot[j] is None:
                self.first_threshold_reached_slot[j] = slot
        association = np.asarray(info.get('association', []), dtype=int)
        fallback = (float(np.mean(association == 0)),
            float(np.mean((association > 0) & (association <= count))),
            float(np.mean(association == count + 1))) if len(association) else (0., 0., 0.)
        for name, default in zip(('local', 'uav', 'satellite'), fallback):
            key = name + '_fraction_sum'
            setattr(self, key, getattr(self, key) + float(info.get(name + '_task_fraction', default)))
        self.frames += 1

    def record(self):
        if not self.frames:
            raise ValueError('Cannot record an empty service episode')
        uav_frames = sum(self.per_uav_disabled_frames)
        return dict(service_cutoff_enabled=bool(self.enabled),
            service_disabled_uav_frames=uav_frames,
            service_disabled_any_frames=self.disabled_any_frames,
            service_disabled_uav_frame_fraction=uav_frames / max(1, self.frames * len(self.per_uav_disabled_frames)),
            service_disabled_any_frame_fraction=self.disabled_any_frames / self.frames,
            per_uav_service_disabled_frames=self.per_uav_disabled_frames.copy(),
            per_uav_first_service_disabled_slot=self.first_disabled_slot.copy(),
            per_uav_first_threshold_reached_slot=self.first_threshold_reached_slot.copy(),
            local_task_fraction=self.local_fraction_sum / self.frames,
            uav_task_fraction=self.uav_fraction_sum / self.frames,
            satellite_task_fraction=self.satellite_fraction_sum / self.frames)


@dataclass
class EpisodeAccumulator:
    episode: int
    env_index: int
    start_step: int = 0
    frames: int = 0
    scenario_hash: str = ''
    episode_return: float = 0.
    delay_sum: float = 0.
    delay_reward_cost_sum: float = 0.
    actual_delay_sum: float = 0.
    deadline_sum: float = 0.
    deadline_cost_sum: float = 0.
    invalid_sum: float = 0.
    solver_feasible_sum: float = 0.
    fallback_sum: float = 0.
    policy_wall_sum: float = 0.
    environment_wall_sum: float = 0.
    collision_attempt_frames: int = 0
    boundary_attempt_frames: int = 0
    projection_failure_frames: int = 0
    deadline_feasible: bool = True
    per_uav_energy_j: list = field(default_factory=list)
    per_uav_propulsion_energy_j: list = field(default_factory=list)
    per_uav_compute_energy_j: list = field(default_factory=list)
    decision_wall_s: list = field(default_factory=list)
    service_log: ServiceCutoffAccumulator = field(default_factory=ServiceCutoffAccumulator)
    num_users: int | None = None

    def add(self, reward, info, step, policy_wall_s=0.):
        # Legacy scalar-only logs have no task vectors; their historical count
        # was twenty. Current environment records always provide num_users.
        count = _task_count(info, self.num_users if self.num_users is not None else 20)
        if self.num_users is not None and self.num_users != count:
            raise ValueError('User count changed inside an episode')
        self.num_users = count
        if not self.frames:
            self.start_step = int(step)
            self.scenario_hash = str(info['scenario_hash'])
        elif self.scenario_hash != str(info['scenario_hash']):
            raise ValueError('Scenario changed inside an episode')
        self.frames += 1
        self.service_log.add(info, self.frames)
        self.episode_return += float(reward)
        self.delay_sum += normalized_delay(info, self.num_users)
        self.delay_reward_cost_sum += delay_reward_cost(info)
        self.actual_delay_sum += float(info['mean_delay_s'])
        completion = float(info['deadline_completion'])
        invalid = float(info.get('allocation_invalid_rate', 0.))
        self.deadline_sum += completion
        self.deadline_cost_sum += float(info.get('deadline_violation_cost', 0.))
        self.invalid_sum += invalid
        self.solver_feasible_sum += bool(info.get('solver_feasible', True))
        self.fallback_sum += float(info.get('fallback_rate', 0.))
        self.deadline_feasible &= completion >= 1. - 1e-12 and invalid == 0.
        self.collision_attempt_frames += bool(info['collision_attempt'])
        self.boundary_attempt_frames += bool(info['boundary_attempt'])
        self.projection_failure_frames += bool(info.get('projection_failed', False))
        self.per_uav_energy_j = [float(x) for x in info['cumulative_energy_j']]
        self.per_uav_propulsion_energy_j = [float(x) for x in info['cumulative_propulsion_energy_j']]
        self.per_uav_compute_energy_j = [float(x) for x in info['cumulative_compute_energy_j']]
        policy_wall_s = float(policy_wall_s)
        environment_wall_s = float(info.get('decision_wall_s', 0.))
        self.policy_wall_sum += policy_wall_s
        self.environment_wall_sum += environment_wall_s
        self.decision_wall_s.append(policy_wall_s + environment_wall_s)

    def record(self, end_step, energy_budget_j):
        if not self.frames:
            raise ValueError('Cannot record an empty episode')
        n = self.frames
        energy = np.asarray(self.per_uav_energy_j, dtype=float)
        budget_ok = bool(np.all(energy <= energy_budget_j + 1e-6))
        delay = self.delay_sum / n
        return dict(episode=self.episode, env_index=self.env_index,
            start_step=self.start_step, end_step=int(end_step), frames=n, num_users=self.num_users,
            scenario_hash=self.scenario_hash, episode_return=self.episode_return,
            delay_reward_cost_sum=self.delay_reward_cost_sum,
            delay_reward_cost_mean=self.delay_reward_cost_sum / n,
            delay_reward_sum=-self.delay_reward_cost_sum,
            penalty_sum=-self.episode_return-self.delay_reward_cost_sum,
            weighted_delay_mean=delay, weighted_delay_sum_per_frame=self.num_users * delay,
            mean_delay_s=self.actual_delay_sum / n,
            deadline_completion=self.deadline_sum / n,
            deadline_violation_rate=1. - self.deadline_sum / n,
            deadline_violation_cost=self.deadline_cost_sum / n,
            allocation_invalid_rate=self.invalid_sum / n,
            solver_feasible_frame_fraction=self.solver_feasible_sum / n,
            per_uav_energy_j=self.per_uav_energy_j.copy(), total_energy_j=float(energy.sum()),
            per_uav_propulsion_energy_j=self.per_uav_propulsion_energy_j.copy(),
            per_uav_compute_energy_j=self.per_uav_compute_energy_j.copy(),
            total_propulsion_energy_j=sum(self.per_uav_propulsion_energy_j),
            total_compute_energy_j=sum(self.per_uav_compute_energy_j),
            budget_feasible=budget_ok, deadline_feasible=bool(self.deadline_feasible),
            fully_feasible=budget_ok and bool(self.deadline_feasible),
            collision_attempt_frames=self.collision_attempt_frames,
            boundary_attempt_frames=self.boundary_attempt_frames,
            request_safe=self.collision_attempt_frames == 0 and self.boundary_attempt_frames == 0,
            projection_failure_frames=self.projection_failure_frames,
            fallback_rate=self.fallback_sum / n,
            policy_decision_mean_s=self.policy_wall_sum / n,
            environment_decision_mean_s=self.environment_wall_sum / n,
            total_decision_mean_s=float(np.mean(self.decision_wall_s)),
            total_decision_p95_s=float(np.percentile(self.decision_wall_s, 95)),
            **self.service_log.record())

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        if isinstance(state.get('service_log'), dict):
            state['service_log'] = ServiceCutoffAccumulator(**state['service_log'])
        return cls(**state)
