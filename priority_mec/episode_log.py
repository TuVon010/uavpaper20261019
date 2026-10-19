"""Small, checkpointable episode summaries for vector training.

Each environment keeps only sums, final energy counters, and decision timings.
CSV I/O happens once when an episode terminates; a checkpoint also retains any
unfinished episode so resuming does not lose its early frames.
"""
from dataclasses import asdict, dataclass, field
import numpy as np


def normalized_delay(info):
    for key in ('normalized_weighted_delay', 'weighted_delay_mean', 'relative_delay'):
        if key in info:
            return float(info[key])
    return float(info['weighted_delay_sum']) / 20.


@dataclass
class EpisodeAccumulator:
    episode: int
    env_index: int
    start_step: int = 0
    frames: int = 0
    scenario_hash: str = ''
    episode_return: float = 0.
    delay_sum: float = 0.
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

    def add(self, reward, info, step, policy_wall_s=0.):
        if not self.frames:
            self.start_step = int(step)
            self.scenario_hash = str(info['scenario_hash'])
        elif self.scenario_hash != str(info['scenario_hash']):
            raise ValueError('Scenario changed inside an episode')
        self.frames += 1
        self.episode_return += float(reward)
        self.delay_sum += normalized_delay(info)
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
            start_step=self.start_step, end_step=int(end_step), frames=n,
            scenario_hash=self.scenario_hash, episode_return=self.episode_return,
            penalty_sum=-self.episode_return-self.delay_sum,
            weighted_delay_mean=delay, weighted_delay_sum_per_frame=20. * delay,
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
            total_decision_p95_s=float(np.percentile(self.decision_wall_s, 95)))

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
