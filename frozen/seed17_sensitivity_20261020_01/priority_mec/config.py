"""SI-valued parameters of the independently reconstructed priority scenario."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path


@dataclass
class Config:
    num_users: int = 20
    num_uavs: int = 3
    episode_frames: int = 60
    frame_s: float = 1.0
    area_m: float = 1000.0
    altitude_m: float = 50.0
    speed_max_mps: float = 20.0
    safe_distance_m: float = 50.0
    local_cpu_hz: float = 1e9
    uav_cpu_hz: float = 10e9
    satellite_cpu_hz: float = 50e9
    uav_bandwidth_hz: float = 10e6
    satellite_bandwidth_hz: float = 100e6
    data_min_bit: float = 0.5e6
    data_max_bit: float = 2e6
    cycles_per_bit: float = 500.0
    deadline_min_s: float = 0.5
    deadline_max_s: float = 2.0
    user_speed_alpha: float = 0.6
    user_direction_alpha: float = 0.9
    user_speed_noise_variance: float = 2.0
    user_direction_noise_std_rad: float = 2.0
    tx_power_w: float = 0.2
    carrier_hz: float = 2e9
    noise_reference_dbm: float = -114.0
    noise_reference_bandwidth_hz: float = 1e6
    noise_figure_db: float = 7.0
    los_a: float = 9.61
    los_b: float = 0.16
    los_extra_db: float = 1.0
    nlos_extra_db: float = 20.0
    rician_k_db: float = 6.0
    satellite_altitude_m: float = 600e3
    satellite_carrier_hz: float = 2e9
    satellite_gt_db_k: float = 1.1
    satellite_extra_loss_db: float = 3.0
    satellite_min_elevation_deg: float = 20.0
    earth_radius_m: float = 6371e3
    earth_mu_m3_s2: float = 3.986004418e14
    earth_spin_rad_s: float = 7.2921159e-5
    energy_budget_j: float = 12000.0
    uav_joule_per_cycle: float = 1e-9
    rotor_p0_w: float = 79.86
    rotor_pi_w: float = 88.63
    rotor_tip_mps: float = 120.0
    rotor_drag_ratio: float = 0.6
    air_density: float = 1.225
    rotor_solidity: float = 0.05
    rotor_area_m2: float = 0.5
    delay_cost_weight: float = 1.0
    deadline_penalty: float = 1.0
    collision_penalty: float = 5.0
    boundary_penalty: float = 2.0
    energy_penalty: float = 2.0
    allocation_failure_penalty: float = 1.0
    failure_delay_ratio: float = 100.0
    solver_max_sweeps: int = 5
    solver_epsilon: float = 1e-7
    solver_certificate_tolerance: float = 1e-6
    feasibility_tol: float = 1e-7
    pure_route_temperature_scale: float = 5.0
    pure_resource_temperature_scale: float = 2.0
    user_cluster_halfwidth_m: float = 100.0
    user_cluster_center_x_fraction: float = .5
    user_cluster_center_y_fraction: float = .5
    greedy_direction_count: int = 16
    greedy_speed_levels: int = 4
    greedy_refine_rounds: int = 0
    uav_service_cutoff_enabled: bool = False

    @property
    def n_users(self): return self.num_users

    @property
    def n_uavs(self): return self.num_uavs

    @property
    def total_uav_bandwidth_hz(self): return self.num_uavs * self.uav_bandwidth_hz

    @property
    def noise_dbm_hz(self):
        return self.noise_reference_dbm - 10 * math.log10(self.noise_reference_bandwidth_hz)

    def validate(self):
        if isinstance(self.num_users, bool) or not isinstance(self.num_users, int) or self.num_users < 1:
            raise ValueError('num_users must be a positive integer')
        if isinstance(self.num_uavs, bool) or not isinstance(self.num_uavs, int) or self.num_uavs != 3:
            raise ValueError('This scenario fixes three UAVs')
        if not isinstance(self.uav_service_cutoff_enabled, bool):
            raise ValueError('uav_service_cutoff_enabled must be Boolean')
        for f in fields(self):
            if not math.isfinite(getattr(self, f.name)):
                raise ValueError(f"{f.name} must be finite")
        if (not isinstance(self.greedy_direction_count,int) or self.greedy_direction_count<4
                or not isinstance(self.greedy_speed_levels,int) or self.greedy_speed_levels<2
                or not isinstance(self.greedy_refine_rounds,int) or self.greedy_refine_rounds!=0):
            raise ValueError('Invalid greedy direction/speed grid or refinement count')
        center=[self.area_m*self.user_cluster_center_x_fraction,self.area_m*self.user_cluster_center_y_fraction]
        if (self.user_cluster_halfwidth_m<=0 or min(center)<self.user_cluster_halfwidth_m
                or max(center)+self.user_cluster_halfwidth_m>self.area_m):
            raise ValueError('Initial user square must lie inside the map')
        for k in ('episode_frames', 'solver_max_sweeps'):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise ValueError(f'{k} must be a positive integer')
        positive = ('frame_s','area_m','altitude_m','speed_max_mps','safe_distance_m',
                    'local_cpu_hz','uav_cpu_hz','satellite_cpu_hz','uav_bandwidth_hz',
                    'satellite_bandwidth_hz','energy_budget_j','data_min_bit','deadline_min_s',
                    'noise_reference_bandwidth_hz','carrier_hz','tx_power_w','delay_cost_weight')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('Physical capacities must be positive')
        if self.data_max_bit < self.data_min_bit or self.deadline_max_s < self.deadline_min_s:
            raise ValueError('Invalid uniform interval')
        if self.failure_delay_ratio <= 1 or self.pure_resource_temperature_scale <= 0:
            raise ValueError('Failure proxy must exceed its deadline and resource scale must be positive')
        if not 0 <= self.user_speed_alpha < 1 or not 0 <= self.user_direction_alpha < 1:
            raise ValueError('Gauss-Markov coefficients must lie in [0,1)')
        return self

    def to_dict(self): return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))).validate()
