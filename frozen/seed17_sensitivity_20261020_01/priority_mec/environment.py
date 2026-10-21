"""Priority-weighted delay environment with common physical accounting for all methods."""
from copy import deepcopy
import time
import numpy as np
from .config import Config
from .scenarios import Scenario, gauss_markov_update, reflect_positions
from .physics import (initial_uav_positions, safe_velocity, propulsion_power,
                      channel_parameters, link_rate)


def _softmax(x):
    x = np.asarray(x, dtype=float)
    z = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return z/np.sum(z, axis=-1, keepdims=True)


class UAVMECEnv:
    """Gym-style five-value step. No queue, no task dropping, no hidden inner energy price.

    Pure control uses 6 velocities + Nx5 route preferences + N bandwidth + N CPU
    preferences. Routes are WHOLE-task categorical draws, using an independent common
    exogenous stream. Resource normalization never invokes an optimization solver.
    """
    STATE_VERSION = 8
    CONTROLS = {'hybrid','pure','hover_random','hover_balanced','hover_solver','local','follow_solver','greedy_rate','greedy_solver'}

    def __init__(self, cfg=None, seed=1, control='hybrid'):
        self.cfg = (cfg or Config()).validate()
        self.seed, self.control = int(seed), control
        if control not in self.CONTROLS:
            raise ValueError(f'Unknown control {control}')
        self.motion_size = 2*self.cfg.num_uavs
        self.route_end = self.motion_size+self.cfg.num_users*(self.cfg.num_uavs+2)
        self.bandwidth_end = self.route_end+self.cfg.num_users
        self.action_size = self.bandwidth_end+self.cfg.num_users if control == 'pure' else self.motion_size
        self.reset(0)
        self.observation_size = len(self._observation())

    def reset(self, episode=0, **kwargs):
        if 'seed' in kwargs and kwargs['seed'] is not None:
            self.seed = int(kwargs['seed'])
        self.episode, self.frame = int(episode), 0
        self.scenario = Scenario(self.cfg, self.seed, self.episode)
        self.user_position = self.scenario.initial_position.copy()
        self.user_speed, self.user_angle = self.scenario.mean_speed.copy(), self.scenario.mean_angle.copy()
        self.uav_position = initial_uav_positions(self.cfg)
        self.uav_velocity = np.zeros((self.cfg.num_uavs,2))
        self.cumulative_energy_j = np.zeros(self.cfg.num_uavs)
        self.cumulative_propulsion_j = np.zeros(self.cfg.num_uavs)
        self.cumulative_compute_j = np.zeros(self.cfg.num_uavs)
        self.previous_association = np.zeros(self.cfg.num_users, dtype=int)
        self.history = []
        self._refresh_channel()
        return self._observation()

    def _refresh_channel(self):
        self._service_cache = None
        k = min(self.frame, self.cfg.episode_frames-1)
        self.rho_hz, self.offset_s, self.channel_info = channel_parameters(
            self.user_position, self.uav_position, k, self.scenario, self.cfg)

    def service_available(self):
        """Slot-start remote-node eligibility; flight is never disabled.

        Index j=0..2 identifies a UAV; the final entry is the satellite.
        A crossing during this slot changes eligibility only next slot.
        """
        available = np.ones(self.cfg.num_uavs+1, dtype=bool)
        if self.cfg.uav_service_cutoff_enabled:
            available[:self.cfg.num_uavs] = self.cumulative_energy_j < self.cfg.energy_budget_j
        return available

    def prepare_service(self):
        """Cache the unchanged slot-start solver, once per observed state.

        This reveals an already determined current cost to every V4 comparison
        policy. It neither moves a UAV nor advances tasks, mobility or RNG.
        A terminal has no service to prepare and a zero continuation feature.
        The returned positive cost is delay_cost_weight * J in reward units;
        the cached physical cost and the resource optimizer remain unscaled.
        """
        if self.frame >= self.cfg.episode_frames:
            return 0.
        if self.control not in {'hybrid', 'hover_solver', 'follow_solver', 'greedy_solver'}:
            raise ValueError('Known solver cost requires a solver control')
        if self._service_cache is None:
            from .solvers import solve
            started = time.perf_counter()
            c, t = self.cfg, self.frame
            data, deadline = self.scenario.data_bit[t], self.scenario.deadline_s[t]
            allocation = solve(data, data*c.cycles_per_bit, 1/deadline,
                               self.rho_hz, self.offset_s, self.previous_association,
                               c, deadline_s=deadline,
                               **({'remote_available':self.service_available()} if c.uav_service_cutoff_enabled else {}))
            solver_wall = time.perf_counter()-started
            delay = self._evaluate_allocation(allocation, data, data*c.cycles_per_bit, deadline)[3]
            self._service_cache = dict(frame=t, allocation=allocation,
                remote_available=self.service_available().tolist(),
                cost=float(np.mean(delay/deadline)), solver_wall_s=solver_wall,
                prepare_wall_s=time.perf_counter()-started)
        if self._service_cache['remote_available'] != self.service_available().tolist():
            raise RuntimeError('Stale service eligibility in prepared allocation')
        return self.cfg.delay_cost_weight*self._service_cache['cost']

    def _observation(self):
        c, n = self.cfg, self.cfg.num_users
        k = min(self.frame, c.episode_frames-1)
        one_hot = np.eye(c.num_uavs+2)[self.previous_association]
        users = np.column_stack((
            self.user_position/c.area_m,
            self.user_speed/5,
            self.user_angle/np.pi,
            self.scenario.mean_speed,
            self.scenario.mean_angle/np.pi,
            self.scenario.data_bit[k]/c.data_max_bit,
            self.scenario.deadline_s[k]/c.deadline_max_s,
            np.log1p(self.rho_hz)/30,
            self.offset_s[:, -1]/.01,
            one_hot))
        uav = np.column_stack((self.uav_position/c.area_m,
                               self.uav_velocity/c.speed_max_mps,
                               self.cumulative_energy_j/c.energy_budget_j))
        return np.concatenate((users.ravel(), uav.ravel(),
                               [1-self.frame/c.episode_frames])).astype(np.float32)

    def _simple_resources(self, association, bandwidth_preference=None, cpu_preference=None):
        c = self.cfg
        a = np.asarray(association, dtype=int)
        bandwidth = np.zeros(c.num_users)
        cpu = np.zeros(c.num_users)
        for node in range(1,c.num_uavs+2):
            budget = c.uav_bandwidth_hz if node <= c.num_uavs else c.satellite_bandwidth_hz
            indices = np.flatnonzero(a == node)
            if len(indices):
                preference = np.ones(len(indices)) if bandwidth_preference is None else _softmax(bandwidth_preference[indices])
                bandwidth[indices] = budget*preference/np.sum(preference)
        for node in range(1,c.num_uavs+2):
            indices = np.flatnonzero(a == node)
            if len(indices):
                preference = np.ones(len(indices)) if cpu_preference is None else _softmax(cpu_preference[indices])
                capacity = c.uav_cpu_hz if node <= c.num_uavs else c.satellite_cpu_hz
                cpu[indices] = capacity*preference/np.sum(preference)
        return {'association': a, 'bandwidth_hz': bandwidth, 'cpu_hz': cpu, 'metrics': {}}

    def _evaluate_allocation(self, allocation, data, cycles, deadline):
        """Evaluate submitted resources without a hidden local fallback or time truncation.

        A zero-rate/zero-CPU remote path has infinite physical latency. Only this
        numerical failure uses an explicit deadline-scaled proxy for a finite RL
        reward; the raw delay and invalid mask retain the failure evidence.
        """
        c = self.cfg
        a = np.asarray(allocation['association'], dtype=int).copy()
        b = np.asarray(allocation['bandwidth_hz'], dtype=float).copy()
        f = np.asarray(allocation['cpu_hz'], dtype=float).copy()
        if a.shape != (c.num_users,) or np.any(a < 0) or np.any(a > c.num_uavs+1):
            raise ValueError('Invalid whole-task association')
        remote_users = a > 0
        if np.any(~self.service_available()[a[remote_users]-1]):
            raise ValueError('Allocation assigned a task to a disabled UAV service')
        if b.shape != a.shape or f.shape != a.shape or np.any(~np.isfinite(b)) or np.any(~np.isfinite(f)) or np.any(b < 0) or np.any(f < 0):
            raise ValueError('Invalid resource allocation')
        tol = c.feasibility_tol
        for node in range(1,c.num_uavs+2):
            capacity = c.uav_cpu_hz if node <= c.num_uavs else c.satellite_cpu_hz
            if np.sum(f[a == node]) > capacity*(1+tol):
                raise ValueError('CPU capacity exceeded')
            budget = c.uav_bandwidth_hz if node <= c.num_uavs else c.satellite_bandwidth_hz
            if np.sum(b[a == node]) > budget*(1+tol):
                raise ValueError('Bandwidth capacity exceeded')
        delay = cycles/c.local_cpu_hz
        remote = np.flatnonzero(a > 0)
        if len(remote):
            node = a[remote]-1
            rates = link_rate(b[remote], self.rho_hz[remote,node])
            with np.errstate(divide='ignore', invalid='ignore'):
                delay[remote] = data[remote]/rates + cycles[remote]/f[remote] + self.offset_s[remote,node]
        raw_delay = delay.copy()
        invalid = ~np.isfinite(delay)
        delay[invalid] = c.failure_delay_ratio*deadline[invalid]
        b[a == 0] = 0
        f[a == 0] = 0
        return a, b, f, delay, raw_delay, invalid

    def step(self, action=None):
        c, t = self.cfg, self.frame
        if t >= c.episode_frames:
            raise RuntimeError('Episode ended; reset before stepping again')
        started = time.perf_counter()
        if action is None: action = np.zeros(self.action_size)
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or np.any(~np.isfinite(action)):
            raise ValueError(f'Action must be finite shape ({self.action_size},)')
        bounded = np.clip(action, -1, 1)
        requested = bounded[:self.motion_size].reshape(c.num_uavs, 2)*c.speed_max_mps
        motion_diagnostics = {}
        if self.control.startswith('hover') or self.control == 'local':
            requested[:] = 0
        elif self.control == 'follow_solver':
            direction = np.mean(self.user_position, axis=0)-self.uav_position
            distance = np.linalg.norm(direction, axis=1)
            requested = direction*np.minimum(1/c.frame_s, c.speed_max_mps/np.maximum(distance,1e-30))[:,None]
        elif self.control in {'greedy_rate', 'greedy_solver'}:
            from .greedy_baseline import plan_velocity
            requested, motion_diagnostics = plan_velocity(self.user_position,self.uav_position,
                1/self.scenario.deadline_s[t],self.rho_hz,c,
                direction_count=c.greedy_direction_count,
                speed_fractions=(0.,.25,.5,1.) if c.greedy_speed_levels==4 else np.linspace(0.,1.,c.greedy_speed_levels),
                refine_rounds=c.greedy_refine_rounds)
        velocity, safety = safe_velocity(self.uav_position, requested, c)
        data = self.scenario.data_bit[t]
        cycles = data*c.cycles_per_bit
        deadline = self.scenario.deadline_s[t]
        weights = 1/deadline
        available = self.service_available()
        solve_start = time.perf_counter()
        prepared_wall_s = 0.
        if self.control in {'hybrid','hover_solver','follow_solver','greedy_solver'}:
            if self._service_cache is not None:
                if self._service_cache['frame'] != t:
                    raise RuntimeError('Stale slot-start service cache')
                if self._service_cache['remote_available'] != available.tolist():
                    raise RuntimeError('Stale service eligibility in prepared allocation')
                allocation = self._service_cache['allocation']
                prepared_wall_s = self._service_cache['prepare_wall_s']
            else:
                from .solvers import solve
                allocation = solve(data, cycles, weights, self.rho_hz, self.offset_s,
                                   self.previous_association, c, deadline_s=deadline,
                                   **({'remote_available':available} if c.uav_service_cutoff_enabled else {}))
        elif self.control == 'greedy_rate':
            from .greedy_baseline import greedy_allocation
            allocation = greedy_allocation(data,cycles,deadline,self.rho_hz,self.offset_s,c,
                **({'remote_available':available} if c.uav_service_cutoff_enabled else {}))
        elif self.control == 'pure':
            score = c.pure_route_temperature_scale*bounded[self.motion_size:self.route_end].reshape(c.num_users,c.num_uavs+2)
            if c.uav_service_cutoff_enabled:
                score[:,1:] = np.where(available[None,:],score[:,1:],-np.inf)
            probability = _softmax(score)
            a = np.sum(self.scenario.route_uniform[t,:,None] > np.cumsum(probability, axis=1), axis=1)
            a = np.minimum(a, c.num_uavs+1)
            allocation = self._simple_resources(a,
                c.pure_resource_temperature_scale*bounded[self.route_end:self.bandwidth_end],
                c.pure_resource_temperature_scale*bounded[self.bandwidth_end:self.action_size])
        elif self.control == 'hover_random':
            nodes = np.flatnonzero(np.concatenate(([True],available)))
            a = nodes[np.minimum((self.scenario.route_uniform[t]*len(nodes)).astype(int), len(nodes)-1)]
            allocation = self._simple_resources(a)
        elif self.control == 'hover_balanced':
            # Only UAV nodes; indivisible tasks are balanced and rotated each slot.
            nodes = np.flatnonzero(available[:c.num_uavs])+1
            if not len(nodes): nodes = np.array([0,c.num_uavs+1])
            allocation = self._simple_resources(nodes[(np.arange(c.num_users)+t)%len(nodes)])
        else:
            allocation = self._simple_resources(np.zeros(c.num_users, dtype=int))
        solver_wall_s = time.perf_counter()-solve_start
        if prepared_wall_s:
            solver_wall_s += self._service_cache['solver_wall_s']
        a, bandwidth, cpu, delay, raw_delay, invalid = self._evaluate_allocation(allocation, data, cycles, deadline)
        before_excess = np.maximum(self.cumulative_energy_j/c.energy_budget_j-1, 0)
        propulsion = propulsion_power(np.linalg.norm(velocity,axis=1), c)*c.frame_s
        compute = np.asarray([c.uav_joule_per_cycle*np.sum(cycles[a == j+1]) for j in range(c.num_uavs)])
        energy = propulsion+compute
        self.cumulative_propulsion_j += propulsion
        self.cumulative_compute_j += compute
        self.cumulative_energy_j += energy
        newly_disabled = available[:c.num_uavs] & ~self.service_available()[:c.num_uavs]
        after_excess = np.maximum(self.cumulative_energy_j/c.energy_budget_j-1, 0)
        normalized = float(np.mean(delay/deadline))
        late = float(np.mean(np.maximum(delay/deadline-1,0)))
        energy_cost = c.energy_penalty*c.episode_frames*np.mean(after_excess-before_excess)
        delay_reward_cost = c.delay_cost_weight*normalized
        penalty_cost = c.deadline_penalty*late+energy_cost+c.collision_penalty*safety['collision_attempt']+c.boundary_penalty*safety['boundary_attempt']+c.allocation_failure_penalty*np.mean(invalid)
        reward = -delay_reward_cost-c.deadline_penalty*late-energy_cost-c.collision_penalty*safety['collision_attempt']-c.boundary_penalty*safety['boundary_attempt']-c.allocation_failure_penalty*np.mean(invalid)
        service_satellite_channel = {key: np.asarray(value).tolist() for key,value in self.channel_info.items()
            if key.startswith('satellite_')}
        old_user_position, old_uav_position = self.user_position.copy(), self.uav_position.copy()
        # Pre-generated signed innovations are applied without clipping, abs, or angle wrap.
        self.user_speed, self.user_angle = gauss_markov_update(
            self.user_speed,self.user_angle,self.scenario.mean_speed,self.scenario.mean_angle,
            self.scenario.speed_noise[t],self.scenario.angle_noise[t],c)
        movement = self.user_speed[:,None]*np.column_stack((np.cos(self.user_angle),np.sin(self.user_angle)))*c.frame_s
        self.user_position = reflect_positions(self.user_position+movement,c.area_m)
        self.uav_position = np.clip(self.uav_position+velocity*c.frame_s,0,c.area_m)
        self.uav_velocity = velocity
        self.previous_association = a.copy()
        self.frame += 1
        terminated = self.frame == c.episode_frames
        self._refresh_channel()
        info = {
            'episode': self.episode, 'frame': t, 'num_users': c.num_users, 'normalized_weighted_delay': normalized,
            'delay_cost_weight': c.delay_cost_weight,
            'delay_reward_cost': float(delay_reward_cost),
            'delay_reward_component': float(-delay_reward_cost), 'penalty_cost': float(penalty_cost),
            'weighted_delay_sum': float(np.sum(delay/deadline)), 'mean_delay_s': float(np.mean(delay)),
            'deadline_completion': float(np.mean((delay <= deadline+c.feasibility_tol)&~invalid)),
            'deadline_violation_rate': float(np.mean((delay > deadline+c.feasibility_tol)|invalid)),
            'deadline_violation_cost': late, 'energy_violation_cost': float(energy_cost),
            'energy_j': energy.tolist(), 'propulsion_energy_j': propulsion.tolist(),
            'compute_energy_j': compute.tolist(), 'total_energy_j': float(np.sum(energy)),
            'cumulative_energy_j': self.cumulative_energy_j.tolist(),
            'cumulative_propulsion_energy_j': self.cumulative_propulsion_j.tolist(),
            'cumulative_compute_energy_j': self.cumulative_compute_j.tolist(),
            'energy_budget_ok': bool(np.all(self.cumulative_energy_j <= c.energy_budget_j+c.feasibility_tol)),
            'service_cutoff_enabled': c.uav_service_cutoff_enabled,
            'service_cutoff_threshold_j': c.energy_budget_j,
            'uav_service_available': available[:c.num_uavs].tolist(),
            'uav_service_disabled': (~available[:c.num_uavs]).tolist(),
            'newly_service_disabled': newly_disabled.tolist(),
            'local_task_fraction': float(np.mean(a==0)),
            'uav_task_fraction': float(np.mean((a>0)&(a<=c.num_uavs))),
            'satellite_task_fraction': float(np.mean(a==c.num_uavs+1)),
            'scenario_hash': self.scenario.hash, 'frame_scenario_hash': self.scenario.frame_hash(t),
            'association': a.tolist(), 'association_counts': np.bincount(a,minlength=c.num_uavs+2).tolist(),
            'bandwidth_hz': bandwidth.tolist(), 'cpu_hz': cpu.tolist(), 'delay_s': delay.tolist(),
            'raw_delay_s': [float(x) if np.isfinite(x) else None for x in raw_delay],
            'delay_failure_proxy_s': np.where(invalid, delay, 0).tolist(),
            'deadline_s': deadline.tolist(), 'data_bit': data.tolist(),
            'allocation_invalid_count': int(np.sum(invalid)), 'allocation_invalid_rate': float(np.mean(invalid)),
            'allocation_invalid_mask': invalid.tolist(),
            'solver_feasible': bool(np.all((delay <= deadline+c.feasibility_tol)&~invalid)),
            'fallback_count': 0, 'fallback_rate': 0.0,
            'solver_wall_s': solver_wall_s, 'solver_metrics': allocation.get('metrics',{}),
            'prepared_service_wall_s': prepared_wall_s,
            'user_position': old_user_position.tolist(), 'uav_position': old_uav_position.tolist(),
            'motion_diagnostics': motion_diagnostics,
            'satellite_channel': service_satellite_channel,
            'next_user_position': self.user_position.tolist(), 'next_uav_position': self.uav_position.tolist(),
            'uav_velocity': velocity.tolist(), 'requested_velocity_mps': requested.tolist(),
            'user_signed_speed_mps': self.user_speed.tolist(),
            'reward': float(reward), **safety}
        info['decision_wall_s'] = time.perf_counter()-started+prepared_wall_s
        self.history.append(info)
        return self._observation(), float(reward), terminated, False, info

    def state_dict(self):
        return {'state_version': self.STATE_VERSION, 'config_fingerprint': self.cfg.fingerprint(),
                'seed': self.seed, 'control': self.control, 'episode': self.episode, 'frame': self.frame,
                'scenario_hash': self.scenario.hash, 'user_position': self.user_position.copy(),
                'user_speed': self.user_speed.copy(), 'user_angle': self.user_angle.copy(),
                'uav_position': self.uav_position.copy(), 'uav_velocity': self.uav_velocity.copy(),
                'cumulative_energy_j': self.cumulative_energy_j.copy(),
                'cumulative_propulsion_j': self.cumulative_propulsion_j.copy(),
                'cumulative_compute_j': self.cumulative_compute_j.copy(),
                'previous_association': self.previous_association.copy(), 'history': deepcopy(self.history),
                'service_cache': deepcopy(self._service_cache)}

    def load_state_dict(self, state):
        if state['state_version'] != self.STATE_VERSION or state['config_fingerprint'] != self.cfg.fingerprint():
            raise ValueError('Environment checkpoint schema/configuration mismatch')
        if state['control'] != self.control or state['seed'] != self.seed:
            raise ValueError('Environment control/seed mismatch')
        self.reset(state['episode'])
        if state['scenario_hash'] != self.scenario.hash:
            raise ValueError('Exogenous scenario differs from checkpoint')
        for k in ('frame','user_position','user_speed','user_angle','uav_position','uav_velocity',
                  'cumulative_energy_j','cumulative_propulsion_j','cumulative_compute_j','previous_association','history'):
            setattr(self,k,deepcopy(state[k]))
        self._refresh_channel()
        self._service_cache = deepcopy(state.get('service_cache'))
        if self._service_cache is not None and self._service_cache['frame'] != self.frame:
            raise ValueError('Checkpoint service cache is stale')
        if self._service_cache is not None and self._service_cache['remote_available'] != self.service_available().tolist():
            raise ValueError('Checkpoint service eligibility differs')


Environment = UAVMECEnv
