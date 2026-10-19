"""Deterministic rate-greedy flight and priority-greedy resource baseline.

This module deliberately does not import the association or KKT solvers. Flight
uses a declared finite direction/speed search, not a continuous global optimum.
Counterfactual channel prediction rescales *observed* CSI by geometric free-space
distance loss; it neither reads hidden channel innovations nor draws randomness.
"""
import math
import numpy as np

from .physics import link_rate, safe_velocity


def _bounded_velocity_batch(position, proposals, cfg):
    proposals = np.asarray(proposals, float).copy()
    norm = np.linalg.norm(proposals, axis=-1)
    proposals *= np.minimum(1., cfg.speed_max_mps/np.maximum(norm, 1e-30))[..., None]
    return (np.clip(position[None]+cfg.frame_s*proposals, 0., cfg.area_m)-position[None])/cfg.frame_s


def _continuous_safe_batch(position, velocities, cfg):
    safe = np.ones(len(velocities), dtype=bool)
    for a in range(cfg.num_uavs):
        for b in range(a):
            r = position[a]-position[b]
            relative = velocities[:, a]-velocities[:, b]
            squared = np.sum(relative*relative, axis=1)
            tau = np.clip(-np.sum(r*relative, axis=1)/np.maximum(squared, 1e-300), 0., cfg.frame_s)
            safe &= np.linalg.norm(r[None]+tau[:, None]*relative, axis=1) >= cfg.safe_distance_m-1e-10
    return safe


def _safe_velocity_batch(position, proposals, cfg):
    """Exactly share the environment's projection; solve only unsafe members.

    Safe members are checked vectorially; only conflicting members invoke the
    small convex solve. This preserves projected candidates in the greedy search
    rather than weaken it by discarding them.
    """
    executed = _bounded_velocity_batch(position, proposals, cfg)
    valid = _continuous_safe_batch(position, executed, cfg)
    for index in np.flatnonzero(~valid):
        executed[index] = safe_velocity(position, proposals[index], cfg)[0]
    return executed


def plan_velocity(user_position, uav_position, weights, rho_hz, cfg, *,
                  direction_count=72, speed_fractions=(0., .25, .5, .75, 1.),
                  refine_rounds=2):
    """Return ``(safe_velocity_mps, diagnostics)`` by UAV-index coordinate search.

    Each UAV maximizes ``sum_i w_i R_ij(B_j)`` using its full standalone bandwidth
    as a common motion score, independent of the later greedy resource decisions.
    Current excess loss/fading are frozen while inverse-square geometry changes.
    Candidate joint trajectories use the same speed/boundary bounds and
    continuous-separation projection as the environment. Safe batch members
    bypass the convex solve; conflicting candidates are repaired before scoring. A deterministic tie favors the lower joint speed norm. The finite
    search includes hover; refinement searches locally in both speed and angle.
    """
    users, position = np.asarray(user_position, float), np.asarray(uav_position, float)
    weights, rho = np.asarray(weights, float), np.asarray(rho_hz, float)
    if position.shape != (cfg.num_uavs, 2) or users.ndim != 2 or users.shape[1] != 2:
        raise ValueError('Expected users [N,2] and UAVs [J,2]')
    if weights.shape != (len(users),) or rho.ndim != 2 or rho.shape[0] != len(users) or rho.shape[1] < cfg.num_uavs:
        raise ValueError('Inconsistent user weights or channel shape')
    if np.any(~np.isfinite(users)) or np.any(~np.isfinite(position)) or np.any(~np.isfinite(weights)) or np.any(weights <= 0) or np.any(~np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError('Invalid positions, weights, or observed channel')
    fractions = np.unique(np.asarray(speed_fractions, float))
    if not isinstance(direction_count, int) or direction_count < 4 or not isinstance(refine_rounds, int) or refine_rounds < 0 or fractions.ndim != 1 or len(fractions) == 0 or np.any(~np.isfinite(fractions)) or np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError('Invalid finite search grid')
    fractions = np.unique(np.concatenate(([0.], fractions)))
    angles = np.arange(direction_count) * (2*np.pi/direction_count)
    current_distance2 = np.sum((users[:, None, :]-position[None, :, :])**2, axis=-1)+cfg.altitude_m**2
    chosen = np.zeros_like(position)
    objective_before, objective_after, evaluated = [], [], 0

    for j in range(cfg.num_uavs):
        baseline_rho = rho[:, j]

        def objective(velocity):
            q = position[j]+cfg.frame_s*velocity[j]
            future_distance2 = np.sum((users-q)**2, axis=1)+cfg.altitude_m**2
            prediction = baseline_rho * current_distance2[:, j]/future_distance2
            return float(np.dot(weights, link_rate(cfg.uav_bandwidth_hz, prediction)))

        best, best_score = chosen.copy(), objective(chosen)
        objective_before.append(best_score)

        def consider_batch(speeds, trial_angles):
            nonlocal best, best_score, evaluated
            speeds, trial_angles = np.broadcast_arrays(speeds, trial_angles)
            proposals = np.broadcast_to(chosen, (len(speeds),)+chosen.shape).copy()
            proposals[:, j, 0] = speeds*np.cos(trial_angles)
            proposals[:, j, 1] = speeds*np.sin(trial_angles)
            executed = _safe_velocity_batch(position, proposals, cfg)
            q = position[j]+cfg.frame_s*executed[:, j]
            future_distance2 = np.sum((users[None]-q[:, None])**2, axis=-1)+cfg.altitude_m**2
            prediction = baseline_rho[None]*current_distance2[None, :, j]/future_distance2
            scores = np.sum(weights[None]*link_rate(cfg.uav_bandwidth_hz, prediction), axis=1)
            evaluated += len(speeds)
            maximum = max(best_score, float(np.max(scores)))
            tolerance = 1e-12*max(1., abs(maximum))
            finalists = np.flatnonzero(scores >= maximum-tolerance)
            if len(finalists):
                energy_proxy = np.sum(executed[finalists]**2, axis=(1, 2))
                index = int(finalists[np.argmin(energy_proxy)])
                score = float(scores[index])
                if score > best_score+tolerance or (abs(score-best_score) <= tolerance and np.sum(executed[index]**2) < np.sum(best**2)-1e-12):
                    best, best_score = executed[index].copy(), score

        grid_speed, grid_angle = np.meshgrid(fractions[fractions > 0]*cfg.speed_max_mps, angles, indexing='ij')
        consider_batch(np.concatenate(([0.], grid_speed.ravel())),
                       np.concatenate(([0.], grid_angle.ravel())))
        angle_step = 2*np.pi/direction_count
        speed_step = cfg.speed_max_mps * (float(np.min(np.diff(fractions))) if len(fractions) > 1 else .25)
        for _ in range(refine_rounds):
            center_angle = math.atan2(best[j, 1], best[j, 0])
            center_speed = float(np.linalg.norm(best[j]))
            angle_step *= .5
            speed_step *= .5
            local_speed, local_angle = np.meshgrid(
                np.unique(np.clip(center_speed+speed_step*np.arange(-2, 3), 0., cfg.speed_max_mps)),
                center_angle+angle_step*np.arange(-2, 3), indexing='ij')
            consider_batch(local_speed.ravel(), local_angle.ravel())
        chosen = best
        objective_after.append(best_score)

    chosen, safety = safe_velocity(position, chosen, cfg)
    return chosen, {
        'motion_rule': 'observed_csi_inverse_square_rate_coordinate_grid',
        'direction_count': direction_count,
        'speed_fractions': fractions.tolist(),
        'refine_rounds': refine_rounds,
        'candidates_evaluated': evaluated,
        'coordinate_score_before': objective_before,
        'coordinate_score_after': objective_after,
        'final_safety': safety,
    }


def greedy_allocation(data_bit, cycles, deadline_s, rho_hz, offset_s, cfg):
    """Priority-order whole-task routing with explicit remaining-resource quotas.

    At processing step k, r tasks remain and K=J+1 remote nodes are available.
    A candidate receives 1/ceil(r/K) of that node's remaining bandwidth and CPU.
    This is an equal-demand reservation heuristic, not a KKT solution. Local CPU
    is private. Among deadline-feasible candidates choose minimum delay; if none
    is feasible choose minimum delay without hiding the deadline violation.

    Assigned remote quotas are debited immediately. Finally, any unused resource
    of a used node is returned proportionally to its already granted quotas; this
    can only lower completion times and does not reroute tasks or call a solver.
    """
    data, work, deadline = (np.asarray(x, float) for x in (data_bit, cycles, deadline_s))
    rho, offset = np.asarray(rho_hz, float), np.asarray(offset_s, float)
    n, nodes = len(data), cfg.num_uavs+1
    if data.shape != (n,) or work.shape != (n,) or deadline.shape != (n,) or rho.shape != (n, nodes) or offset.shape != rho.shape:
        raise ValueError('Inconsistent task/channel shapes')
    if np.any(~np.isfinite(data)) or np.any(data <= 0) or np.any(~np.isfinite(work)) or np.any(work <= 0) or np.any(~np.isfinite(deadline)) or np.any(deadline <= 0) or np.any(~np.isfinite(rho)) or np.any(rho < 0) or np.any(~np.isfinite(offset)) or np.any(offset < 0):
        raise ValueError('Tasks and links must be finite and nonnegative')
    capacities_b = np.array([cfg.uav_bandwidth_hz]*cfg.num_uavs+[cfg.satellite_bandwidth_hz], float)
    capacities_f = np.array([cfg.uav_cpu_hz]*cfg.num_uavs+[cfg.satellite_cpu_hz], float)
    remaining_b, remaining_f = capacities_b.copy(), capacities_f.copy()
    association = np.zeros(n, int)
    bandwidth, cpu = np.zeros(n), np.zeros(n)
    order = np.argsort(-1/deadline, kind='stable')
    estimated_delay = np.zeros(n)
    quota_divisors = np.zeros(n, int)
    no_feasible = 0
    for step, i in enumerate(order):
        reserved_count = int(math.ceil((n-step)/nodes))
        quota_divisors[i] = reserved_count
        offer_b, offer_f = remaining_b/reserved_count, remaining_f/reserved_count
        rates = link_rate(offer_b, rho[i])
        remote_delay = np.full(nodes, np.inf)
        possible = (rates > 0) & (offer_f > 0)
        remote_delay[possible] = data[i]/rates[possible]+work[i]/offer_f[possible]+offset[i, possible]
        delay = np.concatenate(([work[i]/cfg.local_cpu_hz], remote_delay))
        candidates = np.flatnonzero(delay <= deadline[i]+cfg.feasibility_tol)
        if len(candidates) == 0:
            candidates = np.arange(nodes+1)
            no_feasible += 1
        node = int(candidates[np.argmin(delay[candidates])])
        association[i], estimated_delay[i] = node, delay[node]
        if node:
            bandwidth[i], cpu[i] = offer_b[node-1], offer_f[node-1]
            remaining_b[node-1] -= bandwidth[i]
            remaining_f[node-1] -= cpu[i]

    reserved_bandwidth, reserved_cpu = bandwidth.copy(), cpu.copy()
    for node in range(1, nodes+1):
        selected = association == node
        if np.any(selected):
            bandwidth[selected] *= capacities_b[node-1]/np.sum(bandwidth[selected])
            cpu[selected] *= capacities_f[node-1]/np.sum(cpu[selected])
    final_delay = work/cfg.local_cpu_hz
    remote = np.flatnonzero(association > 0)
    if len(remote):
        indices = association[remote]-1
        final_delay[remote] = (data[remote]/link_rate(bandwidth[remote], rho[remote, indices])
                               + work[remote]/cpu[remote] + offset[remote, indices])
    return {
        'association': association,
        'bandwidth_hz': bandwidth,
        'cpu_hz': cpu,
        'metrics': {
            'resource_rule': 'remaining_equal_demand_quota_then_proportional_spare_return',
            'order': order.tolist(),
            'quota_divisors_by_user': quota_divisors.tolist(),
            'initial_estimated_delay_s': estimated_delay.tolist(),
            'final_estimated_delay_s': final_delay.tolist(),
            'no_deadline_feasible_candidate_count': no_feasible,
            'reserved_bandwidth_hz': reserved_bandwidth.tolist(),
            'reserved_cpu_hz': reserved_cpu.tolist(),
            'unused_bandwidth_before_return_hz': remaining_b.tolist(),
            'unused_cpu_before_return_hz': remaining_f.tolist(),
        },
    }
