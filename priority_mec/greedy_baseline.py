"""Independent coarse rate search and singleton-route/equal-share baseline.

All channel predictions freeze the observed fading and rescale by geometry.
Motion scoring never observes other UAV positions, future draws or battery use.
The environment applies the common joint safety projection AFTER selection.
"""
import numpy as np
from .physics import link_rate

BASELINE_VERSION = 2


def plan_velocity(user_position, uav_position, weights, rho_hz, cfg, *,
                  direction_count=16, speed_fractions=(0., .25, .5, 1.),
                  refine_rounds=0):
    """Return independent requested velocities, before joint collision repair.

    Sixteen angles and three positive speed levels produce 48 nonzero actions;
    hover is one additional action, for 49 distinct requested actions per UAV.
    Boundary clipping uses only that UAV's position. The other UAVs do not enter
    its score. Ties prefer lower speed, then the fixed angle enumeration order.
    ``refine_rounds`` is retained as an explicit compatibility/validation input;
    only zero is supported by this baseline definition.
    """
    users, position = np.asarray(user_position, float), np.asarray(uav_position, float)
    weight, rho = np.asarray(weights, float), np.asarray(rho_hz, float)
    if position.shape != (cfg.num_uavs, 2) or users.ndim != 2 or users.shape[1] != 2:
        raise ValueError('Expected users [N,2] and UAVs [J,2]')
    if weight.shape != (len(users),) or rho.ndim != 2 or rho.shape[0] != len(users) or rho.shape[1] < cfg.num_uavs:
        raise ValueError('Inconsistent user weights or channel shape')
    if (any(not np.isfinite(x).all() for x in (users, position, weight, rho))
            or np.any(weight <= 0) or np.any(rho < 0)):
        raise ValueError('Invalid positions, weights or observed channel')
    fractions = np.asarray(speed_fractions, float)
    if (isinstance(direction_count, bool) or not isinstance(direction_count, int)
            or direction_count < 4 or refine_rounds != 0 or fractions.ndim != 1
            or not len(fractions) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or np.any(fractions > 1)):
        raise ValueError('Invalid coarse grid; local refinement is disabled')
    fractions = np.unique(np.concatenate(([0.], fractions)))
    angles = np.arange(direction_count)*2*np.pi/direction_count
    positive = fractions[fractions > 0]*cfg.speed_max_mps
    speeds = np.concatenate(([0.], np.repeat(positive, direction_count)))
    headings = np.concatenate(([0.], np.tile(angles, len(positive))))
    candidates = speeds[:, None]*np.column_stack((np.cos(headings), np.sin(headings)))
    chosen = np.zeros_like(position)
    before, after, selected = [], [], []
    for j in range(cfg.num_uavs):
        current_distance2 = np.sum((users-position[j])**2, axis=1)+cfg.altitude_m**2
        endpoints = np.clip(position[j]+cfg.frame_s*candidates, 0., cfg.area_m)
        velocities = (endpoints-position[j])/cfg.frame_s
        distance2 = np.sum((users[None]-endpoints[:, None])**2, axis=-1)+cfg.altitude_m**2
        prediction = rho[None, :, j]*current_distance2[None]/distance2
        scores = np.sum(weight[None]*link_rate(cfg.uav_bandwidth_hz, prediction), axis=1)
        maximum = float(scores.max())
        finalists = np.flatnonzero(scores >= maximum-1e-12*max(1., abs(maximum)))
        index = int(finalists[np.argmin(np.sum(velocities[finalists]**2, axis=1))])
        chosen[j] = velocities[index]
        before.append(float(scores[0])); after.append(float(scores[index])); selected.append(index)
    return chosen, dict(baseline_version=BASELINE_VERSION,
        motion_rule='independent_observed_csi_inverse_square_coarse_rate',
        direction_count=direction_count, speed_fractions=fractions.tolist(),
        speed_levels_mps=(fractions*cfg.speed_max_mps).tolist(), refine_rounds=0,
        nonzero_candidates_per_uav=int(len(positive)*direction_count),
        distinct_candidates_per_uav=int(len(candidates)),
        candidates_evaluated=int(cfg.num_uavs*len(candidates)),
        selected_candidate_index=selected, score_before=before, score_after=after,
        other_uavs_considered_during_selection=False,
        joint_safety_projection='applied once by environment after independent selection')


def greedy_allocation(data_bit, cycles, deadline_s, rho_hz, offset_s, cfg,
                      remote_available=None):
    """Choose each task's fastest singleton node, then equal-share once.

    Deadline order is recorded and used to traverse tasks. Since no resource is
    debited during routing, it does not affect the final association or shares.
    There is no congestion repair, deadline filtering or resource optimization.
    Shared-resource delays, including true deadline misses, are evaluated after
    every task has selected exactly one of local/three UAVs/satellite.
    Closed remote nodes are excluded before singleton selection; local always
    remains selectable. No fake channel degradation is used to disable service.
    """
    data, work, deadline = (np.asarray(x, float) for x in (data_bit, cycles, deadline_s))
    rho, offset = np.asarray(rho_hz, float), np.asarray(offset_s, float)
    n, nodes = len(data), cfg.num_uavs+1
    if data.shape != (n,) or work.shape != (n,) or deadline.shape != (n,) or rho.shape != (n, nodes) or offset.shape != rho.shape:
        raise ValueError('Inconsistent task/channel shapes')
    if (any(not np.isfinite(x).all() for x in (data, work, deadline, rho, offset))
            or np.any(data <= 0) or np.any(work <= 0) or np.any(deadline <= 0)
            or np.any(rho < 0) or np.any(offset < 0)):
        raise ValueError('Tasks and links must be finite and nonnegative')
    available = np.ones(nodes, dtype=bool) if remote_available is None else np.asarray(remote_available)
    if available.shape != (nodes,) or available.dtype.kind != 'b':
        raise ValueError('remote_available must be a boolean vector of length num_uavs+1')
    capacity_b = np.array([cfg.uav_bandwidth_hz]*cfg.num_uavs+[cfg.satellite_bandwidth_hz])
    capacity_f = np.array([cfg.uav_cpu_hz]*cfg.num_uavs+[cfg.satellite_cpu_hz])
    rates = link_rate(capacity_b[None], rho)
    remote = np.full((n, nodes), np.inf)
    np.divide(data[:, None], rates, out=remote, where=rates > 0)
    remote += work[:, None]/capacity_f[None]+offset
    remote[:, ~available] = np.inf
    singleton = np.column_stack((work/cfg.local_cpu_hz, remote))
    order = np.argsort(deadline, kind='stable')
    association = np.zeros(n, int)
    for i in order:
        association[i] = int(np.argmin(singleton[i]))
    bandwidth, cpu = np.zeros(n), np.zeros(n)
    final = work/cfg.local_cpu_hz
    for node in range(1, nodes+1):
        users = np.flatnonzero(association == node)
        if not len(users):
            continue
        bandwidth[users] = capacity_b[node-1]/len(users)
        cpu[users] = capacity_f[node-1]/len(users)
        final[users] = data[users]/link_rate(bandwidth[users], rho[users, node-1])+work[users]/cpu[users]+offset[users, node-1]
    counts = np.bincount(association, minlength=nodes+1)
    return dict(association=association, bandwidth_hz=bandwidth, cpu_hz=cpu,
        metrics=dict(baseline_version=BASELINE_VERSION,
            resource_rule='singleton_min_delay_routing_then_one_shot_equal_share',
            order=order.tolist(), singleton_delay_s=[
                [float(x) if np.isfinite(x) else None for x in row] for row in singleton],
            initial_estimated_delay_s=singleton[np.arange(n), association].tolist(),
            final_estimated_delay_s=final.tolist(), association_counts=counts.tolist(),
            no_deadline_feasible_candidate_count=int(np.sum(np.min(singleton, axis=1) > deadline+cfg.feasibility_tol)),
            actual_deadline_violation_count=int(np.sum(final > deadline+cfg.feasibility_tol)),
            deadline_order_changes_allocation=False, congestion_repair=False,
            spare_return=False, allocation_iterations=1))
