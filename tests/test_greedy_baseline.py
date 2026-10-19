"""The greedy comparator obeys capacities and does not use optimizer fallback."""
import inspect
import inspect
import numpy as np
import pytest

from priority_mec.config import Config
from priority_mec.greedy_baseline import greedy_allocation, plan_velocity, _safe_velocity_batch
from priority_mec.physics import initial_uav_positions, minimum_separation, safe_velocity


def _tasks(c, n=20):
    data = np.linspace(1e6, 2e6, n)
    deadline = np.linspace(.5, 2., n)[::-1]
    rho = np.full((n, c.num_uavs+1), 2e9)
    return data, data*c.cycles_per_bit, deadline, rho, np.zeros_like(rho)


def test_priority_route_quotas_and_actual_capacity():
    c = Config()
    data, work, deadline, rho, offset = _tasks(c)
    result = greedy_allocation(data, work, deadline, rho, offset, c)
    a, b, f = (result[key] for key in ('association', 'bandwidth_hz', 'cpu_hz'))
    assert result['metrics']['order'] == list(range(len(data)-1, -1, -1))
    for node in range(1, c.num_uavs+2):
        mask = a == node
        if np.any(mask):
            assert b[mask].sum() == pytest.approx(c.uav_bandwidth_hz if node <= c.num_uavs else c.satellite_bandwidth_hz)
            assert f[mask].sum() == pytest.approx(c.uav_cpu_hz if node <= c.num_uavs else c.satellite_cpu_hz)
    initial = np.asarray(result['metrics']['initial_estimated_delay_s'])
    final = np.asarray(result['metrics']['final_estimated_delay_s'])
    assert np.all(final <= initial+1e-12)
    assert np.all(np.isfinite(final))
    # The first selected remote task does not consume an entire node's resources.
    first = result['metrics']['order'][0]
    assert result['metrics']['reserved_cpu_hz'][first] <= max(c.uav_cpu_hz, c.satellite_cpu_hz)/5


def test_failed_links_local_fallback_is_explicit_and_not_dropped():
    c = Config()
    data, work, deadline, rho, offset = _tasks(c)
    result = greedy_allocation(data, work, deadline, np.zeros_like(rho), offset, c)
    np.testing.assert_array_equal(result['association'], 0)
    np.testing.assert_array_equal(result['cpu_hz'], 0)
    np.testing.assert_allclose(result['metrics']['final_estimated_delay_s'], work/c.local_cpu_hz)


def test_infeasible_deadline_is_retained_not_replaced():
    c = Config()
    data, work, deadline, rho, offset = _tasks(c)
    deadline[:] = .0001
    result = greedy_allocation(data, work, deadline, rho, offset, c)
    assert result['metrics']['no_deadline_feasible_candidate_count'] == len(data)
    assert np.all(np.asarray(result['metrics']['final_estimated_delay_s']) > deadline)


def test_grid_motion_deterministic_safe_and_no_hidden_random():
    c = Config()
    c.speed_max_mps = 20.
    users = np.full((c.num_users, 2), 500.)
    initial = initial_uav_positions(c)
    rho = np.full((c.num_users, c.num_uavs+1), 1e7)
    arguments = (users, initial, np.ones(c.num_users), rho, c)
    state = np.random.get_state()
    first, diagnostics = plan_velocity(*arguments, direction_count=24, refine_rounds=1)
    second, _ = plan_velocity(*arguments, direction_count=24, refine_rounds=1)
    after = np.random.get_state()
    np.testing.assert_array_equal(first, second)
    for a, b in zip(state, after):
        np.testing.assert_array_equal(a, b)
    assert np.all(np.linalg.norm(first, axis=1) <= c.speed_max_mps+1e-12)
    assert minimum_separation(initial, first, c.frame_s) >= c.safe_distance_m-1e-8
    assert np.all(initial+first*c.frame_s >= 0)
    assert np.all(initial+first*c.frame_s <= c.area_m)
    # With symmetric central users, each UAV moves toward the center.
    assert np.all(np.linalg.norm(initial+first*c.frame_s-500, axis=1) < 500)
    assert np.all(np.asarray(diagnostics['coordinate_score_after']) >= np.asarray(diagnostics['coordinate_score_before'])-1e-5)


def test_zero_rate_motion_tie_prefers_hover():
    c = Config()
    users = np.full((c.num_users, 2), 500.)
    velocity, _ = plan_velocity(users, initial_uav_positions(c), np.ones(c.num_users),
                                np.zeros((c.num_users, c.num_uavs+1)), c,
                                direction_count=8, refine_rounds=0)
    np.testing.assert_array_equal(velocity, 0)


def test_collision_and_boundary_search_keeps_full_segment_safe():
    c = Config()
    c.speed_max_mps = 20.
    initial = np.array([[475., 500.], [525., 500.], [999., 999.]])
    users = np.full((c.num_users, 2), 500.)
    rho = np.full((c.num_users, c.num_uavs+1), 1e8)
    velocity, _ = plan_velocity(users, initial, np.ones(c.num_users), rho, c,
                                direction_count=12, speed_fractions=(0., 1.), refine_rounds=1)
    assert minimum_separation(initial, velocity, c.frame_s) >= c.safe_distance_m-1e-8
    assert np.all(initial+velocity*c.frame_s >= 0)
    assert np.all(initial+velocity*c.frame_s <= c.area_m)


def test_random_link_resource_conservation_and_explicit_estimate_accounting():
    c = Config()
    rng = np.random.default_rng(145)
    data, work, deadline, _, offset = _tasks(c)
    for _ in range(10):
        rho = np.exp(rng.uniform(-10., 27., (len(data), c.num_uavs+1)))
        result = greedy_allocation(data, work, deadline, rho, offset, c)
        a, b, f = (result[key] for key in ('association', 'bandwidth_hz', 'cpu_hz'))
        assert np.all(b >= 0) and np.all(f >= 0)
        assert np.all(b[a == 0] == 0) and np.all(f[a == 0] == 0)
        for node in range(1, c.num_uavs+2):
            mask = a == node
            capacity_b = c.uav_bandwidth_hz if node <= c.num_uavs else c.satellite_bandwidth_hz
            capacity_f = c.uav_cpu_hz if node <= c.num_uavs else c.satellite_cpu_hz
            assert b[mask].sum() <= capacity_b*(1+1e-12)
            assert f[mask].sum() <= capacity_f*(1+1e-12)
        initial = np.asarray(result['metrics']['initial_estimated_delay_s'])
        final = np.asarray(result['metrics']['final_estimated_delay_s'])
        assert np.all(final <= initial+1e-12)


def test_analytic_batch_prefix_matches_common_safety_random_boundary_and_tangent():
    c = Config()
    c.speed_max_mps = 20.
    rng = np.random.default_rng(728)
    starts = [
        initial_uav_positions(c),
        np.array([[0., 0.], [50., 0.], [1000., 1000.]]),
        np.array([[475., 500.], [525., 500.], [500., 500.+25*np.sqrt(3)]]),
        np.array([[475., 500.], [525.+1e-9, 500.], [800., 800.]]),
        np.array([[475., 500.], [525.-1e-12, 500.], [800., 800.]]),
    ]
    for _ in range(5):
        while True:
            start = rng.uniform(0., c.area_m, (3, 2))
            if minimum_separation(start, np.zeros((3, 2)), 1.) >= c.safe_distance_m:
                starts.append(start)
                break
    for start in starts:
        proposals = rng.normal(0., 30., (40, 3, 2))
        # Exact tangency and inward contact are frequent hard cases.
        proposals = np.concatenate((proposals, np.array([
            [[0., 20.], [0., -20.], [0., 0.]],
            [[20., 0.], [-20., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.]],
        ])))
        expected = np.stack([safe_velocity(start, request, c)[0] for request in proposals])
        actual = _safe_velocity_batch(start, proposals, c)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-9)


def test_solver_independence_and_input_validation():
    import priority_mec.greedy_baseline as module
    assert 'from .solvers' not in inspect.getsource(module)
    assert 'from .resources' not in inspect.getsource(module)
    c = Config()
    data, work, deadline, rho, offset = _tasks(c)
    with pytest.raises(ValueError):
        greedy_allocation(data, work, deadline, -rho, offset, c)
    with pytest.raises(ValueError):
        plan_velocity(np.zeros((20, 2)), initial_uav_positions(c), np.ones(20), rho, c,
                      direction_count=0)
