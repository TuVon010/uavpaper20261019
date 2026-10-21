"""Unavailable service nodes leave the action set, not the physical channel."""
import json
import numpy as np
import pytest
from priority_mec.config import Config
from priority_mec.greedy_baseline import greedy_allocation
from priority_mec.solvers import solve


def problem():
    data = np.array([.5e6, .8e6, 1.2e6])
    work = data*500
    deadline = np.array([.7, 1.1, 1.7])
    rho = np.array([[1e10, 2e9, 3e9, 1e8],
                    [2e9, 1e10, 4e9, 2e8],
                    [5e9, 3e9, 1e10, 3e8]])
    offset = np.zeros_like(rho)
    offset[:, 3] = .004
    return data, work, deadline, rho, offset


def equal_result(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key != 'wall_s':
                equal_result(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            equal_result(x, y)
    else:
        assert left == right


def test_all_available_preserves_every_non_timing_solver_and_baseline_output():
    data, work, deadline, rho, offset = problem()
    cfg = Config()
    previous = np.array([1, 2, 3])
    original = solve(data, work, 1/deadline, rho, offset, previous, cfg,
                     deadline_s=deadline)
    masked = solve(data, work, 1/deadline, rho, offset, previous, cfg,
                   deadline_s=deadline, remote_available=np.ones(4, bool))
    equal_result(original, masked)
    equal_result(greedy_allocation(data, work, deadline, rho, offset, cfg),
                 greedy_allocation(data, work, deadline, rho, offset, cfg,
                                   remote_available=np.ones(4, bool)))


@pytest.mark.parametrize('available', [
    [False, True, True, True], [True, False, True, True],
    [True, True, False, True], [False, False, False, True],
    [True, False, False, False], [False, False, False, False]])
def test_closed_node_excluded_from_warm_start_and_all_resource_solves(available, monkeypatch):
    import priority_mec.solvers as module
    data, work, deadline, rho, offset = problem()
    cfg = Config()
    channel_copy = rho.copy()
    original = module.allocate_resources
    called = []

    def tracked(data, work, weights, rho, offset, association, cfg, **kwargs):
        for node in np.unique(association):
            assert node == 0 or available[node-1]
            called.append(int(node))
        return original(data, work, weights, rho, offset, association, cfg, **kwargs)

    monkeypatch.setattr(module, 'allocate_resources', tracked)
    for previous in (None, np.array([1, 2, 3]), np.array([4, 4, 4])):
        result = solve(data, work, 1/deadline, rho, offset, previous, cfg,
                       deadline_s=deadline, remote_available=np.array(available))
        for node in result['association']:
            assert node == 0 or available[node-1]
        # Native Python node IDs must remain JSON serializable after filtering.
        json.dumps(result['metrics'], allow_nan=False)
    baseline = greedy_allocation(data, work, deadline, rho, offset, cfg,
                                 remote_available=np.array(available))
    for node in baseline['association']:
        assert node == 0 or available[node-1]
    singleton = baseline['metrics']['singleton_delay_s']
    for row in singleton:
        for node, active in enumerate(available, start=1):
            if not active:
                assert row[node] is None
    np.testing.assert_array_equal(rho, channel_copy)


def test_all_remote_closed_preserves_late_local_tasks_and_real_delays():
    cfg = Config()
    data = np.array([1.6e6, 2e6])
    work = data*500
    deadline = np.array([.3, .4])
    rho = np.full((2, 4), 1e12)
    offset = np.zeros_like(rho)
    result = solve(data, work, 1/deadline, rho, offset, np.array([1, 4]), cfg,
                   deadline_s=deadline, remote_available=np.zeros(4, bool))
    np.testing.assert_array_equal(result['association'], [0, 0])
    np.testing.assert_array_equal(result['delay_s'], work/cfg.local_cpu_hz)
    np.testing.assert_array_equal(result['cpu_hz'], cfg.local_cpu_hz)
    np.testing.assert_array_equal(result['bandwidth_hz'], 0)
    assert not result['solver_feasible']
    assert result['metrics']['deadline_violation_count'] == 2
    assert result['metrics']['resource_solves'] == 0
    baseline = greedy_allocation(data, work, deadline, rho, offset, cfg,
                                 remote_available=np.zeros(4, bool))
    np.testing.assert_array_equal(baseline['association'], [0, 0])
    np.testing.assert_array_equal(baseline['metrics']['final_estimated_delay_s'],
                                  work/cfg.local_cpu_hz)
    assert baseline['metrics']['actual_deadline_violation_count'] == 2


def test_two_user_congestion_repair_only_uses_available_destinations(monkeypatch):
    import priority_mec.solvers as module

    def limited_node(data, cycles, weights, rho, offset, association, cfg, deadline_s=None):
        node = int(association[0])
        assert node in (1, 2)
        feasible = len(data) == 1 and rho[0, node-1] > 0
        delay = np.full(len(data), (.2 if node == 2 else .1) if feasible else .9)
        value = float(weights @ delay)
        return dict(association=association, bandwidth_hz=np.full(len(data), 1e6/len(data)),
                    cpu_hz=np.full(len(data), 1e9/len(data)), delay_s=delay,
                    value=value, lower_bound=value, gap=0., feasible=bool(feasible),
                    certified=bool(feasible), certified_infeasible=not feasible,
                    status='optimal' if feasible else 'infeasible', metrics={})

    monkeypatch.setattr(module, 'allocate_resources', limited_node)
    data = np.array([1.6e6, 1.7e6])
    work = np.full(2, .8e9)
    deadline = np.array([.4, .5])
    rho = np.array([[1e12, 1e12, 1e12, 1e12], [1e12, 0, 1e12, 1e12]])
    result = solve(data, work, 1/deadline, rho, np.zeros((2, 4)), None, Config(),
                   deadline_s=deadline, remote_available=np.array([True, True, False, False]))
    np.testing.assert_array_equal(result['association'], [2, 1])
    assert result['solver_feasible']
    assert result['metrics']['repair_pair_moves'] == 1


@pytest.mark.parametrize('invalid', [[True], [True]*5, [1, 1, 1, 1],
                                    [[True]*4], ['yes']*4])
def test_availability_requires_boolean_remote_node_vector(invalid):
    data, work, deadline, rho, offset = problem()
    with pytest.raises(ValueError, match='remote_available'):
        solve(data, work, 1/deadline, rho, offset, None, Config(),
              deadline_s=deadline, remote_available=invalid)
    with pytest.raises(ValueError, match='remote_available'):
        greedy_allocation(data, work, deadline, rho, offset, Config(),
                          remote_available=invalid)
