"""Slot-start service shutdown preserves flight and all twenty whole tasks."""
from dataclasses import replace

import numpy as np
import pytest

from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.physics import propulsion_power


def cutoff_config(**overrides):
    return replace(Config(), uav_service_cutoff_enabled=True, **overrides)


def prefer_first_uav(env):
    action = np.zeros(env.action_size)
    action[6:106] = np.tile([-1., 1., -1., -1., -1.], env.cfg.num_users)
    return action


def assert_available_allocation(info, cfg):
    association = np.asarray(info['association'])
    available = np.asarray(info['uav_service_available'], dtype=bool)
    assert association.shape == (cfg.num_users,)
    assert np.sum(info['association_counts']) == cfg.num_users
    for node in range(1, cfg.num_uavs + 1):
        if not available[node - 1]:
            assert not np.any(association == node)
            assert info['compute_energy_j'][node - 1] == 0.
    for node in range(1, cfg.num_uavs + 2):
        capacity = cfg.uav_cpu_hz if node <= cfg.num_uavs else cfg.satellite_cpu_hz
        bandwidth = cfg.uav_bandwidth_hz if node <= cfg.num_uavs else cfg.satellite_bandwidth_hz
        assert np.sum(np.asarray(info['cpu_hz'])[association == node]) <= capacity * (1 + cfg.feasibility_tol)
        assert np.sum(np.asarray(info['bandwidth_hz'])[association == node]) <= bandwidth * (1 + cfg.feasibility_tol)


def test_flag_defaults_off_and_rejects_non_boolean_values():
    assert Config().uav_service_cutoff_enabled is False
    for invalid in (1, 'true', None):
        with pytest.raises(ValueError, match='Boolean'):
            replace(Config(), uav_service_cutoff_enabled=invalid).validate()


def test_threshold_is_exact_slot_start_energy_and_satellite_is_always_available():
    env = UAVMECEnv(cutoff_config(), control='pure')
    budget = env.cfg.energy_budget_j
    env.cumulative_energy_j[:] = [np.nextafter(budget, -np.inf), budget, budget + 1.]
    available = env.service_available()
    np.testing.assert_array_equal(available, [True, False, False, True])
    assert available.dtype == np.bool_
    available[:] = False
    np.testing.assert_array_equal(env.service_available(), [True, False, False, True])
    # The cutoff is optional: old parameter sets retain their old service behavior.
    old = UAVMECEnv(Config(), control='pure')
    old.cumulative_energy_j[:] = 100 * budget
    np.testing.assert_array_equal(old.service_available(), [True] * 4)


def test_crossing_applies_next_slot_and_flight_accounting_continues():
    env = UAVMECEnv(cutoff_config(), seed=1, control='pure')
    env.reset(30000)
    env.scenario.route_uniform[:2] = .5
    env.cumulative_energy_j[0] = env.cfg.energy_budget_j - 1.
    action = prefer_first_uav(env)
    action[:6] = [-.3, 0., .1, 0., .1, 0.]
    first = env.step(action)[4]
    assert first['service_cutoff_enabled'] is True
    assert first['uav_service_available'] == [True, True, True]
    assert first['uav_service_disabled'] == [False, False, False]
    assert first['newly_service_disabled'] == [True, False, False]
    assert first['association_counts'][1] == env.cfg.num_users
    assert first['compute_energy_j'][0] > 0
    assert first['cumulative_energy_j'][0] > env.cfg.energy_budget_j
    second = env.step(action)[4]
    assert second['uav_service_available'] == [False, True, True]
    assert second['uav_service_disabled'] == [True, False, False]
    assert second['newly_service_disabled'] == [False, False, False]
    assert_available_allocation(second, env.cfg)
    assert second['compute_energy_j'][0] == 0
    assert second['propulsion_energy_j'][0] > 0
    assert np.linalg.norm(second['uav_velocity'][0]) > 0
    assert second['cumulative_energy_j'][0] > first['cumulative_energy_j'][0]
    assert second['cumulative_compute_energy_j'][0] == first['cumulative_compute_energy_j'][0]
    expected_power = propulsion_power(np.linalg.norm(second['uav_velocity'], axis=1), env.cfg)
    np.testing.assert_allclose(second['propulsion_energy_j'], expected_power * env.cfg.frame_s)
    assert second['energy_violation_cost'] > 0
    assert not second['energy_budget_ok']


def test_pure_masks_before_categorical_draw_and_redistributes_resources_without_solver(monkeypatch):
    import priority_mec.solvers as solvers
    env = UAVMECEnv(cutoff_config(), control='pure')
    env.cumulative_energy_j[:] = [env.cfg.energy_budget_j, 0., env.cfg.energy_budget_j + 1.]
    uniforms = env.scenario.route_uniform.copy()
    action = np.linspace(-1., 1., env.action_size)
    score = env.cfg.pure_route_temperature_scale * action[6:106].reshape(20, 5)
    score[:, [1, 3]] = -np.inf
    exponentials = np.exp(score - np.max(score, axis=1, keepdims=True))
    probability = exponentials / exponentials.sum(axis=1, keepdims=True)
    expected = np.sum(uniforms[0, :, None] > np.cumsum(probability, axis=1), axis=1)
    monkeypatch.setattr(solvers, 'solve', lambda *args, **kwargs: pytest.fail('Pure control invoked solver'))
    monkeypatch.setattr(np.random, 'default_rng', lambda *args, **kwargs: pytest.fail('Extra route RNG'))
    _, _, _, _, info = env.step(action)
    np.testing.assert_array_equal(info['association'], expected)
    np.testing.assert_array_equal(env.scenario.route_uniform, uniforms)
    assert_available_allocation(info, env.cfg)
    association = np.asarray(info['association'])
    for node in (2, 4):
        selected = np.flatnonzero(association == node)
        if len(selected):
            for key, values, budget in (
                ('bandwidth_hz', action[106:126] * env.cfg.pure_resource_temperature_scale,
                 env.cfg.uav_bandwidth_hz if node == 2 else env.cfg.satellite_bandwidth_hz),
                ('cpu_hz', action[126:146] * env.cfg.pure_resource_temperature_scale,
                 env.cfg.uav_cpu_hz if node == 2 else env.cfg.satellite_cpu_hz)):
                shares = np.exp(values[selected] - np.max(values[selected]))
                np.testing.assert_allclose(np.asarray(info[key])[selected], budget * shares / shares.sum())


@pytest.mark.parametrize('control', ['pure', 'hover_random', 'hover_balanced', 'greedy_rate'])
def test_all_uavs_closed_keeps_all_tasks_and_does_not_borrow_uav_capacity(control, monkeypatch):
    import priority_mec.solvers as solvers
    env = UAVMECEnv(cutoff_config(), control=control)
    env.cumulative_energy_j[:] = env.cfg.energy_budget_j
    monkeypatch.setattr(solvers, 'solve', lambda *args, **kwargs: pytest.fail('Non-solver control invoked solver'))
    action = prefer_first_uav(env) if control == 'pure' else None
    _, reward, _, _, info = env.step(action)
    assert np.isfinite(reward)
    assert set(info['association']).issubset({0, 4})
    assert_available_allocation(info, env.cfg)
    assert info['compute_energy_j'] == [0., 0., 0.]
    assert np.all(np.asarray(info['propulsion_energy_j']) > 0)
    assert np.all(np.asarray(info['cumulative_energy_j']) > env.cfg.energy_budget_j)
    assert info['fallback_count'] == 0
    assert info['allocation_invalid_count'] == 0
    if control == 'hover_balanced':
        assert info['association_counts'] == [10, 0, 0, 0, 10]


def test_only_remaining_uav_receives_its_own_capacity():
    env = UAVMECEnv(cutoff_config(), control='hover_balanced')
    env.cumulative_energy_j[:] = [env.cfg.energy_budget_j, 0., env.cfg.energy_budget_j]
    info = env.step()[4]
    assert info['association_counts'] == [0, 0, 20, 0, 0]
    assert np.sum(info['cpu_hz']) == pytest.approx(env.cfg.uav_cpu_hz)
    assert np.sum(info['bandwidth_hz']) == pytest.approx(env.cfg.uav_bandwidth_hz)
    assert_available_allocation(info, env.cfg)


def test_prepared_service_uses_mask_once_and_known_cost_keeps_reward_units(monkeypatch):
    import priority_mec.solvers as solvers
    calls = []
    def local_solver(*args, **kwargs):
        calls.append(kwargs['remote_available'].copy())
        return {'association': np.zeros(20, dtype=int), 'bandwidth_hz': np.zeros(20),
                'cpu_hz': np.zeros(20), 'metrics': {}}
    monkeypatch.setattr(solvers, 'solve', local_solver)
    env = UAVMECEnv(cutoff_config(delay_cost_weight=5.), control='hybrid')
    env.cumulative_energy_j[:] = [env.cfg.energy_budget_j, 0., 0.]
    uniforms = env.scenario.route_uniform.copy()
    expected_j = np.mean(env.scenario.data_bit[0] * env.cfg.cycles_per_bit
                         / env.cfg.local_cpu_hz / env.scenario.deadline_s[0])
    known = env.prepare_service()
    assert known == pytest.approx(5 * expected_j)
    assert env.prepare_service() == known
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [False, True, True, True])
    assert env._service_cache['remote_available'] == [False, True, True, True]
    _, _, _, _, info = env.step(np.array([-.8, 0., .5, 0., .5, 0.]))
    assert len(calls) == 1
    assert info['delay_reward_cost'] == pytest.approx(known)
    np.testing.assert_array_equal(env.scenario.route_uniform, uniforms)
    assert env._service_cache is None


def test_prepared_service_rejects_changed_eligibility(monkeypatch):
    import priority_mec.solvers as solvers
    monkeypatch.setattr(solvers, 'solve', lambda *args, **kwargs: {
        'association': np.zeros(20, dtype=int), 'bandwidth_hz': np.zeros(20),
        'cpu_hz': np.zeros(20), 'metrics': {}})
    env = UAVMECEnv(cutoff_config(), control='hybrid')
    env.prepare_service()
    env.cumulative_energy_j[0] = env.cfg.energy_budget_j
    with pytest.raises(RuntimeError, match='eligibility'):
        env.prepare_service()
    with pytest.raises(RuntimeError, match='eligibility'):
        env.step(np.zeros(6))


@pytest.mark.parametrize('control', ['hybrid', 'greedy_solver'])
def test_real_solver_reoptimizes_after_all_uavs_shut_down(control):
    env = UAVMECEnv(cutoff_config(), seed=1, control=control)
    env.reset(30000)
    # Previous-frame associations cannot keep a now-disabled service alive.
    env.previous_association[:] = np.arange(20) % 3 + 1
    env.cumulative_energy_j[:] = env.cfg.energy_budget_j
    if control == 'hybrid':
        env.prepare_service()
    _, reward, _, _, info = env.step(np.zeros(6))
    assert np.isfinite(reward)
    assert_available_allocation(info, env.cfg)
    assert set(info['association']).issubset({0, 4})
    assert info['allocation_invalid_count'] == 0
    assert info['compute_energy_j'] == [0., 0., 0.]


def test_environment_resume_preserves_cutoff_state_and_prepared_cost(monkeypatch):
    import priority_mec.solvers as solvers
    monkeypatch.setattr(solvers, 'solve', lambda *args, **kwargs: {
        'association': np.zeros(20, dtype=int), 'bandwidth_hz': np.zeros(20),
        'cpu_hz': np.zeros(20), 'metrics': {}})
    cfg = cutoff_config(delay_cost_weight=5.)
    env = UAVMECEnv(cfg, control='hybrid')
    env.cumulative_energy_j[:] = [cfg.energy_budget_j, 0., 0.]
    env.prepare_service()
    saved = env.state_dict()
    clone = UAVMECEnv(cfg, control='hybrid')
    clone.load_state_dict(saved)
    np.testing.assert_array_equal(env.service_available(), clone.service_available())
    assert env.prepare_service() == clone.prepare_service()
    left, right = env.step(np.zeros(6)), clone.step(np.zeros(6))
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1] == right[1]
    for key in ('association', 'cumulative_energy_j', 'uav_service_available', 'newly_service_disabled'):
        assert left[4][key] == right[4][key]


def test_under_threshold_flag_does_not_change_scenarios_actions_rewards_or_observations():
    old = UAVMECEnv(Config(), seed=1, control='pure')
    new = UAVMECEnv(cutoff_config(), seed=1, control='pure')
    np.testing.assert_array_equal(old.reset(30000), new.reset(30000))
    assert old.scenario.hash == new.scenario.hash
    action = np.linspace(-.2, .2, old.action_size)
    for _ in range(2):
        a, b = old.step(action), new.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
        for key in ('association', 'bandwidth_hz', 'cpu_hz', 'delay_s', 'uav_velocity',
                    'energy_j', 'frame_scenario_hash', 'next_user_position'):
            assert a[4][key] == b[4][key]
