import numpy as np
import pytest
from priority_mec.config import Config
from priority_mec.physics import initial_uav_positions, minimum_separation, safe_velocity


def check(position, velocity, cfg):
    assert np.all(np.linalg.norm(velocity, axis=1) <= cfg.speed_max_mps+1e-8)
    final = position+cfg.frame_s*velocity
    assert np.all(final >= -1e-8) and np.all(final <= cfg.area_m+1e-8)
    assert minimum_separation(position, velocity, cfg.frame_s) >= cfg.safe_distance_m-1e-9


def test_contact_pair_does_not_freeze_remote_uav():
    cfg = Config()
    position = np.array([[500., 500.], [550., 500.], [900., 800.]])
    requested = np.array([[10., 0.], [-10., 0.], [-10., -10.]])
    actual, info = safe_velocity(position, requested, cfg)
    check(position, actual, cfg)
    np.testing.assert_allclose(actual[2], requested[2], atol=1e-9)
    np.testing.assert_allclose(actual[:2], 0., atol=1e-9)
    assert info['projection_called'] and not info['projection_failed']
    assert info['collision_attempt']


def test_speed_boundary_and_projection_idempotence():
    cfg = Config()
    position = initial_uav_positions(cfg)
    actual, info = safe_velocity(position, [[40., 30.], [50., -50.], [-50., 40.]], cfg)
    check(position, actual, cfg)
    again, info_again = safe_velocity(position, actual, cfg)
    np.testing.assert_allclose(again, actual, atol=1e-12)
    assert info['boundary_attempt'] and not info_again['projection_called']


def test_continuous_crossing_and_idempotence():
    cfg = Config()
    position = np.array([[480., 500.], [540., 500.], [900., 800.]])
    actual, info = safe_velocity(position, [[20., 10.], [-20., -10.], [0., -10.]], cfg)
    check(position, actual, cfg)
    assert info['projection_called']
    again, second = safe_velocity(position, actual, cfg)
    np.testing.assert_allclose(again, actual, atol=1e-10)
    assert not second['projection_called']


def test_random_safe_initial_states_and_requests():
    cfg, rng, solves = Config(), np.random.default_rng(6248), 0
    for iteration in range(240):
        if iteration % 2:
            position = np.array([[470., 500.], [530.+rng.uniform(0, 8), 500.], [900., 800.]])
        else:
            while True:
                position = rng.uniform(0., cfg.area_m, (3, 2))
                if minimum_separation(position, np.zeros((3, 2)), 1.) >= cfg.safe_distance_m:
                    break
        actual, info = safe_velocity(position, rng.normal(0., 35., (3, 2)), cfg)
        check(position, actual, cfg)
        again, _ = safe_velocity(position, actual, cfg)
        np.testing.assert_allclose(again, actual, atol=1e-9)
        assert not info['projection_failed']
        solves += info['projection_called']
    assert solves > 5


def test_solver_failure_explicit_safe_zero_fallback(monkeypatch):
    import priority_mec.physics as physics
    def fail(*args, **kwargs):
        raise RuntimeError('injected solver failure')
    monkeypatch.setattr(physics, 'minimize', fail)
    cfg = Config()
    position = np.array([[500., 500.], [550., 500.], [900., 800.]])
    actual, info = safe_velocity(position, [[10., 0.], [-10., 0.], [-10., -10.]], cfg)
    check(position, actual, cfg)
    np.testing.assert_array_equal(actual, np.zeros((3, 2)))
    assert info['projection_failed'] and info['projection_status'] == 'solver_exception'


def test_invalid_initial_state_fails_instead_of_claiming_safety():
    with pytest.raises(ValueError, match='initially safe'):
        safe_velocity(np.array([[500., 500.], [549., 500.], [900., 800.]]), np.zeros((3, 2)), Config())
