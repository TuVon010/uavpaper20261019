"""NTN model boundaries, gain calibration and paired exogenous randomness."""
import numpy as np
import pytest

from priority_mec.config import Config
from priority_mec.physics import (
    LIGHT_MPS, channel_parameters, initial_uav_positions, link_rate,
    satellite_channel_components, satellite_urban_parameters,
)
from priority_mec.scenarios import Scenario, keyed_rng


def test_ntn_urban_reference_tables_and_nearest_angle():
    probability, clutter, k = satellite_urban_parameters([10., 20., 34.9, 35., 90., 100.])
    np.testing.assert_array_equal(probability, [.246, .386, .493, .613, .992, .992])
    np.testing.assert_array_equal(clutter, [34.3, 30.9, 29., 27.7, 25.5, 25.5])
    np.testing.assert_array_equal(k, [31.83, 18.78, 10.49, 7.46, 3.68, 3.68])


def test_ntn_separates_large_and_small_scale_loss_without_double_counting():
    channel = satellite_channel_components(np.array([90., 90.]),
        np.array([0., 1.]), np.array([1., -1.]), np.array([[0., 0.], [1., 1.]]))
    np.testing.assert_array_equal(channel['satellite_los'], [True, False])
    np.testing.assert_array_equal(channel['satellite_shadow_loss_db'], [4., -6.])
    np.testing.assert_array_equal(channel['satellite_clutter_loss_db'], [0., 25.5])
    k = 10**(3.68/10)
    np.testing.assert_allclose(channel['satellite_fading_power'], [k/(k+1), 1.])


@pytest.mark.parametrize('is_los', [False, True])
def test_ntn_conditional_small_scale_power_is_unit_mean(is_los):
    n = 100_000
    rng = np.random.default_rng(419)
    channel = satellite_channel_components(np.full(n, 80.),
        np.full(n, 0. if is_los else 1.), np.zeros(n), rng.normal(size=(n, 2)))
    fading = channel['satellite_fading_power']
    assert np.all(fading >= 0)
    assert np.mean(fading) == pytest.approx(1., abs=.012)
    assert np.var(fading) > .1


def test_ntn_cn0_uses_gt_not_a_second_receiver_noise_figure():
    cfg = Config()
    scene = Scenario(cfg, 1, 52)
    scene.satellite_los_uniform[:] = 0.
    scene.satellite_shadow_normal[:] = 0.
    scene.satellite_fade_normal[:] = 0.
    position = np.full((cfg.num_users, 2), cfg.area_m/2)
    frame = cfg.episode_frames//2
    rho, offset, diag = channel_parameters(position, initial_uav_positions(cfg), frame, scene, cfg)
    fspl = 20*np.log10(4*np.pi*cfg.satellite_altitude_m*cfg.satellite_carrier_hz/LIGHT_MPS)
    k = 10**(3.68/10)
    expected = 10**((10*np.log10(cfg.tx_power_w)+cfg.satellite_gt_db_k
                     -fspl-cfg.satellite_extra_loss_db+228.6)/10)*k/(k+1)
    np.testing.assert_allclose(rho[:, -1], expected)
    np.testing.assert_allclose(offset[:, -1], 2*cfg.satellite_altitude_m/LIGHT_MPS)
    np.testing.assert_allclose(diag['satellite_fspl_db'], fspl)
    # Ground receiver NF is not applied again to the satellite's specified G/T.
    cfg.noise_figure_db += 9
    changed, _, _ = channel_parameters(position, initial_uav_positions(cfg), frame, scene, cfg)
    np.testing.assert_array_equal(changed[:, -1], rho[:, -1])
    assert np.all(changed[:, :-1] < rho[:, :-1])


def test_ntn_minimum_elevation_is_an_explicit_link_outage():
    cfg = Config(satellite_min_elevation_deg=95.)
    scene = Scenario(cfg, 1, 5)
    rho, _, _ = channel_parameters(scene.initial_position, initial_uav_positions(cfg), 0, scene, cfg)
    np.testing.assert_array_equal(rho[:, -1], np.zeros(cfg.num_users))


def test_new_satellite_streams_do_not_change_existing_task_or_uav_randomness():
    cfg = Config()
    scene = Scenario(cfg, 1, 72)
    for frame in (0, 1, cfg.episode_frames-1):
        n, m = cfg.num_users, cfg.num_uavs
        np.testing.assert_array_equal(scene.data_bit[frame],
            keyed_rng(1, 72, frame, 10).uniform(cfg.data_min_bit, cfg.data_max_bit, n))
        np.testing.assert_array_equal(scene.los_uniform[frame],
            keyed_rng(1, 72, frame, 14).random((n, m)))
        np.testing.assert_array_equal(scene.fade_normal[frame],
            keyed_rng(1, 72, frame, 15).normal(size=(n, m, 2)))
        np.testing.assert_array_equal(scene.route_uniform[frame],
            keyed_rng(1, 72, frame, 16).random(n))
    duplicate = Scenario(cfg, 1, 72)
    assert duplicate.hash == scene.hash
    assert duplicate.frame_hash(0) == scene.frame_hash(0)
    for name in ('satellite_los_uniform', 'satellite_shadow_normal', 'satellite_fade_normal'):
        duplicate = Scenario(cfg, 1, 72)
        getattr(duplicate, name)[0].flat[0] += .125
        assert duplicate._hash() != scene.hash
        assert duplicate.frame_hash(0) != scene.frame_hash(0)
        assert duplicate.frame_hash(1) == scene.frame_hash(1)


def test_channel_queries_do_not_redraw_fading_and_bandwidth_gain_is_noise_limited():
    cfg = Config()
    scene = Scenario(cfg, 1, 17)
    state_before = np.random.get_state()
    first, _, _ = channel_parameters(scene.initial_position, initial_uav_positions(cfg), 0, scene, cfg)
    second, _, _ = channel_parameters(scene.initial_position, initial_uav_positions(cfg), 0, scene, cfg)
    np.testing.assert_array_equal(first, second)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    # Increasing pool size cannot create a linear fivefold rate gain at fixed P.
    rho = first[:, -1]
    small, large = link_rate(20e6, rho), link_rate(100e6, rho)
    assert np.all(large >= small)
    assert np.all(large <= rho/np.log(2))
    assert np.all(large < 5*small)
