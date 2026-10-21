"""Closed-service accounting distinguishes threshold crossing from the next slot."""
import csv
import json

import numpy as np
import pytest

from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.episode_log import EpisodeAccumulator, ServiceCutoffAccumulator
from priority_mec.io import load_checkpoint
from priority_mec.reporting import COMPARISON_METRICS
from priority_mec.training import TrainConfig, evaluate, train


def fixture(slot):
    return dict(scenario_hash='fixed', normalized_weighted_delay=.5,
        mean_delay_s=.4, deadline_completion=1., collision_attempt=False,
        boundary_attempt=False, cumulative_energy_j=[100. * slot, 50. * slot, 34. * slot],
        cumulative_propulsion_energy_j=[90. * slot, 45. * slot, 30. * slot],
        cumulative_compute_energy_j=[10. * slot, 5. * slot, 4. * slot],
        service_cutoff_enabled=True, service_cutoff_threshold_j=100.,
        uav_service_disabled=[slot >= 2, slot >= 3, False],
        association=[0, 1, 2, 3, 4])


def test_threshold_and_next_slot_closure_are_distinct_and_resume_exactly():
    whole = EpisodeAccumulator(7, 0)
    partial = EpisodeAccumulator(7, 0)
    for slot in (1, 2, 3):
        whole.add(-.5, fixture(slot), slot)
        partial.add(-.5, fixture(slot), slot)
        partial = EpisodeAccumulator.from_state_dict(json.loads(json.dumps(partial.state_dict())))
    expected = whole.record(3, 100.)
    assert partial.record(3, 100.) == expected
    assert expected['per_uav_first_threshold_reached_slot'] == [1, 2, 3]
    assert expected['per_uav_first_service_disabled_slot'] == [2, 3, None]
    assert expected['per_uav_service_disabled_frames'] == [2, 1, 0]
    assert expected['service_disabled_uav_frames'] == 3
    assert expected['service_disabled_any_frames'] == 2
    assert expected['service_disabled_uav_frame_fraction'] == 1 / 3
    assert expected['service_disabled_any_frame_fraction'] == 2 / 3
    assert expected['local_task_fraction'] == pytest.approx(.2)
    assert expected['uav_task_fraction'] == pytest.approx(.6)
    assert expected['satellite_task_fraction'] == pytest.approx(.2)
    # Turning off computing never relabels the old strict energy criterion.
    assert not expected['budget_feasible']
    assert not expected['fully_feasible']


def test_old_info_defaults_do_not_claim_service_closure_or_fabricate_routes():
    info = fixture(1)
    for key in ('service_cutoff_enabled', 'service_cutoff_threshold_j',
                'uav_service_disabled', 'association'):
        info.pop(key)
    service = ServiceCutoffAccumulator()
    service.add(info)
    row = service.record()
    assert not row['service_cutoff_enabled']
    assert row['service_disabled_uav_frames'] == 0
    assert row['per_uav_first_service_disabled_slot'] == [None] * 3
    assert row['per_uav_first_threshold_reached_slot'] == [None] * 3
    assert row['local_task_fraction'] == row['uav_task_fraction'] == row['satellite_task_fraction'] == 0.


def test_threshold_is_reported_even_with_cutoff_disabled():
    info = fixture(1)
    info['service_cutoff_enabled'] = False
    service = ServiceCutoffAccumulator()
    service.add(info)
    row = service.record()
    assert row['per_uav_first_threshold_reached_slot'] == [1, None, None]
    assert row['service_disabled_uav_frames'] == 0


def test_training_and_evaluation_use_identical_service_summary_rules(tmp_path):
    cfg = Config(episode_frames=4, energy_budget_j=1., uav_service_cutoff_enabled=True)
    env = UAVMECEnv(cfg, seed=1, control='pure')
    env.reset(30000)
    accumulator = EpisodeAccumulator(30000, 0)
    for slot in range(1, 5):
        _, reward, _, _, info = env.step(np.zeros(146))
        accumulator.add(reward, info, slot)
    training = accumulator.record(4, cfg.energy_budget_j)
    result = evaluate(cfg, control='pure', episodes=1, episode_start=30000,
                      out=tmp_path/'evaluation')
    episode = result['episodes'][0]
    service_fields = set(accumulator.service_log.record())
    for key in service_fields:
        assert training[key] == episode[key], key
    assert episode['per_uav_first_threshold_reached_slot'] == [1, 1, 1]
    assert episode['per_uav_first_service_disabled_slot'] == [2, 2, 2]
    assert episode['service_disabled_uav_frames'] == 9
    assert episode['service_disabled_any_frames'] == 3
    assert result['summary']['service_disabled_uav_frames'] == 9
    assert result['summary']['energy_feasible_episodes'] == 0
    assert result['summary']['service_cutoff_protocol']['propulsion_continues']
    assert service_fields - {'per_uav_service_disabled_frames',
        'per_uav_first_service_disabled_slot', 'per_uav_first_threshold_reached_slot'} <= set(COMPARISON_METRICS)


def test_actual_training_restores_partial_closed_service_log(tmp_path, monkeypatch):
    cfg = Config(episode_frames=3, energy_budget_j=1., uav_service_cutoff_enabled=True)
    run = TrainConfig(algorithm='ppo', control='pure', device='cpu', hidden=8,
        num_envs=2, use_processes=False, ppo_rollout_frames=4,
        ppo_epochs=1, ppo_critic_epochs=1, ppo_minibatch_size=4,
        eval_every=100, checkpoint_every=100, render_every_episodes=0)
    monkeypatch.setattr('priority_mec.training.source_manifest', lambda: {'test': 'cutoff'})
    train(cfg, tmp_path/'full', 8, run)
    train(cfg, tmp_path/'partial', 2, run)
    checkpoint = load_checkpoint(tmp_path/'partial/last.pt')
    for accumulator in checkpoint['episode_accumulators']:
        assert accumulator['service_log']['first_threshold_reached_slot'] == [1, 1, 1]
        assert accumulator['service_log']['first_disabled_slot'] == [None] * 3
    train(cfg, tmp_path/'partial', 8, run, tmp_path/'partial/last.pt')
    def read(path):
        with path.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    expected, actual = read(tmp_path/'full/episodes.csv'), read(tmp_path/'partial/episodes.csv')
    keys = ('service_cutoff_enabled', 'service_disabled_uav_frames', 'service_disabled_any_frames',
        'per_uav_first_service_disabled_slot', 'per_uav_first_threshold_reached_slot',
        'local_task_fraction', 'uav_task_fraction', 'satellite_task_fraction',
        'episode_return', 'weighted_delay_mean', 'budget_feasible', 'fully_feasible')
    assert [{k: r[k] for k in keys} for r in actual] == [{k: r[k] for k in keys} for r in expected]
    manifest = json.loads((tmp_path/'partial/manifest.json').read_text())
    assert manifest['service_cutoff_protocol']['enabled']
    assert manifest['service_cutoff_protocol']['threshold_j'] == 1.
    assert manifest['service_cutoff_protocol']['energy_penalty_unchanged']
