"""Reward reweighting must leave physics, solver, and penalty attribution intact."""
import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.episode_log import EpisodeAccumulator, delay_reward_cost
from priority_mec.io import load_checkpoint
from priority_mec.reporting import _penalty
from priority_mec.rl.value_components import (
    PopArtLearnedContinuationCritic, reward_components)
from priority_mec.training import (
    TrainConfig, _cost, augment_observations, evaluate, learner_for, train)


@pytest.mark.parametrize('weight', [0., -1., float('nan'), float('inf')])
def test_delay_weight_must_be_finite_and_positive(weight):
    with pytest.raises(ValueError):
        Config(delay_cost_weight=weight).validate()


@pytest.mark.parametrize('control', ['local', 'pure', 'hybrid'])
def test_weight_changes_only_delay_reward_not_physics_allocations_or_penalties(control):
    config = Config(episode_frames=2)
    original = UAVMECEnv(config, seed=1, control=control)
    changed = UAVMECEnv(replace(config, delay_cost_weight=5.), seed=1, control=control)
    for env in (original, changed):
        env.reset(episode=30000)
        # A real budget violation makes accidentally scaling penalties detectable.
        env.cumulative_energy_j[:] = config.energy_budget_j
    for _ in range(2):
        action = np.full(original.action_size, .3)
        left_obs, left_reward, _, _, left = original.step(action)
        right_obs, right_reward, _, _, right = changed.step(action)
        np.testing.assert_array_equal(left_obs, right_obs)
        for key in ('scenario_hash', 'frame_scenario_hash', 'association', 'bandwidth_hz',
                    'cpu_hz', 'normalized_weighted_delay', 'mean_delay_s',
                    'energy_j', 'energy_violation_cost', 'penalty_cost',
                    'collision_attempt', 'boundary_attempt', 'next_uav_position'):
            assert left[key] == right[key], key
        j = left['normalized_weighted_delay']
        assert right_reward == pytest.approx(left_reward-4*j)
        assert right['delay_reward_cost'] == pytest.approx(5*j)
        assert right['delay_reward_component'] == pytest.approx(-5*j)
        assert right_reward == pytest.approx(-right['delay_reward_cost']-right['penalty_cost'])
        assert _cost(right) == j


def test_known_continuation_uses_weighted_cost_and_preserves_single_solver_call(monkeypatch):
    from priority_mec import solvers
    original_solve = solvers.solve
    calls = []
    def observed_solve(*args, **kwargs):
        calls.append(1)
        return original_solve(*args, **kwargs)
    monkeypatch.setattr(solvers, 'solve', observed_solve)
    cfg = Config(episode_frames=2, delay_cost_weight=5.)
    env = UAVMECEnv(cfg, seed=1, control='hybrid')
    run = TrainConfig(algorithm='ppo', control='hybrid', device='cpu', hidden=8,
        ppo_variant='value_decomp_popart', ppo_continuation_critic=True,
        ppo_known_delay_feature=True, ppo_motion_memory=True)
    base = env.reset(episode=30000)
    learner = learner_for(len(base)+7, 6, run, cfg.to_dict())
    for frame in range(2):
        known = env.prepare_service()
        assert env.prepare_service() == known
        obs = augment_observations(base, np.zeros(6), known, run)
        tensor = torch.as_tensor(obs[None])
        components = learner.critic.forward_components(tensor)
        future = learner.critic.continuation_components(tensor)
        torch.testing.assert_close(components[:, 0], future[:, 0]-tensor[:, -1])
        if frame == 1:
            assert components[0, 0].item() == pytest.approx(-known)
        base, reward, _, _, info = env.step(np.zeros(6))
        assert known == pytest.approx(info['delay_reward_cost'])
        terms = reward_components([info], [reward])[0]
        np.testing.assert_allclose(terms, [-known, -info['penalty_cost']], atol=1e-12)
        assert len(calls) == frame+1
    assert env.prepare_service() == 0.


def test_pure_learned_immediate_targets_are_weighted_rewards_without_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure reward decomposition must not use the solver')
    monkeypatch.setattr('priority_mec.solvers.solve', forbidden)
    monkeypatch.setattr(UAVMECEnv, 'prepare_service', forbidden)
    env = UAVMECEnv(Config(episode_frames=1, delay_cost_weight=5.), control='pure')
    _, reward, _, _, info = env.step(np.zeros(146))
    components = reward_components([info], [reward])
    assert components[0, 0] == pytest.approx(-5*info['normalized_weighted_delay'])
    assert components[0, 1] == pytest.approx(-info['penalty_cost'])
    # A future-delay return is separated from the weighted immediate target.
    returns = torch.as_tensor(components, dtype=torch.float32)
    returns[:, 0] -= 2.
    immediate = torch.as_tensor(components[:, 0], dtype=torch.float32)
    targets = PopArtLearnedContinuationCritic.regression_targets(returns, immediate)
    torch.testing.assert_close(targets[:, 0], immediate)
    torch.testing.assert_close(targets[:, 1], torch.tensor([-2.]))
    torch.testing.assert_close(targets[:, :2].sum(-1), returns[:, 0])


def test_weighted_delay_is_separate_from_penalties_in_episode_and_evaluation_logs(tmp_path):
    cfg = Config(episode_frames=2, delay_cost_weight=5.)
    env = UAVMECEnv(cfg, seed=1, control='local')
    accumulator = EpisodeAccumulator(30000, 0)
    env.reset(episode=30000)
    for step in range(2):
        _, reward, _, _, info = env.step()
        accumulator.add(reward, info, step+1)
        accumulator = EpisodeAccumulator.from_state_dict(accumulator.state_dict())
    row = accumulator.record(2, cfg.energy_budget_j)
    evaluation = evaluate(cfg, control='local', episodes=1, seed=1,
        episode_start=30000, out=tmp_path/'evaluation')['episodes'][0]
    for key in ('weighted_delay_mean', 'delay_reward_cost_mean', 'delay_reward_cost_sum',
                'delay_reward_sum', 'penalty_sum', 'episode_return'):
        assert row[key] == pytest.approx(evaluation[key])
    assert row['delay_reward_cost_sum'] == pytest.approx(5*2*row['weighted_delay_mean'])
    assert row['penalty_sum'] == pytest.approx(-row['episode_return']-row['delay_reward_cost_sum'])
    assert _penalty(row) == pytest.approx(row['penalty_sum'])
    assert _penalty(dict(episode_return=-12., frames=2, weighted_delay_mean=1.,
                         delay_cost_weight=5.)) == pytest.approx(2.)
    assert _penalty(dict(episode_return=-12., delay_reward_cost_sum=10.)) == pytest.approx(2.)
    assert delay_reward_cost(dict(normalized_weighted_delay=2.)) == 2.
    assert delay_reward_cost(dict(normalized_weighted_delay=2., delay_cost_weight=5.)) == 10.


@pytest.mark.parametrize('control', ['pure', 'hybrid'])
def test_weighted_continuation_short_training_resumes_exactly(tmp_path, monkeypatch, control):
    cfg = Config(episode_frames=2, delay_cost_weight=5.)
    run = TrainConfig(algorithm='ppo', control=control, device='cpu', hidden=8,
        num_envs=2, use_processes=False, ppo_rollout_frames=4,
        ppo_variant='value_decomp_popart', ppo_motion_memory=True,
        ppo_continuation_critic=True, ppo_known_delay_feature=control == 'hybrid',
        ppo_continuation_mode='learned' if control == 'pure' else 'known',
        ppo_epochs=1, ppo_critic_epochs=1, ppo_minibatch_size=4,
        render_every_episodes=0, eval_every=100, checkpoint_every=100)
    monkeypatch.setattr('priority_mec.training.source_manifest', lambda: {'test': 'fixed'})
    full = tmp_path/'full'
    partial = tmp_path/'partial'
    train(cfg, full, 8, run)
    train(cfg, partial, 2, run)
    checkpoint = load_checkpoint(partial/'last.pt')
    assert len(checkpoint['pending']['reward']) == 2
    for i in range(2):
        frame = checkpoint['environments'][i]['history'][0]
        np.testing.assert_allclose(checkpoint['pending']['reward_components'][i],
            [-frame['delay_reward_cost'], -frame['penalty_cost']], atol=1e-6)
    train(cfg, partial, 8, run, partial/'last.pt')
    complete = load_checkpoint(full/'last.pt')
    resumed = load_checkpoint(partial/'last.pt')
    for name in ('actor', 'critic'):
        for key in complete['learner'][name]:
            torch.testing.assert_close(complete['learner'][name][key],
                                       resumed['learner'][name][key], rtol=0, atol=0)
    with (partial/'episodes.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    for row in rows:
        assert float(row['delay_reward_cost_sum']) == pytest.approx(
            5*float(row['frames'])*float(row['weighted_delay_mean']))
    manifest = json.loads((partial/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['reward_changed'] is True

