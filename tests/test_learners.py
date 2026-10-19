"""Learner math, independent evaluation RNG, exact restore and vector-rollout checks."""
import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from priority_mec.learners import (Replay, SAC, SACConfig, DeterministicConfig,
    DeterministicLearner, PPO, PPOConfig, seed_everything, compute_gae)
from priority_mec.rl.sac import Actor as SACActor
from priority_mec.rl.ppo import Actor as PPOActor, clipped_policy_loss
from priority_mec.rl.common import tanh_log_jacobian

torch.set_num_threads(1)


def test_sac_stable_tanh_density_and_independent_sampling():
    seed_everything(1)
    actor = SACActor(4, 2, 16)
    state = torch.zeros(3, 4)
    rng = torch.Generator().manual_seed(1)
    global_before = torch.get_rng_state().clone()
    (action, log_probability) = actor.sample(state, generator=rng)
    assert torch.equal(global_before, torch.get_rng_state())
    assert torch.isfinite(log_probability).all()
    assert (action.abs() <= 1).all()
    raw = torch.tensor([-100.0, -5.0, 0.0, 5.0, 100.0], requires_grad=True)
    stable = tanh_log_jacobian(raw)
    assert torch.isfinite(stable).all()
    stable.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.allclose(stable[1:4], torch.log1p(-torch.tanh(raw[1:4]).square()), atol=0.0002)

def sac_replay():
    replay = Replay(4, 2, capacity=32, seed=1)
    for i in range(16):
        replay.add(np.full(4, i / 16), np.array([0.1, -0.2]), -0.5, np.full(4, (i + 1) / 16), i % 4 == 3)
    return replay

def test_sac_critic_only_warmup_and_complete_learning_state():
    torch.set_num_threads(1)
    seed_everything(1)
    cfg = SACConfig(hidden=16, batch_size=4, replay_capacity=32)
    learner = SAC(4, 2, cfg)
    replay = sac_replay()
    actor_before = copy.deepcopy(learner.actor.state_dict())
    result = learner.update(replay, update_actor=False)
    assert result['actor_updated'] == 0
    assert all((torch.equal(actor_before[k], v) for (k, v) in learner.actor.state_dict().items()))
    result = learner.update(replay)
    assert result['actor_updated'] == 1
    assert not all((torch.equal(actor_before[k], v) for (k, v) in learner.actor.state_dict().items()))
    assert all((np.isfinite(v) for v in result.values()))
    state = copy.deepcopy(learner.state_dict())
    restored = SAC(4, 2, cfg)
    restored.load_state_dict(state)
    assert np.array_equal(learner.act(np.ones(4)), restored.act(np.ones(4)))
    assert learner.updates == restored.updates
    assert restored.alpha_optimizer.state_dict()['state']
    assert restored.q_optimizer.state_dict()['state']

def test_sac_terminal_transition_has_no_bootstrap():
    seed_everything(1)
    cfg = SACConfig(hidden=8, batch_size=4, replay_capacity=8)
    learner = SAC(4, 2, cfg)
    replay = Replay(4, 2, 8)
    for _ in range(8):
        replay.add(np.zeros(4), np.zeros(2), -3.25, np.ones(4), True)
    metrics = learner.update(replay, update_actor=False)
    assert metrics['target_mean'] == -3.25

def test_sac_replay_rng_resumes_and_keeps_requested_action():
    replay = sac_replay()
    state = replay.state_dict()
    restored = Replay(4, 2, 32)
    restored.load_state_dict(state)
    for (a, b) in zip(replay.sample(6, 'cpu'), restored.sample(6, 'cpu')):
        assert torch.equal(a, b)
    assert np.allclose(restored.action[0], [0.1, -0.2])
    with pytest.raises(ValueError):
        Replay(4, 2, 16).load_state_dict(state)

def test_sac_actor_diagnostics_reuse_forward_without_consuming_rng():
    seed_everything(1)
    actor = SACActor(4, 2, 16)
    observation = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    rng = torch.get_rng_state().clone()
    (action, log_probability) = actor.sample(observation)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    (diagnosed_action, diagnosed_probability, diagnostics) = actor.sample(observation, return_diagnostics=True)
    assert torch.equal(action, diagnosed_action)
    assert torch.equal(log_probability, diagnosed_probability)
    assert torch.equal(expected_rng, torch.get_rng_state())
    (mean, log_std) = actor(observation)
    assert diagnostics['latent_std_mean'] == log_std.exp().mean()
    assert diagnostics['mean_action_abs'] == torch.tanh(mean).abs().mean()
    assert not any((value.requires_grad for value in diagnostics.values()))

def test_sac_initial_temperature_and_update_diagnostics_use_actor_temperature():
    seed_everything(1)
    learner = SAC(4, 2, SACConfig(hidden=16, batch_size=4, replay_capacity=32, initial_alpha=0.01))
    actor_alpha = float(learner.alpha)
    assert actor_alpha == pytest.approx(0.01)
    result = learner.update(sac_replay())
    assert result['latent_std_mean'] > 0
    assert 0 <= result['mean_action_abs'] <= 1
    assert result['entropy_bonus_mean'] == pytest.approx(-actor_alpha * result['log_prob_mean'], rel=1e-06)
    with pytest.raises(ValueError, match='positive'):
        SACConfig(initial_alpha=0)

"""Bellman, gradient, delay and exact-resume checks for DDPG and TD3."""

def make_learner(algorithm='td3', **kwargs):
    values = dict(algorithm=algorithm, hidden=16, batch_size=8, replay_capacity=32)
    values.update(kwargs)
    return DeterministicLearner(3, 2, DeterministicConfig(**values))

def det_replay():
    replay = Replay(3, 2, capacity=32, seed=7)
    rng = np.random.default_rng(5)
    for index in range(32):
        replay.add(rng.normal(size=3), rng.uniform(-1, 1, 2), float(rng.normal()), rng.normal(size=3), index % 4 == 0)
    return replay

def assert_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for (a, b) in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right

def set_constant(network, value):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        list(network.parameters())[-1].fill_(value)

@pytest.mark.parametrize('values', [{'algorithm': 'other'}, {'batch_size': 0}, {'hidden': 1.5}, {'replay_capacity': 4, 'batch_size': 8}, {'tau': 0}, {'gamma': 1.1}, {'actor_lr': -1}, {'target_noise': -1}, {'noise_clip': float('nan')}])
def test_det_invalid_config(values):
    with pytest.raises(ValueError):
        DeterministicConfig(**values)

@pytest.mark.parametrize('algorithm', ['ddpg', 'td3'])
def test_det_architecture_and_target_initialization(algorithm):
    learner = make_learner(algorithm)
    assert learner.cfg.policy_delay == (1 if algorithm == 'ddpg' else 2)
    assert (learner.q2 is None) == (algorithm == 'ddpg')
    assert_equal(learner.actor.state_dict(), learner.target_actor.state_dict())
    assert_equal(learner.q1.state_dict(), learner.target_q1.state_dict())
    if learner.q2 is not None:
        assert_equal(learner.q2.state_dict(), learner.target_q2.state_dict())
    for network in (learner.target_actor, learner.target_q1, learner.target_q2):
        if network is not None:
            assert all((not p.requires_grad for p in network.parameters()))

@pytest.mark.parametrize('algorithm', ['ddpg', 'td3'])
def test_det_terminal_bellman_target_uses_reward_without_bootstrap(algorithm):
    learner = make_learner(algorithm, gamma=0.9)
    set_constant(learner.target_q1, 10.0)
    if learner.target_q2 is not None:
        set_constant(learner.target_q2, 7.0)
    target = learner._bellman_target(torch.tensor([[2.0], [-3.0]]), torch.zeros(2, 3), torch.tensor([[1.0], [0.0]]))
    assert target[0].item() == 2.0
    expected = -3.0 + 0.9 * (7.0 if algorithm == 'td3' else 10.0)
    assert target[1].item() == pytest.approx(expected, abs=1e-06)
    assert not target.requires_grad

def test_det_td3_clips_noise_before_final_action_and_uses_minimum(monkeypatch):
    learner = make_learner('td3', target_noise=0.2, noise_clip=0.1)
    set_constant(learner.target_actor, float(np.arctanh(0.95)))

    class RecordingQ(nn.Module):

        def forward(self, value):
            self.actions = value[:, -2:].clone()
            return value[:, -2:].sum(-1, keepdim=True)
    learner.target_q1 = RecordingQ()
    set_constant(learner.target_q2, 100.0)
    monkeypatch.setattr(torch, 'randn_like', lambda value: value.new_tensor([[50.0, -50.0]]))
    target = learner._bellman_target(torch.zeros(1, 1), torch.zeros(1, 3), torch.zeros(1, 1))
    assert torch.allclose(learner.target_q1.actions, torch.tensor([[1.0, 0.85]]), atol=1e-06)
    assert target.item() == pytest.approx(1.85, abs=1e-06)

def test_det_ddpg_has_no_target_smoothing_or_rng_draw():
    learner = make_learner('ddpg', target_noise=100.0)
    before = torch.get_rng_state().clone()
    learner._bellman_target(torch.zeros(8, 1), torch.zeros(8, 3), torch.zeros(8, 1))
    assert torch.equal(before, torch.get_rng_state())

def test_det_td3_delays_actor_and_all_targets_then_applies_polyak():
    torch.manual_seed(17)
    (learner, replay) = (make_learner('td3', tau=0.2), det_replay())
    actor_before = copy.deepcopy(learner.actor.state_dict())
    targets_before = {name: copy.deepcopy(getattr(learner, name).state_dict()) for name in ('target_actor', 'target_q1', 'target_q2')}
    first = learner.update(replay)
    assert first['actor_updated'] == 0
    assert_equal(actor_before, learner.actor.state_dict())
    for (name, state) in targets_before.items():
        assert_equal(state, getattr(learner, name).state_dict())
    second = learner.update(replay)
    assert second['actor_updated'] == 1
    assert learner.updates == 2
    for (name, source) in (('target_actor', 'actor'), ('target_q1', 'q1'), ('target_q2', 'q2')):
        for (key, value) in getattr(learner, name).state_dict().items():
            expected = targets_before[name][key].lerp(getattr(learner, source).state_dict()[key], 0.2)
            assert torch.equal(value, expected)

@pytest.mark.parametrize('algorithm', ['ddpg', 'td3'])
def test_det_critic_only_updates_do_not_backpropagate_actor_or_targets(algorithm):
    (learner, replay) = (make_learner(algorithm), det_replay())
    learner.update(replay)
    learner.update(replay)
    before = copy.deepcopy(learner.actor.state_dict())
    target_before = copy.deepcopy(learner.target_q1.state_dict())
    metrics = learner.update(replay, update_actor=False)
    assert metrics['actor_updated'] == 0
    assert_equal(before, learner.actor.state_dict())
    assert_equal(target_before, learner.target_q1.state_dict())
    assert all((p.grad is None for p in learner.actor.parameters()))
    for network in (learner.target_actor, learner.target_q1, learner.target_q2):
        if network is not None:
            assert all((p.grad is None for p in network.parameters()))
    assert all((math.isfinite(value) for value in metrics.values()))

@pytest.mark.parametrize('algorithm', ['ddpg', 'td3'])
def test_det_exact_resume_restores_targets_optimizers_counter_and_replay_rng(algorithm):
    torch.manual_seed(31)
    (learner, replay) = (make_learner(algorithm), det_replay())
    for _ in range(3):
        learner.update(replay)
    snapshot = learner.state_dict()
    (replay_snapshot, rng_snapshot) = (replay.state_dict(), torch.get_rng_state().clone())
    expected_metrics = [learner.update(replay) for _ in range(4)]
    (expected_state, expected_replay) = (learner.state_dict(), replay.state_dict())
    expected_rng = torch.get_rng_state().clone()
    (resumed, resumed_replay) = (make_learner(algorithm), det_replay())
    resumed.load_state_dict(snapshot)
    resumed_replay.load_state_dict(replay_snapshot)
    torch.set_rng_state(rng_snapshot)
    actual_metrics = [resumed.update(resumed_replay) for _ in range(4)]
    assert_equal(expected_metrics, actual_metrics)
    assert_equal(expected_state, resumed.state_dict())
    assert_equal(expected_replay, resumed_replay.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())

@pytest.mark.parametrize('algorithm', ['ddpg', 'td3'])
def test_det_act_is_bounded_and_does_not_consume_rng(algorithm):
    learner = make_learner(algorithm)
    before = torch.get_rng_state().clone()
    observation = np.array([100.0, -100.0, 4.0], dtype=np.float32)
    action = learner.act(observation, deterministic=False)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0)
    assert np.array_equal(action, learner.act(observation))
    assert torch.equal(before, torch.get_rng_state())

def test_det_checkpoint_rejects_different_algorithm_or_dimensions():
    state = make_learner('td3').state_dict()
    with pytest.raises(ValueError, match='configuration'):
        make_learner('ddpg').load_state_dict(state)
    with pytest.raises(ValueError, match='size'):
        DeterministicLearner(4, 2, make_learner('td3').cfg).load_state_dict(state)

def make_rollout(learner, count=12):
    rollout = {key: [] for key in ('observation', 'action', 'raw_action', 'log_prob', 'value', 'reward', 'terminated', 'next_value')}
    for index in range(count):
        obs = np.asarray([index / 20.0, math.sin(index), 0.3], np.float32)
        (action, log_prob, value, raw_action) = learner.act_train(obs)
        terminal = index % 4 == 3
        next_value = 0.0 if terminal else learner.value(obs + 0.05)
        for (key, value_item) in {'observation': obs, 'action': action, 'raw_action': raw_action, 'log_prob': log_prob, 'value': value, 'reward': -0.1 - 0.01 * index, 'terminated': terminal, 'next_value': next_value}.items():
            rollout[key].append(value_item)
    return {key: np.asarray(value) for (key, value) in rollout.items()}

def assert_nested_equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for (a, b) in zip(first, second):
            assert_nested_equal(a, b)
    elif torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    else:
        assert first == second

def test_ppo_gae_cannot_leak_between_episodes():
    (advantage, returns) = compute_gae([1.0, 2.0, 10.0, 20.0], [0.5, 0.7, 1.0, 2.0], [0.7, 999.0, 2.0, 999.0], [0, 1, 0, 1], 1.0, 1.0)
    np.testing.assert_allclose(returns, [3.0, 2.0, 30.0, 20.0])
    np.testing.assert_allclose(advantage, [2.5, 1.3, 29.0, 18.0])

def test_ppo_gae_nonterminal_rollout_bootstraps_and_zero_lambda_is_td():
    (advantage, returns) = compute_gae([1.0, 2.0], [0.5, 0.7], [0.7, 3.0], [0, 0], 1.0, 1.0)
    np.testing.assert_allclose(returns, [6.0, 5.0])
    np.testing.assert_allclose(advantage, [5.5, 4.3])
    (advantage, returns) = compute_gae([1.0, 2.0], [0.5, 0.7], [0.7, 3.0], [0, 0], 0.9, 0.0)
    np.testing.assert_allclose(advantage, [1.13, 4.0])
    np.testing.assert_allclose(returns, [1.63, 4.7])

def test_ppo_ppo_clip_uses_minimum_for_both_advantage_signs():
    ratio = torch.tensor([1.5, 0.5, 1.5, 0.5])
    advantage = torch.tensor([2.0, 2.0, -2.0, -2.0])
    loss = clipped_policy_loss(ratio.log(), torch.zeros(4), advantage, 0.2)
    assert float(loss) == pytest.approx(0.3, abs=1e-07)

def test_ppo_tanh_log_prob_finite_at_float32_saturation():
    raw = torch.tensor([[-100.0, 100.0], [-20.0, 20.0]], requires_grad=True)
    log_prob = PPOActor.log_prob(raw, torch.zeros_like(raw), torch.zeros_like(raw))
    assert torch.isfinite(log_prob).all()
    assert (torch.tanh(raw).abs() == 1).all()
    log_prob.sum().backward()
    assert torch.isfinite(raw.grad).all()

def test_ppo_independent_eval_generator_does_not_consume_training_rng():
    torch.manual_seed(17)
    learner = PPO(3, 2, PPOConfig(hidden=8, epochs=1, minibatch_size=4))
    state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(1)
    action = learner.act(np.zeros(3), deterministic=False, generator=generator)
    assert torch.equal(state, torch.get_rng_state())
    assert np.isfinite(action).all() and np.all(np.abs(action) <= 1)
    expected = learner.act(np.zeros(3))
    assert np.array_equal(expected, learner.act(np.zeros(3)))
    assert torch.equal(state, torch.get_rng_state())

@pytest.mark.parametrize('value_clip,entropy_coef', [(None, 0.0), (0.2, 0.0), (None, 0.01)])
def test_ppo_update_finite_diagnostics_and_checkpoint_rng_reproduction(value_clip, entropy_coef):
    torch.manual_seed(5)
    cfg = PPOConfig(hidden=8, epochs=2, minibatch_size=4, target_kl=None, value_clip=value_clip, entropy_coef=entropy_coef)
    learner = PPO(3, 2, cfg)
    initial = copy.deepcopy(learner.state_dict())
    metrics = learner.update(make_rollout(learner))
    assert metrics['epochs_completed'] == 2
    assert metrics['minibatches_updated'] == 6
    assert all((np.isfinite(value) for value in metrics.values()))
    assert not torch.equal(initial['actor']['network.0.weight'], learner.state_dict()['actor']['network.0.weight'])
    rollout = make_rollout(learner)
    checkpoint = copy.deepcopy(learner.state_dict())
    rng = torch.get_rng_state().clone()
    action_a = learner.act_train(np.asarray([0.2, 0.4, 0.6], np.float32))
    metrics_a = learner.update(rollout)
    restored = PPO(3, 2, cfg)
    restored.load_state_dict(checkpoint)
    torch.set_rng_state(rng)
    action_b = restored.act_train(np.asarray([0.2, 0.4, 0.6], np.float32))
    metrics_b = restored.update(rollout)
    assert_nested_equal(action_a, action_b)
    assert metrics_a == metrics_b
    assert_nested_equal(learner.state_dict(), restored.state_dict())

def test_ppo_ppo_refuses_projected_action_and_wrong_checkpoint_config():
    learner = PPO(3, 2, PPOConfig(hidden=8, epochs=1))
    rollout = make_rollout(learner)
    rollout['action'][:] = 0.999
    with pytest.raises(ValueError, match='requested tanh'):
        learner.update(rollout)
    with pytest.raises(ValueError, match='configuration'):
        PPO(3, 2, PPOConfig(hidden=16)).load_state_dict(learner.state_dict())

def test_ppo_ppo_target_kl_can_stop_before_any_invalid_policy_step():
    learner = PPO(3, 2, PPOConfig(hidden=8, epochs=2, target_kl=1e-05))
    rollout = make_rollout(learner)
    rollout['log_prob'] += 5.0
    before = copy.deepcopy(learner.state_dict())
    metrics = learner.update(rollout)
    assert metrics['early_stopped'] == 1
    assert metrics['minibatches_updated'] == 0
    assert_nested_equal(before['actor'], learner.state_dict()['actor'])
    assert metrics['critic_epochs_completed'] == 2
    assert metrics['critic_minibatches_updated'] > 0
    assert any(not torch.equal(before['critic'][name],value)
               for name,value in learner.state_dict()['critic'].items())

@pytest.mark.parametrize('kwargs', [{'gamma': 1.1}, {'gae_lambda': -0.1}, {'value_clip': 0.0}, {'target_kl': 0.0}, {'epochs': 0}, {'entropy_coef': -0.1}])
def test_ppo_invalid_ppo_configuration(kwargs):
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)


@pytest.mark.parametrize('count', [0, 1, 3, 7, 8, 17])
def test_replay_batch_is_identical_to_chronological_single_inserts(count):
    first, second = Replay(3, 2, capacity=7), Replay(3, 2, capacity=7)
    for replay in (first, second):
        replay.add([2, 3, 4], [.3, .1], 1., [3, 4, 5], False)
        replay.add([4, 5, 6], [.5, .2], 2., [5, 6, 7], True)
    rng = np.random.default_rng(3)
    obs, actions = rng.normal(size=(count, 3)), rng.uniform(-1, 1, size=(count, 2))
    rewards, next_obs = rng.normal(size=count), rng.normal(size=(count, 3))
    terminals = np.arange(count) % 3 == 2
    first.add_batch(obs, actions, rewards, next_obs, terminals)
    for index in range(count):
        second.add(obs[index], actions[index], rewards[index], next_obs[index], terminals[index])
    assert_nested_equal(first.state_dict(), second.state_dict())


def test_gae_interleaved_environments_never_share_recursion():
    streams = np.asarray([0, 1, 0, 1])
    advantage, returns = compute_gae([1., 10., 2., 20.], [0.] * 4, [0.] * 4,
                                    [0, 0, 1, 1], 1., 1., stream_id=streams)
    np.testing.assert_array_equal(returns, [3., 30., 2., 20.])
    np.testing.assert_array_equal(advantage, returns)
    _, returns = compute_gae([1., 10., 2., 20.], [0.] * 4, [0., 0., 4., 5.],
                             [0, 0, 0, 1], 1., 1., stream_id=streams)
    np.testing.assert_array_equal(returns, [7., 30., 6., 20.])
    with pytest.raises(ValueError, match='stream_id'):
        compute_gae([1], [0], [0], [1], stream_id=np.array([0.5]))


def test_ppo_update_uses_streamwise_gae():
    learner = PPO(3, 2, PPOConfig(hidden=8, epochs=1, minibatch_size=4,
                                  gae_lambda=1., target_kl=None))
    obs = np.zeros((4, 3), np.float32)
    actions, log_prob, values, raw = learner.act_train_batch(obs)
    rollout = dict(observation=obs, action=actions, log_prob=log_prob, value=values,
                   raw_action=raw, reward=np.array([1., 10., 2., 20.]),
                   next_value=np.array([values[2], values[3], 0., 0.]),
                   terminated=np.array([0, 0, 1, 1]), stream_id=np.array([0, 1, 0, 1]))
    metrics = learner.update(rollout)
    assert metrics['return_mean_gae'] == pytest.approx(13.75, abs=1e-6)


def small_learner(algorithm, actions=6, device='cpu'):
    if algorithm == 'sac':
        return SAC(7, actions, SACConfig(hidden=16, batch_size=4, replay_capacity=32), device)
    if algorithm == 'ppo':
        return PPO(7, actions, PPOConfig(hidden=16, epochs=1, minibatch_size=4,
                                        target_kl=None), device)
    return DeterministicLearner(7, actions,
        DeterministicConfig(algorithm=algorithm, hidden=16, batch_size=4,
                             replay_capacity=32), device)


@pytest.mark.parametrize('algorithm', ['sac', 'td3', 'ddpg', 'ppo'])
@pytest.mark.parametrize('actions', [6, 146])
def test_vector_actions_are_bounded_and_generator_is_separate(algorithm, actions):
    learner = small_learner(algorithm, actions)
    observations = np.arange(28, dtype=np.float32).reshape(4, 7) / 28.
    state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(5)
    batch = learner.act_batch(observations, deterministic=False, generator=generator)
    assert batch.shape == (4, actions)
    assert np.isfinite(batch).all() and np.all(np.abs(batch) <= 1.)
    assert torch.equal(state, torch.get_rng_state())
    # A batched matrix multiply can differ from single-row BLAS by roundoff.
    np.testing.assert_allclose(learner.act_batch(observations),
        np.stack([learner.act(row) for row in observations]), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='shape'):
        learner.act_batch(np.zeros((3, 8), dtype=np.float32))


@pytest.mark.parametrize('algorithm', ['sac', 'td3', 'ddpg', 'ppo'])
def test_high_dimensional_original_learner_updates_and_resume(algorithm):
    seed_everything(19)
    learner = small_learner(algorithm, 146)
    observations = np.arange(56, dtype=np.float32).reshape(8, 7) / 56.
    replay = Replay(7, 146, 32, seed=3)
    replay.add_batch(observations, np.zeros((8, 146)), -np.arange(8) / 8.,
                      observations + .1, np.arange(8) % 2 == 1)
    if algorithm == 'ppo':
        def next_metrics(current):
            action, log_prob, value, raw = current.act_train_batch(observations)
            return current.update(dict(observation=observations, action=action,
                log_prob=log_prob, value=value, raw_action=raw,
                reward=-np.arange(8) / 8., terminated=np.arange(8) % 2 == 1,
                next_value=current.value_batch(observations + .1),
                stream_id=np.arange(8) % 2))
    else:
        def next_metrics(current):
            return current.update(replay)
    next_metrics(learner)
    checkpoint, replay_state = learner.state_dict(), replay.state_dict()
    rng = torch.get_rng_state().clone()
    expected_metrics = next_metrics(learner)
    expected_state = learner.state_dict()
    restored = small_learner(algorithm, 146)
    restored.load_state_dict(checkpoint)
    replay.load_state_dict(replay_state)
    torch.set_rng_state(rng)
    actual_metrics = next_metrics(restored)
    assert_nested_equal(expected_metrics, actual_metrics)
    assert_nested_equal(expected_state, restored.state_dict())
    assert all(math.isfinite(value) for value in actual_metrics.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
@pytest.mark.parametrize('algorithm', ['sac', 'td3', 'ddpg', 'ppo'])
def test_cuda_actions_updates_and_checkpoint_optimizer_device(algorithm):
    learner = small_learner(algorithm, 146, 'cuda')
    obs = np.zeros((8, 7), np.float32)
    generator = torch.Generator(device='cuda').manual_seed(4)
    action = learner.act_batch(obs, deterministic=False, generator=generator)
    assert action.shape == (8, 146) and np.isfinite(action).all()
    if algorithm == 'ppo':
        action, logp, values, raw = learner.act_train_batch(obs)
        data = dict(observation=obs, action=action, raw_action=raw, log_prob=logp,
                    value=values, next_value=np.zeros(8), reward=-np.ones(8),
                    terminated=np.ones(8), stream_id=np.arange(8))
    else:
        data = Replay(7, 146, 32)
        data.add_batch(obs, action, -np.ones(8), obs, np.ones(8))
    metrics = learner.update(data)
    assert all(math.isfinite(value) for value in metrics.values())
    # This exercises host-mapped serialization then loading Adam back on CUDA.
    import io
    buffer = io.BytesIO()
    torch.save(learner.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location='cpu', weights_only=False)
    restored = small_learner(algorithm, 146, 'cuda')
    restored.load_state_dict(state)
    metrics = restored.update(data)
    assert all(math.isfinite(value) for value in metrics.values())
