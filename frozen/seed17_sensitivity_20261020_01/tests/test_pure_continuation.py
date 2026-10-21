"""State-only pure PPO continuation, exact final masking, and PopArt targets."""
import copy

import numpy as np
import pytest
import torch

from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.rl.ppo import Actor, LEARNER_VERSION, PPO, PPOConfig
from priority_mec.rl.value_components import (
    PopArtContinuationCritic, PopArtLearnedContinuationCritic,
    component_value_loss, compute_component_gae, reward_components)


torch.set_num_threads(1)


def _learned_config(**overrides):
    values = dict(hidden=16, epochs=1, critic_epochs=1, minibatch_size=4,
                  rollout_steps=4, target_kl=None, variant='value_decomp_popart',
                  continuation_critic=True, continuation_mode='learned')
    values.update(overrides)
    return PPOConfig(**values)


@pytest.mark.parametrize('overrides', [
    {'continuation_mode': 'other'}, {'variant': 'baseline'},
    {'known_delay_index': 2}, {'continuation_critic': False},
])
def test_learned_continuation_rejects_inconsistent_configuration(overrides):
    with pytest.raises(ValueError):
        _learned_config(**overrides)


def test_continuation_modes_require_their_information_and_actual_horizon():
    with pytest.raises(ValueError, match='observed current delay'):
        PPOConfig(variant='value_decomp_popart', continuation_critic=True)
    for physics in (None, {}, {'episode_frames': 0}, {'episode_frames': True}):
        with pytest.raises(ValueError, match='horizon'):
            PPO(8, 146, _learned_config(), physics=physics)
    with pytest.raises(ValueError, match='Unknown PPO'):
        PPOConfig(variant='relation_film')


@pytest.mark.parametrize('motion_memory', [False, True])
def test_pure_critic_uses_the_existing_observed_time_without_a_cost_feature(motion_memory):
    observation_size = 14 if motion_memory else 8
    learner = PPO(observation_size, 146,
                  _learned_config(motion_memory=motion_memory),
                  physics={'episode_frames': 4})
    assert learner.cfg.known_delay_index is None
    assert learner.critic.time_index == 7
    if motion_memory:
        assert learner.actor.memory_start_index == 8
    obs = np.zeros((2, observation_size), np.float32)
    obs[:, 7] = [.5, .25]
    actions, logp, values, raw, components = learner.act_train_batch(
        obs, return_components=True)
    assert actions.shape == raw.shape == (2, 146)
    assert components.shape == (2, 2)
    assert np.array_equal(values, components.sum(-1))
    assert np.isfinite(logp).all()
    internal = learner.critic.regression_components(torch.from_numpy(obs))
    assert internal.shape == (2, 3)
    assert internal[1, 1].item() == 0.
    torch.testing.assert_close(torch.from_numpy(components[:, 0]),
                               internal[:, 0] + internal[:, 1])


def test_current_future_targets_reconstruct_the_same_component_gae_returns():
    rewards = np.array([[-2., -.1], [-3., -.2], [-5., -.3]], np.float32)
    values = np.array([[-4., -.2], [-5., -.3], [-2., -.1]], np.float32)
    gae = compute_component_gae(rewards, values, np.roll(values, -1, axis=0),
                                [0, 0, 1], gamma=1., gae_lambda=1.)
    expected_returns = np.array([[-10., -.6], [-8., -.5], [-5., -.3]], np.float32)
    np.testing.assert_allclose(gae.component_returns, expected_returns, atol=1e-6)
    targets = PopArtLearnedContinuationCritic.regression_targets(
        torch.from_numpy(gae.component_returns), torch.from_numpy(rewards[:, 0]))
    torch.testing.assert_close(targets[:, 0], torch.tensor([-2., -3., -5.]))
    torch.testing.assert_close(targets[:, 1], torch.tensor([-8., -5., 0.]))
    torch.testing.assert_close(targets[:, 0] + targets[:, 1],
                               torch.from_numpy(gae.component_returns[:, 0]))
    torch.testing.assert_close(targets[:, 2], torch.from_numpy(rewards[:, 1]).flip(0).cumsum(0).flip(0))


def test_final_service_slot_future_value_and_its_gradients_are_exactly_zero():
    critic = PopArtLearnedContinuationCritic(3, 8, time_index=2, episode_frames=60)
    with torch.no_grad():
        critic.heads.bias.copy_(torch.tensor([-2., -7., -.4]))
    obs = torch.tensor([[0., 0., 1. / 60], [0., 0., 0.]])
    internal = critic.regression_components(obs)
    assert torch.equal(internal[:, 1], torch.zeros(2))
    torch.testing.assert_close(critic.forward_components(obs)[:, 0], internal[:, 0])
    internal[:, 1].sum().backward()
    assert torch.count_nonzero(critic.heads.weight.grad).item() == 0
    assert torch.count_nonzero(critic.heads.bias.grad).item() == 0


def test_three_regressions_have_gradients_and_popart_transports_transformed_statistics():
    torch.manual_seed(7)
    critic = PopArtLearnedContinuationCritic(3, 8, beta=1., time_index=2, episode_frames=4)
    obs = torch.tensor([[.2, -.3, 1.], [.4, .5, .75], [-.4, .1, .25]])
    returns = torch.tensor([[-10., -.6], [-8., -.5], [-5., -.3]])
    immediate = torch.tensor([-2., -3., -5.])
    targets = critic.regression_targets(returns, immediate)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3, amsgrad=True)
    loss = component_value_loss(critic.regression_components(obs), targets)
    loss.backward()
    assert (critic.heads.weight.grad.abs().sum(1) > 0).all()
    optimizer.step()
    public_before = critic.forward_components(obs).detach().clone()
    internal_before = critic.regression_components(obs).detach().clone()
    old_std = critic.target_std.clone()
    state_before = copy.deepcopy(optimizer.state[critic.heads.weight])
    critic.update_statistics(returns, optimizer, observations=obs,
                             immediate_delay_rewards=immediate)
    torch.testing.assert_close(critic.target_mean, targets.double().mean(0))
    torch.testing.assert_close(critic.target_second, targets.double().square().mean(0))
    torch.testing.assert_close(critic.forward_components(obs), public_before, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(critic.regression_components(obs), internal_before, atol=2e-6, rtol=2e-6)
    ratio = (old_std / critic.target_std).float()[:, None]
    for name, power in [('exp_avg', 1), ('exp_avg_sq', 2), ('max_exp_avg_sq', 2)]:
        torch.testing.assert_close(optimizer.state[critic.heads.weight][name],
                                   state_before[name] * ratio ** power)
    with pytest.raises(ValueError, match='immediate delay'):
        critic.update_statistics(returns)


@pytest.mark.parametrize('adaptive', [False, True])
def test_pure_autoregression_changes_only_six_motion_distribution_coordinates(adaptive):
    actor = Actor(14, 146, 8, motion_memory=True, ar_rho=.5,
                  memory_start_index=8, ar_adaptive=adaptive)
    with torch.no_grad():
        for parameter in actor.network.parameters():
            parameter.zero_()
        actor.network[-1].bias.fill_(.2)
    obs = torch.zeros(2, 14)
    obs[:, 8:] = torch.tensor([.3, -.2, .5, -.4, .7, -.6])
    mean, log_std = actor(obs)
    torch.testing.assert_close(mean[:, :6], .5 * obs[:, 8:] + .1)
    torch.testing.assert_close(mean[:, 6:], torch.full((2, 140), .2))
    torch.testing.assert_close(log_std[:, 6:], torch.full((2, 140), -.5))
    if adaptive:
        raw = torch.full_like(mean, .8)
        (-actor.log_prob(raw, mean, log_std).mean()).backward()
        assert actor.rho_bias.grad.abs().sum() > 0
        assert actor.rho_weight.grad.abs().sum() > 0


def _collect_pure_rollout(learner, env):
    base = env.reset()
    previous_motion = np.zeros(6, np.float32)
    records = {name: [] for name in ('obs', 'action', 'raw_action', 'log_prob',
               'value', 'reward', 'terminated', 'next_value', 'reward_components',
               'value_components', 'next_value_components')}
    for _ in range(env.cfg.episode_frames):
        obs = np.concatenate((base, previous_motion)) if learner.cfg.motion_memory else base
        action, logp, value, raw, components = learner.act_train_batch(
            obs[None], return_components=True)
        next_base, reward, terminal, truncated, info = env.step(action[0])
        assert not truncated
        next_obs = (np.concatenate((next_base, raw[0, :6]))
                    if learner.cfg.motion_memory else next_base)
        next_components = learner.value_components_batch(next_obs[None])[0]
        row = dict(obs=obs, action=action[0], raw_action=raw[0], log_prob=logp[0],
                   value=value[0], reward=reward, terminated=terminal,
                   next_value=next_components.sum(), value_components=components[0],
                   next_value_components=next_components,
                   reward_components=reward_components([info], [reward])[0])
        for key in records:
            records[key].append(row[key])
        base, previous_motion = next_base, raw[0, :6].copy()
    return {key: np.asarray(value, np.float32) for key, value in records.items()}


@pytest.mark.parametrize('policy', ['plain', 'fixed', 'adaptive'])
@pytest.mark.parametrize('value_clip', [None, .2])
def test_real_pure_rollout_update_has_no_solver_or_preaction_cost(monkeypatch, policy, value_clip):
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure PPO must never prepare or solve a service allocation')

    monkeypatch.setattr('priority_mec.solvers.solve', forbidden)
    monkeypatch.setattr(UAVMECEnv, 'prepare_service', forbidden)
    torch.manual_seed(31)
    env = UAVMECEnv(Config(episode_frames=4), seed=8, control='pure')
    motion = policy != 'plain'
    cfg = _learned_config(motion_memory=motion, ar_rho=.5 if motion else 0.,
                          ar_adaptive=policy == 'adaptive', value_clip=value_clip)
    learner = PPO(env.observation_size + (6 if motion else 0), 146, cfg,
                  physics=env.cfg.to_dict())
    rollout = _collect_pure_rollout(learner, env)
    # Pre-action values are pure state functions, also when new actions differ.
    _, _, _, _, first = learner.act_train_batch(rollout['obs'], return_components=True)
    _, _, _, _, second = learner.act_train_batch(rollout['obs'], return_components=True)
    np.testing.assert_array_equal(first, second)
    actor_before = copy.deepcopy(learner.actor.state_dict())
    metrics = learner.update(rollout)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['critic_minibatches_updated'] == 1
    assert metrics['popart_std_current_delay'] > 0
    assert metrics['popart_std_future_delay'] > 0
    assert any(not torch.equal(value, actor_before[key])
               for key, value in learner.actor.state_dict().items())
    future = learner.critic.regression_components(torch.from_numpy(rollout['obs']))[:, 1]
    assert future[-1].item() == 0.


def test_pure_baseline_cannot_read_future_events_or_action_dependent_current_cost():
    cfg = Config(episode_frames=4)
    local_env = UAVMECEnv(cfg, seed=8, control='pure')
    remote_env = UAVMECEnv(cfg, seed=8, control='pure')
    learner = PPO(local_env.observation_size, 146, _learned_config(), physics=cfg.to_dict())
    obs = local_env.reset()
    np.testing.assert_array_equal(obs, remote_env.reset())
    prediction = learner.value_components_batch(obs[None])
    # Even changing every unobserved future task leaves the state baseline fixed.
    remote_env.scenario.data_bit[1:] *= 1.5
    remote_env.scenario.deadline_s[1:] *= .5
    remote_env.scenario.route_uniform[1:] = 1. - remote_env.scenario.route_uniform[1:]
    np.testing.assert_array_equal(obs, remote_env._observation())
    np.testing.assert_array_equal(prediction,
                                 learner.value_components_batch(remote_env._observation()[None]))
    local_action, remote_action = np.zeros(146), np.zeros(146)
    for action, selected_node in [(local_action, 0), (remote_action, 1)]:
        route_preferences = action[6:106].reshape(20, 5)
        route_preferences[:] = -1.
        route_preferences[:, selected_node] = 1.
    local_info = local_env.step(local_action)[-1]
    remote_info = remote_env.step(remote_action)[-1]
    assert local_info['normalized_weighted_delay'] != remote_info['normalized_weighted_delay']
    # One observed state has different realized costs under different resources;
    # W therefore estimates an expectation and never copies a realized cost.
    np.testing.assert_array_equal(prediction, learner.value_components_batch(obs[None]))


def test_learned_checkpoint_restores_three_heads_optimizer_and_active_schedule():
    torch.manual_seed(41)
    cfg = _learned_config(motion_memory=True, ar_rho=.5, ar_adaptive=True,
                          schedule_steps=100, noise_final_scale=.2,
                          actor_lr_final_scale=.4)
    env = UAVMECEnv(Config(episode_frames=4), seed=8, control='pure')
    learner = PPO(env.observation_size + 6, 146, cfg, physics=env.cfg.to_dict())
    learner.set_training_progress(60)
    rollout = _collect_pure_rollout(learner, env)
    learner.update(rollout)
    state = learner.state_dict()
    assert state['version'] == LEARNER_VERSION == 6
    restored = PPO(env.observation_size + 6, 146, cfg, physics=env.cfg.to_dict())
    restored.load_state_dict(state)
    obs = rollout['obs']
    np.testing.assert_array_equal(learner.act_batch(obs), restored.act_batch(obs))
    np.testing.assert_array_equal(learner.value_components_batch(obs), restored.value_components_batch(obs))
    assert learner.policy_diagnostics(obs) == restored.policy_diagnostics(obs)
    assert restored.critic_optimizer.state_dict()['state']
    assert restored.critic.target_mean.shape == (3,)
    # Full optimizer/RNG restoration yields the same subsequent update.
    random_state = torch.get_rng_state().clone()
    learner.update(rollout)
    torch.set_rng_state(random_state)
    restored.update(rollout)
    for key, value in learner.critic.state_dict().items():
        assert torch.equal(value, restored.critic.state_dict()[key])
    old = copy.deepcopy(state)
    old['version'] = 5
    with pytest.raises(ValueError, match='version'):
        restored.load_state_dict(old)


def test_known_hybrid_continuation_keeps_exact_current_delay_and_two_heads():
    cfg = PPOConfig(hidden=8, variant='value_decomp_popart', continuation_critic=True,
                    known_delay_index=8, motion_memory=False)
    learner = PPO(9, 6, cfg, physics={'episode_frames': 4})
    assert isinstance(learner.critic, PopArtContinuationCritic)
    obs = torch.zeros(2, 9)
    obs[:, 7] = torch.tensor([.5, .25])
    obs[:, 8] = torch.tensor([2., 3.])
    continuation = learner.critic.continuation_components(obs)
    components = learner.critic.forward_components(obs)
    torch.testing.assert_close(components[:, 0], continuation[:, 0] - obs[:, 8])
    assert components[-1, 0].item() == -3.
    assert learner.critic.target_mean.shape == (2,)
