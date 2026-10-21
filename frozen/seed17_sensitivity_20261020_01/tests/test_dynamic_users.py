"""User-count sweeps retain exact task accounting and the N=20 execution rule."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.episode_log import EpisodeAccumulator, normalized_delay
from priority_mec.training import evaluate


@pytest.mark.parametrize('count', [0, -1, True, False, 10.5, '20', None])
def test_user_count_requires_a_positive_integer(count):
    with pytest.raises(ValueError, match='num_users'):
        Config(num_users=count).validate()


@pytest.mark.parametrize('count', [1, 10, 15, 20, 25, 30])
def test_user_count_is_configurable_without_changing_uav_count(count):
    assert Config(num_users=count).validate().num_users == count
    for invalid in (True, False, 2, 4, 3.):
        with pytest.raises(ValueError, match='three UAVs'):
            Config(num_users=count, num_uavs=invalid).validate()


@pytest.mark.parametrize('count', [10, 15, 20, 25, 30])
@pytest.mark.parametrize('control', ['pure', 'greedy_rate', 'hybrid'])
def test_real_steps_resources_prepared_service_and_resume_at_each_size(count, control):
    cfg = Config(num_users=count, episode_frames=2, cycles_per_bit=1000.,
                 user_cluster_halfwidth_m=250., uav_service_cutoff_enabled=True)
    env = UAVMECEnv(cfg, seed=17, control=control)
    observation = env.reset(23)
    assert observation.shape == (18*count+16,)
    assert env.action_size == (6+7*count if control == 'pure' else 6)
    accumulator = EpisodeAccumulator(23, 0, num_users=count)
    for slot in (1, 2):
        # Prepared allocation and its checkpoint cache must work for every N.
        if control == 'hybrid':
            known_cost = env.prepare_service()
        state = env.state_dict()
        clone = UAVMECEnv(cfg, seed=17, control=control)
        clone.load_state_dict(state)
        action = np.linspace(-.7, .7, env.action_size)
        left, right = env.step(action), clone.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:4] == right[1:4]
        info = left[4]
        assert info['num_users'] == count
        if control == 'hybrid':
            assert info['delay_reward_cost'] == known_cost
        for key in ('association', 'bandwidth_hz', 'cpu_hz', 'delay_s',
                    'cumulative_energy_j', 'frame_scenario_hash'):
            assert info[key] == right[4][key]
        association = np.asarray(info['association'])
        bandwidth, cpu = np.asarray(info['bandwidth_hz']), np.asarray(info['cpu_hz'])
        assert len(association) == count and sum(info['association_counts']) == count
        for node in range(1, 5):
            selected = association == node
            assert bandwidth[selected].sum() <= (cfg.uav_bandwidth_hz if node <= 3 else cfg.satellite_bandwidth_hz)*(1+cfg.feasibility_tol)
            assert cpu[selected].sum() <= (cfg.uav_cpu_hz if node <= 3 else cfg.satellite_cpu_hz)*(1+cfg.feasibility_tol)
        assert info['weighted_delay_sum'] == pytest.approx(count*info['normalized_weighted_delay'])
        accumulator.add(left[1], info, slot)
        accumulator = EpisodeAccumulator.from_state_dict(json.loads(json.dumps(accumulator.state_dict())))
    row = accumulator.record(2, cfg.energy_budget_j)
    assert row['num_users'] == count
    assert row['weighted_delay_sum_per_frame'] == count*row['weighted_delay_mean']
    assert accumulator.state_dict()['num_users'] == count


@pytest.mark.parametrize('count', [10, 30])
def test_evaluation_delay_sum_uses_configured_user_count(count):
    cfg = Config(num_users=count, episode_frames=2)
    result = evaluate(cfg, control='pure', episodes=1, episode_start=23, seed=17)
    row = result['episodes'][0]
    assert row['weighted_delay_sum_per_frame'] == count*row['weighted_delay_mean']
    assert result['summary']['environment_config']['num_users'] == count


def test_sum_only_delay_logs_require_an_explicit_or_inferred_count():
    assert normalized_delay({'weighted_delay_sum': 15., 'num_users': 30}) == .5
    assert normalized_delay({'weighted_delay_sum': 15., 'association': [0]*30}) == .5
    assert normalized_delay({'weighted_delay_sum': 15.}, 30) == .5
    with pytest.raises(ValueError, match='task count'):
        normalized_delay({'weighted_delay_sum': 15.})
    accumulator = EpisodeAccumulator(1, 0, num_users=10)
    with pytest.raises(ValueError, match='User count changed'):
        accumulator.add(0., {'num_users': 30}, 1)


# Optional standalone source regression: point at the retained frozen release.
# It runs each version in its own interpreter; no module aliasing or relaxed
# checkpoint/source checks are used. This test does not copy legacy code.
N20_PROBE = r'''
import json, sys
import numpy as np
from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
def exact(value):
    if isinstance(value,np.ndarray):
        return dict(dtype=value.dtype.str,shape=list(value.shape),bytes=value.tobytes().hex())
    if isinstance(value,np.generic):return exact(value.item())
    if isinstance(value,float):return dict(float_hex=value.hex())
    if isinstance(value,dict):return {k:exact(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [exact(x) for x in value]
    return value
cfg=Config(episode_frames=3,cycles_per_bit=1000.,user_cluster_halfwidth_m=250.,
           delay_cost_weight=5.,uav_service_cutoff_enabled=True)
env=UAVMECEnv(cfg,seed=17,control=sys.argv[1]);obs=env.reset(23)
np.random.seed(1729)
payload=dict(initial=exact(obs),scenario=env.scenario.hash,rows=[])
keys=('association','bandwidth_hz','cpu_hz','delay_s','raw_delay_s','allocation_invalid_mask',
      'normalized_weighted_delay','weighted_delay_sum','mean_delay_s','deadline_completion',
      'energy_j','propulsion_energy_j','compute_energy_j','cumulative_energy_j',
      'next_uav_position','next_user_position','uav_velocity','requested_velocity_mps',
      'frame_scenario_hash','uav_service_available')
for slot in range(3):
    if slot==1:env.cumulative_energy_j[0]=cfg.energy_budget_j
    known=env.prepare_service() if sys.argv[1]=='hybrid' else None
    action=np.roll(np.linspace(-.8,.8,env.action_size),slot)
    obs,reward,terminated,truncated,info=env.step(action)
    row={k:info[k] for k in keys}
    row.update(obs=obs,reward=reward,terminated=terminated,truncated=truncated,
               rho=env.rho_hz,offset=env.offset_s,known=known)
    payload['rows'].append(exact(row))
payload['global_rng']=exact(np.random.get_state())
print(json.dumps(payload,sort_keys=True,allow_nan=False))
'''


@pytest.mark.parametrize('control', ['pure', 'hybrid', 'greedy_rate', 'greedy_solver'])
def test_n20_frozen_release_physics_actions_rewards_and_rng_are_bitwise_equal(control):
    reference = os.environ.get('MEC_USER_COUNT_REFERENCE')
    if not reference:
        pytest.skip('Set MEC_USER_COUNT_REFERENCE to the retained frozen N=20 release')
    current = Path(__file__).resolve().parents[1]
    payloads = []
    for project in (Path(reference), current):
        environment = dict(os.environ, PYTHONPATH=str(project), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', '-c', N20_PROBE, control],
            cwd=project, env=environment, check=True, capture_output=True, text=True, timeout=120)
        payloads.append(json.loads(result.stdout))
    assert payloads[0] == payloads[1]
