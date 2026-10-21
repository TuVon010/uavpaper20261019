"""Physical regression against the preserved project and greedy control pairing."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np

from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv
from priority_mec.greedy_baseline import plan_velocity


PHYSICAL_PROBE = r'''
import json
import numpy as np
from priority_mec.config import Config
from priority_mec.environment import UAVMECEnv

def exact(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), bytes=value.tobytes().hex())
    if isinstance(value, np.generic):
        return exact(value.item())
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(item) for item in value]
    return value

cfg = Config()
env = UAVMECEnv(cfg, seed=17, control='hybrid')
initial = env.reset(episode=23)
payload = dict(config=cfg.to_dict(), config_fingerprint=cfg.fingerprint(),
    scenario_hash=env.scenario.hash,
    initial=exact(dict(observation=initial, user_position=env.user_position,
        uav_position=env.uav_position, rho_hz=env.rho_hz, offset_s=env.offset_s)), rows=[])
actions = [[.2,-.4,.7,.1,-.3,.8],[-.6,.5,0.,-.8,.1,.2],[0.,0.,0.,0.,0.,0.]]
keys = ('normalized_weighted_delay','weighted_delay_sum','mean_delay_s',
    'deadline_completion','deadline_violation_cost','energy_violation_cost',
    'energy_j','propulsion_energy_j','compute_energy_j','cumulative_energy_j',
    'cumulative_propulsion_energy_j','cumulative_compute_energy_j',
    'association','bandwidth_hz','cpu_hz','delay_s','raw_delay_s',
    'allocation_invalid_mask','user_position','uav_position',
    'next_user_position','next_uav_position','uav_velocity','requested_velocity_mps')
for action in actions:
    observation, reward, terminated, truncated, info = env.step(np.asarray(action))
    row = {key: info[key] for key in keys}
    row.update(observation=observation,reward=reward,terminated=terminated,truncated=truncated,
        scenario_hash=info['scenario_hash'], frame_scenario_hash=info['frame_scenario_hash'],
        next_rho_hz=env.rho_hz, next_offset_s=env.offset_s)
    payload['rows'].append(exact(row))
print(json.dumps(payload, sort_keys=True, allow_nan=False))
'''


def physical_probe(project):
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(project)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', '-c', PHYSICAL_PROBE], cwd=project,
        env=environment, check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


class FinalEnvironmentTests(unittest.TestCase):
    def test_default_hybrid_is_bitwise_equal_to_preserved_priority_project(self):
        project = Path(__file__).resolve().parents[1]
        preserved = project.parent.parent / 'uav_mec_priority'
        if not (preserved / 'priority_mec' / 'environment.py').exists():
            self.skipTest('Preserved priority project is not present on this machine')
        original = physical_probe(preserved)
        final = physical_probe(project)
        additions = {'user_cluster_halfwidth_m', 'user_cluster_center_x_fraction',
            'user_cluster_center_y_fraction', 'greedy_direction_count',
            'greedy_speed_levels', 'greedy_refine_rounds', 'delay_cost_weight',
            'uav_service_cutoff_enabled'}
        self.assertEqual(set(final['config']) - set(original['config']), additions)
        self.assertEqual(original['config'], {key: final['config'][key] for key in original['config']})
        # The configuration fingerprint expands with its newly editable fields.
        self.assertNotEqual(original['config_fingerprint'], final['config_fingerprint'])
        # Scenario hashes digest realized exogenous arrays, so their identities stay equal.
        self.assertEqual(original['scenario_hash'], final['scenario_hash'])
        self.assertEqual(original['initial'], final['initial'])
        self.assertEqual(original['rows'], final['rows'])

    def test_greedy_pair_shares_geometry_but_uses_distinct_allocations(self):
        cfg = Config()
        rate = UAVMECEnv(cfg, seed=17, control='greedy_rate')
        solver = UAVMECEnv(cfg, seed=17, control='greedy_solver')
        rate.reset(episode=23)
        solver.reset(episode=23)
        allocation_differences = []
        from priority_mec import greedy_baseline, solvers
        with patch.object(greedy_baseline, 'greedy_allocation', wraps=greedy_baseline.greedy_allocation) as greedy, \
                patch.object(solvers, 'solve', wraps=solvers.solve) as optimized:
            for frame in range(3):
                _, _, _, _, left = rate.step()
                _, _, _, _, right = solver.step()
                for key in ('requested_velocity_mps', 'uav_velocity', 'next_uav_position',
                            'next_user_position', 'propulsion_energy_j'):
                    np.testing.assert_array_equal(left[key], right[key], err_msg=f'{key}, frame {frame}')
                self.assertEqual(left['motion_diagnostics'], right['motion_diagnostics'])
                self.assertEqual(left['scenario_hash'], right['scenario_hash'])
                self.assertEqual(left['frame_scenario_hash'], right['frame_scenario_hash'])
                allocation_differences.append(any(left[key] != right[key] for key in
                    ('association', 'bandwidth_hz', 'cpu_hz')))
            self.assertEqual(greedy.call_count, 3)
            self.assertEqual(optimized.call_count, 3)
        self.assertTrue(all(allocation_differences))

    def test_greedy_flight_does_not_use_energy_budget_accounting_or_penalty(self):
        cfg = Config()
        env = UAVMECEnv(cfg, seed=17, control='greedy_rate')
        env.reset(episode=23)
        altered = replace(cfg, energy_budget_j=.001, uav_joule_per_cycle=100.,
            energy_penalty=1234., rotor_p0_w=1e9, rotor_pi_w=1e9)
        inputs = (env.user_position, env.uav_position,
            1. / env.scenario.deadline_s[0], env.rho_hz)
        first, diagnostics = plan_velocity(*inputs, cfg)
        second, changed_diagnostics = plan_velocity(*inputs, altered)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(diagnostics, changed_diagnostics)
        # Cumulative accounting changes the observation/reward only, never this planner.
        clone = UAVMECEnv(cfg, seed=17, control='greedy_rate')
        clone.reset(episode=23)
        clone.cumulative_energy_j[:] = 100. * cfg.energy_budget_j
        left, right = env.step()[4], clone.step()[4]
        np.testing.assert_array_equal(left['requested_velocity_mps'], right['requested_velocity_mps'])
        np.testing.assert_array_equal(left['uav_velocity'], right['uav_velocity'])
        self.assertNotEqual(left['cumulative_energy_j'], right['cumulative_energy_j'])


if __name__ == '__main__':
    unittest.main()
