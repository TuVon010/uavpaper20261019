"""Episode summaries survive partial vector episodes and checkpoint rollback."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import random
import numpy as np
import torch

from priority_mec.config import Config
from priority_mec.episode_log import EpisodeAccumulator
from priority_mec.environment import UAVMECEnv
from priority_mec.io import append_csv, load_checkpoint
from priority_mec.training import TrainConfig, _trim_history, learner_for, train


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def without_timings(rows):
    return [{k: v for k, v in row.items() if not k.endswith('_s') or k == 'mean_delay_s'}
            for row in rows]


class EpisodeLoggingTests(unittest.TestCase):
    def test_json_ppo_knobs_reach_learner_and_invalid_counts_fail_early(self):
        options=json.loads('{"algorithm":"ppo","device":"cpu","hidden":8,'
            '"ppo_clip_ratio":0.11,"ppo_value_clip":0.3,"ppo_target_kl":null,'
            '"ppo_entropy_coef":0.004,"ppo_value_coef":0.7,"ppo_max_grad_norm":0.9}')
        learner=learner_for(376,146,TrainConfig(**options),Config().to_dict())
        for key,value in options.items():
            if key.startswith('ppo_'):
                self.assertEqual(getattr(learner.cfg,key.removeprefix('ppo_')),value)
        default=TrainConfig(algorithm='ppo',device='cpu')
        self.assertEqual((default.ppo_clip_ratio,default.ppo_value_clip,default.ppo_target_kl,
            default.ppo_entropy_coef,default.ppo_value_coef,default.ppo_max_grad_norm),
            (.2,None,.02,0.,.5,.5))
        for key,value in (('num_envs',0),('num_envs',1.5),('hidden',False),('batch_size',0),
                ('eval_episodes',0),('eval_start',-1),('ppo_epochs',0),('ppo_critic_epochs',1.5),
                ('ppo_minibatch_size',0),('ppo_clip_ratio',1.),('ppo_value_clip',0.),
                ('ppo_target_kl',float('nan')),('ppo_entropy_coef',-.1),
                ('ppo_value_coef',0.),('ppo_max_grad_norm',0.)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                TrainConfig(**{key:value})

    def test_accumulator_preserves_partial_metrics_and_final_energy(self):
        def info(frame):
            return dict(scenario_hash='same-scenario', normalized_weighted_delay=frame,
                mean_delay_s=frame / 2, deadline_completion=1. if frame == 1 else .5,
                deadline_violation_cost=frame / 4, allocation_invalid_rate=0.,
                solver_feasible=frame == 1, fallback_rate=0., collision_attempt=frame == 2,
                boundary_attempt=False, projection_failed=False,
                cumulative_energy_j=[10. * frame, 20. * frame, 30. * frame],
                cumulative_propulsion_energy_j=[6. * frame, 12. * frame, 18. * frame],
                cumulative_compute_energy_j=[4. * frame, 8. * frame, 12. * frame],
                decision_wall_s=.02 * frame)
        accumulator = EpisodeAccumulator(7, 1)
        accumulator.add(-2., info(1), 11, .01)
        restored = EpisodeAccumulator.from_state_dict(accumulator.state_dict())
        restored.add(-3., info(2), 13, .01)
        row = restored.record(13, 50.)
        self.assertEqual((row['episode'], row['start_step'], row['end_step'], row['frames']), (7, 11, 13, 2))
        self.assertEqual(row['episode_return'], -5.)
        self.assertEqual(row['weighted_delay_mean'], 1.5)
        self.assertEqual(row['mean_delay_s'], .75)
        self.assertEqual(row['deadline_completion'], .75)
        self.assertEqual(row['per_uav_energy_j'], [20., 40., 60.])
        self.assertEqual(row['total_energy_j'], 120.)
        self.assertEqual(row['total_propulsion_energy_j'] + row['total_compute_energy_j'], 120.)
        self.assertEqual(row['collision_attempt_frames'], 1)
        self.assertFalse(row['budget_feasible'])
        self.assertFalse(row['deadline_feasible'])
        self.assertFalse(row['request_safe'])
        self.assertAlmostEqual(row['total_decision_mean_s'], .04)
        self.assertAlmostEqual(row['total_decision_p95_s'], .049)

    def test_trim_uses_episode_end_step_and_preserves_nested_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name, key in (('episodes.csv', 'end_step'), ('train.csv', 'step'), ('validation.csv', 'step')):
                for step in (2, 4, 6):
                    append_csv(out / name, {key: step, 'value': step * 2})
            (out / 'updates.jsonl').write_text('{"step":2}\n{"step":6}\n{"ste', encoding='utf-8')
            validation = out / 'validation'
            validation.mkdir()
            (validation / 'step_00000002.json').write_text('{}')
            (validation / 'step_00000006.json').write_text('{}')
            evaluation = out / 'evaluation_last'
            evaluation.mkdir()
            append_csv(evaluation / 'episodes.csv', {'episode': 30000, 'frames': 60})
            _trim_history(out, 4)
            self.assertEqual([r['end_step'] for r in read_csv(out / 'episodes.csv')], ['2', '4'])
            self.assertEqual([r['step'] for r in read_csv(out / 'train.csv')], ['2', '4'])
            self.assertEqual([r['step'] for r in read_csv(out / 'validation.csv')], ['2', '4'])
            self.assertEqual((out / 'updates.jsonl').read_text(), '{"step":2}\n')
            self.assertFalse((validation / 'step_00000006.json').exists())
            self.assertEqual(read_csv(evaluation / 'episodes.csv')[0]['episode'], '30000')

    def test_resume_keeps_early_frames_and_writes_each_episode_once(self):
        cfg = Config(episode_frames=3)
        run = TrainConfig(algorithm='ppo', control='pure', device='cpu', hidden=8,
            num_envs=2, use_processes=False, ppo_rollout_frames=8, eval_every=100,
            ppo_epochs=1, ppo_critic_epochs=1, ppo_minibatch_size=4,
            ppo_motion_memory=True, ppo_variant='value_decomp_popart',
            ppo_continuation_critic=True, ppo_continuation_mode='learned',
            checkpoint_every=100, render_every_episodes=0)
        with tempfile.TemporaryDirectory() as temporary, \
                patch('priority_mec.training.source_manifest', return_value={'test': 'fixed-source'}), \
                patch('priority_mec.environment.UAVMECEnv.prepare_service', side_effect=AssertionError('Pure PPO must not prepare a solver cost')), \
                patch('priority_mec.solvers.solve', side_effect=AssertionError('Pure PPO must not solve resources')):
            root = Path(temporary)
            train(cfg, root / 'full', 12, run)
            train(cfg, root / 'partial', 4, run)
            checkpoint = load_checkpoint(root / 'partial' / 'last.pt')
            self.assertEqual([s['frames'] for s in checkpoint['episode_accumulators']], [2, 2])
            self.assertEqual(checkpoint['completed_episodes'], 0)
            self.assertEqual(len(checkpoint['pending']['reward']), 4)
            self.assertEqual(checkpoint['action_size'], 146)
            self.assertEqual(checkpoint['observation_size'], UAVMECEnv(cfg, control='pure').observation_size + 6)
            self.assertFalse((root / 'partial' / 'episodes.csv').exists())
            self.assertFalse((root / 'full' / 'train.csv').exists())
            expected = read_csv(root / 'full' / 'episodes.csv')
            # Emulate rows appended after the last atomic checkpoint, before a crash.
            for row in expected:
                append_csv(root / 'partial' / 'episodes.csv', row)
            status = train(cfg, root / 'partial', 12, run, root / 'partial' / 'last.pt')
            actual = read_csv(root / 'partial' / 'episodes.csv')
            self.assertEqual(without_timings(actual), without_timings(expected))
            self.assertEqual([r['episode'] for r in actual], ['0', '1', '2', '3'])
            self.assertEqual([r['frames'] for r in actual], ['3'] * 4)
            self.assertEqual(status['completed_episodes'], 4)
            self.assertEqual(status['session_steps'], 8)
            self.assertAlmostEqual(status['session_fps'], status['session_steps'] / status['session_wall_s'])
            full_cp = load_checkpoint(root / 'full' / 'last.pt')
            resumed_cp = load_checkpoint(root / 'partial' / 'last.pt')
            np.testing.assert_array_equal(full_cp['observations'], resumed_cp['observations'])
            self.assertEqual(full_cp['episode_ids'], resumed_cp['episode_ids'])
            self.assertEqual(full_cp['action_rng'], resumed_cp['action_rng'])
            for model in ('actor', 'critic'):
                for key, value in full_cp['learner'][model].items():
                    np.testing.assert_array_equal(value.cpu().numpy(), resumed_cp['learner'][model][key].cpu().numpy())

    def test_frame_csv_is_explicit_opt_in(self):
        run = TrainConfig(algorithm='sac', device='cpu', hidden=8,
            num_envs=2, use_processes=False, random_steps=100, replay_capacity=32, batch_size=2,
            ppo_rollout_frames=4, eval_every=100, checkpoint_every=100,
            log_frames=True, render_every_episodes=0)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            train(Config(episode_frames=2), out, 4, run)
            self.assertEqual(len(read_csv(out / 'train.csv')), 4)
            self.assertEqual(len(read_csv(out / 'episodes.csv')), 2)

    def test_pure_learned_continuation_configuration(self):
        pure = TrainConfig(algorithm='ppo', control='pure', ppo_motion_memory=True,
            ppo_variant='value_decomp_popart', ppo_continuation_critic=True,
            ppo_continuation_mode='learned', ppo_known_delay_feature=False)
        self.assertEqual(pure.ppo_continuation_mode, 'learned')
        with self.assertRaises(ValueError):
            replace(pure, ppo_continuation_mode='known')
        with self.assertRaises(ValueError):
            replace(pure, ppo_known_delay_feature=True)

    def test_live_rendering_preserves_policy_rng(self):
        run = TrainConfig(algorithm='ppo', device='cpu', hidden=8,
            num_envs=2, use_processes=False, ppo_rollout_frames=4,
            ppo_epochs=1, ppo_critic_epochs=1, ppo_minibatch_size=4,
            eval_every=100, checkpoint_every=100, render_every_episodes=0)
        def random_renderer(_):
            random.random()
            np.random.random(7)
            torch.rand(11)
        with tempfile.TemporaryDirectory() as temporary, \
                patch('priority_mec.training.source_manifest', return_value={'test': 'fixed-source'}):
            root = Path(temporary)
            train(Config(episode_frames=2), root / 'quiet', 8, run)
            with patch('priority_mec.reporting.render_run_curves', side_effect=random_renderer) as renderer:
                status = train(Config(episode_frames=2), root / 'rendered', 8,
                    replace(run, render_every_episodes=1))
                self.assertEqual(renderer.call_count, 3)
            quiet = load_checkpoint(root / 'quiet' / 'last.pt')
            rendered = load_checkpoint(root / 'rendered' / 'last.pt')
            self.assertIsNone(status['plot_error'])
            self.assertGreater(status['plot_wall_s'], 0.)
            self.assertEqual(without_timings(read_csv(root / 'quiet' / 'episodes.csv')),
                without_timings(read_csv(root / 'rendered' / 'episodes.csv')))
            for model in ('actor', 'critic'):
                for key, value in quiet['learner'][model].items():
                    np.testing.assert_array_equal(value.cpu().numpy(), rendered['learner'][model][key].cpu().numpy())
            np.testing.assert_array_equal(quiet['rng']['torch'].numpy(), rendered['rng']['torch'].numpy())


if __name__ == '__main__':
    unittest.main()
