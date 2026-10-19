"""Actual experiment recipes, CLI switches, evaluation and output preservation."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from priority_mec import experiments
from priority_mec.config import Config
from priority_mec.io import write_json


EXPECTED_IDS = {
    'baseline_priority_greedy', 'baseline_greedy_solver',
    'pure_ppo', 'pure_sac', 'pure_ddpg',
    'pure_ppo_ar_adaptive', 'pure_ppo_ar05',
    'hybrid_ppo', 'hybrid_ppo_ar_adaptive', 'hybrid_ppo_ar05',
}


def _args(**overrides):
    values = dict(common=None, environment=None, episodes=None, device='cpu',
                  num_envs=None, synchronous=None, noise_decay=None, lr_decay=None,
                  smoke=False, eval_episodes=None, eval_start=None, out=None, resume=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_catalog_has_all_ten_requested_groups_with_baselines_first():
    catalog = experiments.recipes()
    assert set(catalog) == EXPECTED_IDS
    assert [r['kind'] for r in list(catalog.values())[:2]] == ['baseline', 'baseline']
    assert all(r['id'] == name for name, r in catalog.items())


def test_actual_recipes_share_physical_scenes_training_budget_and_evaluation_protocol():
    resolved = {name: experiments.settings(_args(), recipe)
                for name, recipe in experiments.recipes().items()}
    reference_cfg, _, reference_steps, reference_protocol, _ = resolved['pure_ppo']
    for name, (cfg, run, steps, protocol, _) in resolved.items():
        assert cfg == reference_cfg, name
        assert steps == reference_steps, name
        assert protocol == reference_protocol, name
        assert steps % run.num_envs == 0, name
        assert run.seed == resolved['pure_ppo'][1].seed, name
        if name.startswith(('pure_', 'hybrid_')) and run.algorithm == 'ppo':
            assert steps % run.ppo_rollout_frames == 0, name
    assert reference_protocol['episode_start'] >= 30000


@pytest.mark.parametrize('suffix', ['ar_adaptive', 'ar05'])
def test_pure_and_hybrid_improvements_use_the_appropriate_continuation_information(suffix):
    catalog = experiments.recipes()
    _, pure, _, _, pure_flags = experiments.settings(_args(), catalog['pure_ppo_' + suffix])
    _, hybrid, steps, _, hybrid_flags = experiments.settings(_args(), catalog['hybrid_ppo_' + suffix])
    for run in (pure, hybrid):
        assert run.algorithm == 'ppo'
        assert run.ppo_variant == 'value_decomp_popart'
        assert run.ppo_motion_memory and run.ppo_continuation_critic
        assert run.ppo_ar_adaptive == (suffix == 'ar_adaptive')
        assert run.ppo_ar_rho == (.8 if suffix == 'ar_adaptive' else .5)
    assert pure.control == 'pure' and hybrid.control == 'hybrid'
    assert pure.ppo_continuation_mode == 'learned'
    assert not pure.ppo_known_delay_feature
    assert hybrid.ppo_continuation_mode == 'known'
    assert hybrid.ppo_known_delay_feature
    assert pure.ppo_schedule_steps == 0
    assert pure.ppo_noise_final_scale == pure.ppo_actor_lr_final_scale == 1.
    assert not pure_flags
    assert hybrid_flags == {'noise_decay': True, 'lr_decay': True}
    assert hybrid.ppo_schedule_steps == steps


@pytest.mark.parametrize('noise,lr', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('name', ['hybrid_ppo_ar_adaptive', 'hybrid_ppo_ar05'])
def test_optional_hybrid_schedules_resolve_independently(name, noise, lr):
    recipe = experiments.recipes()[name]
    _, run, steps, _, flags = experiments.settings(_args(noise_decay=noise, lr_decay=lr), recipe)
    assert flags == {'noise_decay': noise, 'lr_decay': lr}
    assert run.ppo_schedule_steps == (steps if noise or lr else 0)
    assert run.ppo_noise_final_scale == (.1 if noise else 1.)
    assert run.ppo_actor_lr_final_scale == (.2 if lr else 1.)


@pytest.mark.parametrize('name', sorted(EXPECTED_IDS - {'hybrid_ppo_ar_adaptive', 'hybrid_ppo_ar05'}))
def test_schedule_switches_cannot_modify_other_experiment_groups(name):
    with pytest.raises(ValueError, match='reserved'):
        experiments.settings(_args(noise_decay=True), experiments.recipes()[name])


def test_smoke_is_an_explicit_small_software_protocol_with_its_own_schedule_horizon():
    recipe = experiments.recipes()['hybrid_ppo_ar_adaptive']
    cfg, run, steps, protocol, _ = experiments.settings(_args(smoke=True), recipe)
    assert cfg.episode_frames == 4 and steps == 16
    assert run.ppo_rollout_frames == 8 and run.num_envs == 2
    assert not run.use_processes
    assert run.ppo_schedule_steps == steps
    assert protocol['episodes'] == 1


def test_cli_parses_boolean_schedule_switches_and_preserves_explicit_output(tmp_path, monkeypatch):
    catalog = experiments.recipes()
    monkeypatch.setattr(experiments, 'ROOT', tmp_path)
    monkeypatch.setattr(experiments, 'recipes', lambda: catalog)
    calls = []

    def completed(args, name, recipe, batch):
        calls.append((args, name, recipe, batch))
        out = Path(args.out)
        write_json(out/'job.done', {})
        return out

    monkeypatch.setattr(experiments, 'run_one', completed)
    destination = tmp_path/'original batch'/'my run'
    experiments.main(['run', '--experiment', 'hybrid_ppo_ar05', '--out', str(destination),
                      '--tag', 'cli', '--device', 'cpu', '--noise-decay', '--no-lr-decay',
                      '--synchronous', '--episodes', '200'])
    args, name, _, batch = calls[0]
    assert name == 'hybrid_ppo_ar05'
    assert args.noise_decay is True and args.lr_decay is False and args.synchronous is True
    assert args.episodes == 200 and Path(args.out) == destination
    assert batch == destination.parent
    manifest = json.loads((batch/'experiment_manifest.json').read_text())
    relative_run = Path(manifest['groups'][0]['run_dir'])
    assert not relative_run.is_absolute()
    assert (batch/relative_run).resolve() == destination.resolve()
    # Results may move to another computer without rewriting their manifest.
    from priority_mec.reporting import _contained, _manifest_groups
    copied = tmp_path/'copied batch'
    shutil.copytree(batch, copied)
    copied_run = _contained(copied.resolve(), _manifest_groups(copied)[0]['run_dir'])
    assert copied_run == (copied/destination.name).resolve()
    assert copied_run != destination.resolve()
    assert (copied_run/'job.done').is_file()


def test_run_refuses_nonempty_output_without_changing_any_existing_file(tmp_path):
    out = tmp_path/'existing'
    out.mkdir()
    sentinel = out/'keep.bin'
    sentinel.write_bytes(b'original training output\x00\xff')
    before = sentinel.read_bytes()
    args = _args(out=str(out))
    with pytest.raises(FileExistsError, match='Output exists'):
        experiments.run_one(args, 'pure_ppo', experiments.recipes()['pure_ppo'], tmp_path)
    assert sentinel.read_bytes() == before
    assert list(out.iterdir()) == [sentinel]


@pytest.mark.parametrize('also_sample', [False, True])
def test_postprocessing_evaluates_both_best_and_last_at_saved_policy_settings(tmp_path, monkeypatch, also_sample):
    from priority_mec import reporting, training
    cfg = Config(episode_frames=4)
    checkpoint_steps = {'best': 8, 'last': 16}
    for name in checkpoint_steps:
        (tmp_path/(name+'.pt')).write_bytes(b'checkpoint stub')
    write_json(tmp_path/'status.json', {'step': 16, 'training_complete': True})
    calls = []
    learner = object()

    def load(path, device):
        name = Path(path).stem
        return learner, cfg, {'step': checkpoint_steps[name],
                              'train_config': {'seed': 9, 'control': 'pure'}}

    def evaluate(environment, policy, control, episodes, episode_start, seed, out,
                 deterministic, policy_seed):
        assert policy is learner and environment == cfg and control == 'pure'
        calls.append((Path(out).name, deterministic, policy_seed))
        return {'summary': dict(episodes=episodes, episode_start=episode_start, seed=seed,
                  environment_config=cfg.to_dict(), deterministic_policy=deterministic,
                  policy_seed=None if deterministic else policy_seed)}

    monkeypatch.setattr(training, 'load_learner', load)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    monkeypatch.setattr(reporting, 'render_run_curves', lambda *args, **kwargs: None)
    monkeypatch.setattr(reporting, 'render_trajectories', lambda *args, **kwargs: None)
    protocol = dict(episodes=2, episode_start=30000, policy_seed=73, also_sample=also_sample)
    experiments._postprocess(tmp_path, protocol, 'cpu')
    expected = [('evaluation_best', True, 73), ('evaluation_last', True, 73)]
    if also_sample:
        expected = [('evaluation_best', True, 73), ('evaluation_best_sample', False, 73),
                    ('evaluation_last', True, 73), ('evaluation_last_sample', False, 73)]
    assert calls == expected
    for kind, step in checkpoint_steps.items():
        result = json.loads((tmp_path/('evaluation_'+kind)/'summary.json').read_text())
        assert result['summary']['selected_step'] == step
    assert (tmp_path/'job.done').exists()
    # An identical finish command reuses validated results without evaluating.
    calls.clear()
    experiments._postprocess(tmp_path, protocol, 'cpu')
    assert calls == []
    if also_sample:
        with pytest.raises(ValueError, match='protocol differs'):
            experiments._postprocess(tmp_path, dict(protocol, policy_seed=74), 'cpu')


def test_source_compatibility_tracks_python_implementation_and_not_tunable_json(tmp_path, monkeypatch):
    from priority_mec import io
    module = tmp_path/'priority_mec'/'io.py'
    module.parent.mkdir()
    module.write_text('# implementation A\n')
    recipe = tmp_path/'configs'/'experiment.json'
    recipe.parent.mkdir()
    recipe.write_text('{"actor_lr": 0.0001}')
    monkeypatch.setattr(io, '__file__', str(module))
    before = io.source_manifest()
    recipe.write_text('{"actor_lr": 0.0002}')
    assert io.source_manifest() == before
    module.write_text('# implementation B\n')
    assert io.source_manifest() != before
    assert set(before['files']) == {'priority_mec/io.py'}


def _resume_fixture(tmp_path, monkeypatch, name='hybrid_ppo_ar_adaptive'):
    from priority_mec import io
    recipe = experiments.recipes()[name]
    cfg, run, steps, protocol, flags = experiments.settings(_args(), recipe)
    origin = tmp_path/'origin'
    origin.mkdir()
    checkpoint = origin/'last.pt'
    checkpoint.write_bytes(b'checkpoint stub')
    write_json(origin/'experiment.json', dict(recipe, options=flags))
    write_json(origin/'effective_protocol.json', protocol)
    write_json(origin/'manifest.json', {'steps_target': steps})
    cp = {'config': cfg.to_dict(), 'train_config': asdict(run)}
    monkeypatch.setattr(io, 'load_checkpoint', lambda path: cp)
    return recipe, checkpoint, run, steps, flags


def test_resume_preserves_original_settings_and_schedule_horizon_when_budget_extends(tmp_path, monkeypatch):
    from priority_mec import training
    recipe, checkpoint, original_run, original_steps, original_flags = _resume_fixture(tmp_path, monkeypatch)
    requested_recipe = copy.deepcopy(recipe)
    requested_recipe['options'] = {'noise_decay': False, 'lr_decay': False}
    calls = []

    def train(cfg, out, steps, run, resume):
        calls.append((cfg, out, steps, run, resume))
        return {'training_complete': False}

    monkeypatch.setattr(training, 'train', train)
    args = _args(resume=str(checkpoint), episodes=2000, out=str(tmp_path/'resumed'))
    out = experiments.run_one(args, recipe['id'], requested_recipe, tmp_path)
    cfg, _, steps, run, resume = calls[0]
    assert steps == 2000 * cfg.episode_frames > original_steps
    assert run == original_run
    assert run.ppo_schedule_steps == original_steps
    assert resume == str(checkpoint)
    recorded = experiments.read_json(out/'experiment.json')
    assert recorded['options'] == original_flags
    assert recorded['train_config'] == asdict(original_run)


def test_resume_does_not_read_current_defaults_after_loading_saved_configuration(tmp_path, monkeypatch):
    from priority_mec import training
    recipe, checkpoint, original_run, original_steps, _ = _resume_fixture(tmp_path, monkeypatch)
    calls = []

    def forbidden_settings(*args, **kwargs):
        raise AssertionError('Exact resume must not parse unrelated current config files')

    def train(cfg, out, steps, run, resume):
        calls.append((steps, run))
        return {'training_complete': False}

    monkeypatch.setattr(experiments, 'settings', forbidden_settings)
    monkeypatch.setattr(training, 'train', train)
    args = _args(resume=str(checkpoint), out=str(tmp_path/'resumed'))
    experiments.run_one(args, recipe['id'], recipe, tmp_path)
    assert calls == [(original_steps, original_run)]


def test_resume_rejects_relabeling_a_checkpoint_as_another_method(tmp_path, monkeypatch):
    _, checkpoint, _, _, _ = _resume_fixture(tmp_path, monkeypatch, name='pure_ppo')
    args = _args(resume=str(checkpoint), out=str(tmp_path/'resumed'))
    with pytest.raises(ValueError, match='experiment ID differs'):
        experiments.run_one(args, 'pure_ppo_ar_adaptive',
                            experiments.recipes()['pure_ppo_ar_adaptive'], tmp_path)
    assert not (tmp_path/'resumed').exists()


@pytest.mark.parametrize('field,value', [
    ('num_envs', 4), ('synchronous', True), ('noise_decay', False), ('lr_decay', False),
    ('eval_episodes', 2), ('eval_start', 40000),
])
def test_resume_rejects_nonbudget_overrides_instead_of_silently_ignoring_them(tmp_path, monkeypatch, field, value):
    recipe, checkpoint, _, _, _ = _resume_fixture(tmp_path, monkeypatch)
    args = _args(resume=str(checkpoint), out=str(tmp_path/'resumed'), **{field: value})
    with pytest.raises(ValueError, match='Resume preserves all settings'):
        experiments.run_one(args, recipe['id'], recipe, tmp_path)
    assert not (tmp_path/'resumed').exists()


def _completed_batch_fixture(tmp_path, monkeypatch):
    from priority_mec import reporting
    original_root = experiments.ROOT
    common = original_root/'configs'/'common.json'
    environment = original_root/'configs'/'environment.json'
    catalog = experiments.recipes()
    args = _args(common=str(common), environment=str(environment))
    cfg, run, steps, protocol, flags = experiments.settings(args, catalog['pure_ppo'])
    out = tmp_path/'results'/'completed'/'pure_ppo'
    write_json(out/'experiment.json', dict(catalog['pure_ppo'], environment=cfg.to_dict(),
        train_config=asdict(run), steps_target=steps, protocol=protocol, options=flags, smoke=False))
    write_json(out/'job.done', {})
    for kind in ('best', 'last'):
        write_json(out/('evaluation_'+kind)/'summary.json', {})
    monkeypatch.setattr(experiments, 'ROOT', tmp_path)
    monkeypatch.setattr(experiments, 'recipes', lambda: catalog)
    monkeypatch.setattr(experiments, 'run_one', lambda *args, **kwargs:
                        pytest.fail('A completed batch must not start training again'))
    monkeypatch.setattr(reporting, 'render_comparison', lambda *args, **kwargs: None)
    return out, ['batch', '--experiments', 'pure_ppo', '--tag', 'completed',
                 '--skip-complete', '--device', 'cpu', '--common', str(common),
                 '--environment', str(environment)]


def test_skip_complete_reuses_only_identical_saved_settings(tmp_path, monkeypatch, capsys):
    out, argv = _completed_batch_fixture(tmp_path, monkeypatch)
    sentinel = out/'job.done'
    before = sentinel.read_bytes()
    experiments.main(argv)
    assert 'Skip completed pure_ppo' in capsys.readouterr().out
    assert sentinel.read_bytes() == before
    with pytest.raises(ValueError, match='Completed run settings differ'):
        experiments.main(argv + ['--eval-start', '40000'])
    assert sentinel.read_bytes() == before


def test_skip_complete_requires_both_checkpoint_evaluations(tmp_path, monkeypatch):
    out, argv = _completed_batch_fixture(tmp_path, monkeypatch)
    (out/'evaluation_last'/'summary.json').unlink()
    with pytest.raises(ValueError, match='Missing evaluation'):
        experiments.main(argv)


def test_one_batch_cannot_mix_smoke_and_research_runs(tmp_path, monkeypatch):
    _, argv = _completed_batch_fixture(tmp_path, monkeypatch)
    experiments.main(argv)
    manifest = tmp_path/'results'/'completed'/'experiment_manifest.json'
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='mix smoke'):
        experiments.main(argv + ['--smoke'])
    assert manifest.read_bytes() == before


def test_paused_batch_stops_before_starting_the_next_group_or_comparing(tmp_path, monkeypatch, capsys):
    from priority_mec import reporting
    catalog = experiments.recipes()
    monkeypatch.setattr(experiments, 'ROOT', tmp_path)
    monkeypatch.setattr(experiments, 'recipes', lambda: catalog)
    calls = []

    def paused(args, name, recipe, batch):
        out = batch/name
        out.mkdir(parents=True)
        write_json(out/'status.json', {'state': 'paused_signal', 'training_complete': False,
                                       'signal': 2})
        calls.append(name)
        return out

    monkeypatch.setattr(experiments, 'run_one', paused)
    monkeypatch.setattr(reporting, 'render_comparison', lambda *args, **kwargs:
                        pytest.fail('A paused batch must not publish incomplete comparisons'))
    experiments.main(['batch', '--experiments', 'pure_ppo', 'pure_sac', '--tag', 'paused'])
    assert calls == ['pure_ppo']
    assert not (tmp_path/'results'/'paused'/'pure_sac').exists()
    assert 'stop' in capsys.readouterr().out.lower()


@pytest.mark.parametrize('field,value', [
    ('eval_episodes', 0), ('eval_episodes', -1), ('eval_episodes', True),
    ('eval_episodes', 1.5), ('eval_start', -1), ('eval_start', True),
    ('eval_start', 2.5),
])
def test_invalid_final_evaluation_protocol_is_rejected_before_training(field, value):
    with pytest.raises(ValueError):
        experiments.settings(_args(**{field: value}), experiments.recipes()['pure_ppo'])
