"""Local experiment launcher. JSON recipes are the only experiment defaults."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def recipes():
    entries = {p.stem: read_json(p) for p in sorted((ROOT/'configs'/'experiments').glob('*.json'))}
    order = ('baseline_priority_greedy','baseline_greedy_solver','pure_ppo','pure_sac','pure_ddpg',
             'pure_ppo_ar_adaptive','pure_ppo_ar05','hybrid_ppo','hybrid_sac','hybrid_ddpg',
             'hybrid_ppo_ar_adaptive','hybrid_ppo_ar05')
    ordered = [name for name in order if name in entries]
    return {name: entries[name] for name in ordered+sorted(set(entries)-set(ordered))}


def campaign_settings(path, root=None):
    """Resolve a frozen campaign's explicit groups and config files.

    Campaign JSON names files relative to the project, so the same selection
    remains usable after copying a project to another computer or frozen path.
    """
    root = Path(root or ROOT).resolve()
    path = Path(path)
    if not path.is_absolute(): path = root/path
    campaign = read_json(path)
    required = {'id', 'environment', 'common', 'experiments'}
    if not required.issubset(campaign) or set(campaign)-required-{'description'}:
        raise ValueError('Invalid campaign fields')
    names = campaign['experiments']
    available = recipes()
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or n not in available for n in names)
            or len(names) != len(set(names))):
        raise ValueError('Campaign requires unique known experiment IDs')
    resolved = {}
    for key in ('environment', 'common'):
        item = Path(campaign[key])
        target = (root/item).resolve()
        if item.is_absolute() or not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Campaign config must be a saved file inside this project: '+key)
        resolved[key] = target
    return campaign, resolved


def validate_protocol(protocol):
    for key, minimum in (('episodes', 1), ('episode_start', 0), ('policy_seed', 0)):
        value = protocol.get(key, 1 if key == 'policy_seed' else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('Evaluation '+key+' must be an integer >= '+str(minimum))
    if not isinstance(protocol.get('also_sample', False), bool):
        raise ValueError('also_sample must be Boolean')
    return protocol


def resolve_device(name):
    import torch
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install compatible PyTorch or select --device cpu.')
    return name


def settings(args, recipe):
    from .config import Config
    from .training import TrainConfig
    common = read_json(args.common or ROOT/'configs'/'common.json')
    cfg = Config.load(args.environment or ROOT/'configs'/'environment.json')
    options = dict(common['training']); options.update(recipe.get('training', {}))
    protocol = dict(common['evaluation'])
    episodes = args.episodes if args.episodes is not None else common['episodes']
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    configured_device = options.pop('device', 'auto')
    options['device'] = resolve_device(args.device or configured_device)
    if args.num_envs is not None: options['num_envs'] = args.num_envs
    if args.synchronous is not None: options['use_processes'] = not args.synchronous
    flags = dict(recipe.get('options', {}))
    for key in ('noise_decay', 'lr_decay'):
        if getattr(args, key, None) is not None:
            if not recipe.get('allow_schedules', False):
                raise ValueError('Noise/LR switches are reserved for improved hybrid PPO recipes.')
            flags[key] = getattr(args, key)
    steps = episodes*cfg.episode_frames
    if recipe.get('allow_schedules', False):
        options.update(ppo_schedule_steps=steps if any(flags.values()) else 0,
            ppo_noise_final_scale=.1 if flags.get('noise_decay', False) else 1.,
            ppo_actor_lr_final_scale=.2 if flags.get('lr_decay', False) else 1.)
    if args.smoke:
        # Explicit software test, never reported as a research experiment.
        cfg = replace(cfg, episode_frames=4, greedy_direction_count=8,
                      greedy_speed_levels=3, greedy_refine_rounds=0)
        steps = 16
        options.update(num_envs=2, use_processes=False, hidden=16, batch_size=8,
            replay_capacity=64, random_steps=0, critic_warmup_steps=0,
            ppo_rollout_frames=8, ppo_epochs=1, ppo_critic_epochs=1,
            ppo_minibatch_size=8, eval_every=8, checkpoint_every=8,
            eval_episodes=1, render_every_episodes=0)
        if recipe.get('allow_schedules', False):
            options['ppo_schedule_steps'] = steps if any(flags.values()) else 0
        protocol.update(episodes=1, episode_start=30000)
    if args.eval_episodes is not None: protocol['episodes'] = args.eval_episodes
    if args.eval_start is not None: protocol['episode_start'] = args.eval_start
    validate_protocol(protocol)
    cfg.validate()
    return cfg, TrainConfig(**options), steps, protocol, flags


def _postprocess(out, protocol, device):
    from .training import evaluate, load_learner
    from .io import write_json
    for kind in ('best', 'last'):
        checkpoint = out/(kind+'.pt')
        if not checkpoint.exists(): raise FileNotFoundError(str(checkpoint))
        learner, cfg, cp = load_learner(checkpoint, device)
        for deterministic in ([True, False] if protocol.get('also_sample', False) else [True]):
            folder = out/('evaluation_'+kind+('' if deterministic else '_sample'))
            if (folder/'summary.json').exists():
                s = read_json(folder/'summary.json')['summary']
                expected = dict(selected_step=cp['step'], episodes=protocol['episodes'],
                    episode_start=protocol['episode_start'], seed=cp['train_config']['seed'],
                    environment_config=cfg.to_dict(), deterministic_policy=deterministic,
                    policy_seed=None if deterministic else protocol.get('policy_seed', 1))
                if any(s.get(k) != v for k, v in expected.items()):
                    raise ValueError('Existing evaluation checkpoint/protocol differs: '+str(folder))
            else:
                result = evaluate(cfg, learner, cp['train_config']['control'],
                    protocol['episodes'], protocol['episode_start'], cp['train_config']['seed'], folder,
                    deterministic=deterministic, policy_seed=protocol.get('policy_seed', 1))
                result['summary'].update(selected_step=cp['step'], checkpoint=checkpoint.name)
                write_json(folder/'summary.json', result)
    from .reporting import render_run_curves, render_trajectories
    render_run_curves(out)
    for kind in ('best', 'last'): render_trajectories(out, checkpoint_kind=kind)
    status = read_json(out/'status.json')
    status.update(state='complete', training_complete=True, evaluation_complete=True)
    write_json(out/'status.json', status)
    write_json(out/'job.done', {'evaluation_best':'evaluation_best/summary.json',
        'evaluation_last':'evaluation_last/summary.json', 'training_steps':status['step']})


def run_one(args, name, recipe, batch):
    import torch
    torch.set_num_threads(1)
    from .io import write_json, load_checkpoint
    from .training import train, evaluate, TrainConfig
    from .config import Config
    out = Path(args.out).resolve() if args.out else batch/name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output exists; choose a new --tag/--out: '+str(out))
    if args.resume:
        if recipe['kind'] != 'train' or args.smoke:
            raise ValueError('Resume requires a trained recipe and cannot combine with --smoke.')
        cp = load_checkpoint(args.resume); origin = Path(args.resume).resolve().parent
        original = read_json(origin/'experiment.json')
        if original['id'] != name:
            raise ValueError('Resume experiment ID differs; keep the original method label.')
        for field in ('environment','common','num_envs','synchronous','noise_decay','lr_decay','eval_episodes','eval_start'):
            if getattr(args, field, None) is not None:
                raise ValueError('Resume preserves all settings; remove --'+field.replace('_','-'))
        cfg = Config(**cp['config']).validate(); run = TrainConfig(**cp['train_config'])
        flags = original.get('options', {})
        if run.algorithm != recipe['training']['algorithm'] or run.control != recipe['control']:
            raise ValueError('Resume recipe differs from checkpoint.')
        if args.device and resolve_device(args.device) != run.device:
            raise ValueError('Exact resume retains the original device; evaluate supports portable inference.')
        protocol = read_json(origin/'effective_protocol.json')
        previous = read_json(origin/'manifest.json')['steps_target']
        steps = args.episodes*cfg.episode_frames if args.episodes is not None else previous
        # Preserve the original schedule horizon, optimizer, replay and all RNG states.
    else:
        cfg, run, steps, protocol, flags = settings(args, recipe)
    validate_protocol(protocol)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'experiment.json', dict(recipe, id=name, environment=cfg.to_dict(),
        train_config=asdict(run), steps_target=steps, protocol=protocol, options=flags, smoke=bool(args.smoke)))
    write_json(out/'effective_environment.json', cfg.to_dict())
    write_json(out/'effective_training.json', asdict(run))
    write_json(out/'effective_protocol.json', protocol)
    print(f'[{name}] device={run.device}, control={recipe["control"]}, steps={steps}, output={out}', flush=True)
    try:
        if recipe['kind'] == 'baseline':
            evaluate(cfg, control=recipe['control'], baseline=recipe['control'],
                episodes=protocol['episodes'], episode_start=protocol['episode_start'],
                seed=run.seed, out=out/'evaluation')
            from .reporting import render_trajectories
            render_trajectories(out, checkpoint_kind='best')
            write_json(out/'status.json', dict(state='complete', training_complete=True,
                evaluation_complete=True, baseline=True))
            write_json(out/'job.done', {'evaluation':'evaluation/summary.json'})
        else:
            status = train(cfg, out, steps, run, args.resume)
            if not status['training_complete']: return out
            _postprocess(out, protocol, run.device)
    except BaseException as error:
        write_json(out/'job.failed', {'error':type(error).__name__+': '+str(error)})
        raise
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='UAV-MEC final experiments; tune configs/*.json.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    for command in ('run', 'batch'):
        p = sub.add_parser(command)
        if command == 'run':
            p.add_argument('--experiment', required=True); p.add_argument('--out'); p.add_argument('--resume')
        else:
            p.add_argument('--experiments', nargs='+', default=['all']); p.add_argument('--skip-complete', action='store_true')
        p.add_argument('--tag'); p.add_argument('--environment'); p.add_argument('--common')
        p.add_argument('--episodes', type=int); p.add_argument('--device'); p.add_argument('--num-envs', type=int)
        p.add_argument('--synchronous', action=argparse.BooleanOptionalAction, default=None)
        p.add_argument('--noise-decay', action=argparse.BooleanOptionalAction, default=None)
        p.add_argument('--lr-decay', action=argparse.BooleanOptionalAction, default=None)
        p.add_argument('--eval-episodes', type=int); p.add_argument('--eval-start', type=int)
        p.add_argument('--smoke', action='store_true', help='16-step software check; not research results')
    p = sub.add_parser('finish'); p.add_argument('--run', required=True); p.add_argument('--device', default='auto')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=10); p.add_argument('--episode-start', type=int, default=30000)
    p.add_argument('--device', default='auto'); p.add_argument('--sample', action='store_true')
    for command in ('plot', 'trajectories'):
        p = sub.add_parser(command); p.add_argument('--run', required=True)
        p.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    p = sub.add_parser('compare')
    p.add_argument('--batch', required=True); p.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    p = sub.add_parser('dashboard'); p.add_argument('--run', required=True); p.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == 'list':
        for name, recipe in recipes().items(): print(f'{name:29s} {recipe["label"]}')
        return
    if args.command in {'plot', 'trajectories', 'compare', 'dashboard'}:
        from . import reporting  # No torch import for plots/dashboard.
        if args.command == 'plot': result = reporting.render_run_curves(args.run, checkpoint_kind=args.checkpoint)
        elif args.command == 'trajectories': result = reporting.render_trajectories(args.run, checkpoint_kind=args.checkpoint)
        elif args.command == 'compare': result = reporting.render_comparison(args.batch, checkpoint_kind=args.checkpoint)
        else: reporting.serve(args.run, port=args.port); return
        print(result); return
    if args.command == 'finish':
        out = Path(args.run).resolve()
        if not read_json(out/'status.json').get('training_complete'): raise ValueError('Resume unfinished training first.')
        _postprocess(out, read_json(out/'effective_protocol.json'), resolve_device(args.device)); return
    if args.command == 'evaluate':
        from .training import load_learner, evaluate
        validate_protocol(dict(episodes=args.episodes, episode_start=args.episode_start))
        learner, cfg, cp = load_learner(args.checkpoint, resolve_device(args.device))
        result = evaluate(cfg, learner, cp['train_config']['control'], args.episodes,
            args.episode_start, cp['train_config']['seed'], args.out, deterministic=not args.sample, policy_seed=1)
        print(json.dumps(result['summary'], indent=2)); return
    available = recipes()
    names = ([args.experiment] if args.command == 'run' else list(available)
             if args.experiments == ['all'] else args.experiments)
    if any(name not in available for name in names): parser.error('Unknown experiment; run python run.py list')
    if len(names) != len(set(names)): parser.error('Duplicate experiment IDs')
    if args.command == 'batch': args.out = None; args.resume = None
    tag = args.tag or (('smoke_' if args.smoke else '')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if Path(tag).name != tag or tag in {'.', '..'}: parser.error('--tag must be a folder name')
    batch = (Path(args.out).resolve().parent if args.command == 'run' and args.out
             else ROOT/'results'/tag)
    batch.mkdir(parents=True, exist_ok=True)
    from .io import write_json
    manifest_path = batch/'experiment_manifest.json'
    saved_batch = read_json(manifest_path) if manifest_path.exists() else None
    if saved_batch is not None and saved_batch.get('smoke', False) != bool(args.smoke):
        raise ValueError('Do not mix smoke checks with formal runs in one batch.')
    groups = saved_batch.get('groups', []) if saved_batch else []
    for name in names:
        destination = Path(args.out).resolve() if args.out else batch/name
        entry = dict(id=name, label=available[name]['label'], kind=available[name]['kind'],
                     control=available[name]['control'], run_dir=destination.relative_to(batch).as_posix())
        existing = [g for g in groups if g['id'] == name]
        if existing and existing[0] != entry: raise ValueError('Manifest has a different run for '+name)
        if not existing: groups.append(entry)
    write_json(manifest_path, {'groups':groups, 'tag':tag, 'smoke':bool(args.smoke),
        'protocol':'Paired scenes; slot-start service; best and last evaluated separately.'})
    for name in names:
        out = batch/name
        if args.command == 'batch' and args.skip_complete and (out/'job.done').exists():
            required = ['evaluation'] if available[name]['kind'] == 'baseline' else ['evaluation_best', 'evaluation_last']
            cfg, run, steps, protocol, flags = settings(args, available[name])
            saved = read_json(out/'experiment.json')
            expected = dict(environment=cfg.to_dict(), train_config=asdict(run), steps_target=steps,
                            protocol=protocol, options=flags, smoke=bool(args.smoke))
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError('Completed run settings differ; use a new tag: '+str(out))
            if protocol.get('also_sample') and available[name]['kind'] != 'baseline':
                required += ['evaluation_best_sample', 'evaluation_last_sample']
            if not all((out/p/'summary.json').exists() for p in required): raise ValueError('Missing evaluation: '+str(out))
            print('Skip completed '+name, flush=True); continue
        result = run_one(args, name, available[name], batch)
        if not (result/'job.done').exists():
            print('Training stopped before completion. Batch stopped; resume the saved last.pt explicitly.', flush=True)
            return
    if args.command == 'batch':
        from .reporting import render_comparison
        for kind in ('best', 'last'): render_comparison(batch, checkpoint_kind=kind)
    print('Results: '+str(batch), flush=True)


if __name__ == '__main__': main()
