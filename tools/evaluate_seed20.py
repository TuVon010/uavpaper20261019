"""Evaluate fixed completed checkpoints on paired scenario seeds; never train."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BATCH = 'wide_compute1000_w5_cutoff_20261020_01'
EXPECTED_SOURCE = '8336489dca394491b79654d40f0ce343c4efc2f1615efd8fa8fe44487f46efd9'
METHODS = ('baseline_priority_greedy', 'baseline_greedy_solver', 'pure_ppo',
           'hybrid_ppo', 'hybrid_sac', 'hybrid_ddpg',
           'hybrid_ppo_ar_adaptive', 'hybrid_ppo_ar05')
SEEDS = list(range(1, 21))
EPISODE = 30000


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_runtime(project):
    runtime = project / 'frozen' / BATCH
    if not (runtime / 'priority_mec' / 'training.py').is_file():
        raise FileNotFoundError(runtime)
    sys.path.insert(0, str(runtime))
    from priority_mec.io import source_manifest
    if source_manifest()['sha256'] != EXPECTED_SOURCE:
        raise ValueError('Frozen runtime hash differs; retain checkpoint source checks')
    return runtime


def preflight(project):
    batch = project / 'results' / BATCH
    if not (batch / 'campaign.done').exists():
        raise ValueError('Original campaign has not completed')
    records = {}
    shared_config = None
    for name in METHODS:
        group = batch / name
        status = read_json(group / 'status.json')
        if status['state'] != 'complete' or not status['evaluation_complete'] or not (group / 'job.done').exists():
            raise ValueError('Original group is incomplete: ' + name)
        recipe = read_json(group / 'experiment.json')
        cfg = read_json(group / 'effective_environment.json')
        if shared_config is None:
            shared_config = cfg
        if cfg != shared_config:
            raise ValueError('Methods use different physical configurations')
        trained = recipe['kind'] == 'train'
        original = read_json(group / ('evaluation_best' if trained else 'evaluation') / 'summary.json')
        if original['summary']['source']['sha256'] != EXPECTED_SOURCE:
            raise ValueError('Original evaluation source differs')
        if not trained and original['summary']['baseline'] != recipe['control']:
            raise ValueError('Baseline control differs')
        checkpoint = group / 'best.pt' if trained else None
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        records[name] = dict(kind=recipe['kind'], control=recipe['control'],
                             checkpoint=str(checkpoint) if checkpoint else None,
                             checkpoint_sha256=digest(checkpoint) if checkpoint else None,
                             checkpoint_step=original['summary'].get('selected_step'),
                             training_seed=read_json(group / 'effective_training.json')['seed'])
    if shared_config.get('uav_service_cutoff_enabled') is not True:
        raise ValueError('Retain the completed cutoff protocol')
    return records


def worker(project, output, method, device):
    configure_runtime(project)
    import torch
    torch.set_num_threads(1)
    from priority_mec.config import Config
    from priority_mec.training import evaluate, load_learner
    record = read_json(output / 'request.json')['methods'][method]
    group = project / 'results' / BATCH / method
    cfg = Config(**read_json(group / 'effective_environment.json')).validate()
    trained = record['kind'] == 'train'
    learner = None
    if trained:
        checkpoint = Path(record['checkpoint'])
        if digest(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('Selected model changed since preflight')
        learner, loaded_cfg, checkpoint_data = load_learner(checkpoint, device=device)
        if loaded_cfg.to_dict() != cfg.to_dict() or checkpoint_data['step'] != record['checkpoint_step']:
            raise ValueError('Checkpoint configuration/step differs from selected best')
    original = read_json(group / ('evaluation_best' if trained else 'evaluation') / 'summary.json')
    reference = next(row for row in original['episodes'] if row['episode'] == EPISODE)
    target = output / method
    target.mkdir(exist_ok=True)
    if (target / 'job.done').exists():
        raise FileExistsError('Completed evaluation must not be overwritten')
    started = time.time()
    write_json(target / 'status.json', dict(state='running', method=method, completed=0, total=20))
    for seed in SEEDS:
        destination = target / ('seed_%02d.json' % seed)
        if destination.exists():
            raise FileExistsError('Preserve existing seed output: ' + str(destination))
        result = evaluate(cfg, learner=learner, control=record['control'], episodes=1,
                          episode_start=EPISODE, seed=seed, out=None,
                          baseline=None if trained else record['control'], deterministic=True)
        result['summary'].update(selected_step=record['checkpoint_step'], checkpoint='best.pt' if trained else None)
        result['evaluation_protocol'] = dict(method=method, seed=seed, episode=EPISODE,
            training_seed=record['training_seed'], checkpoint_step=record['checkpoint_step'],
            checkpoint_sha256=record['checkpoint_sha256'], source_sha256=EXPECTED_SOURCE,
            seeds=SEEDS, fixed_checkpoint=True, deterministic_actor_mean=True,
            route_randomness='original keyed categorical uniforms; shared seed/episode/slot across methods',
            use_for_checkpoint_selection=False, scenario_split='additional development evaluation')
        if seed == 1:
            actual = result['episodes'][0]
            keys = ('weighted_delay_mean', 'mean_delay_s', 'episode_return', 'deadline_completion',
                    'total_energy_j', 'total_propulsion_energy_j', 'total_compute_energy_j',
                    'collision_attempt_frames', 'boundary_attempt_frames', 'service_disabled_uav_frames')
            matches = all(math.isclose(float(actual[k]), float(reference[k]), rel_tol=1e-10, abs_tol=1e-10) for k in keys)
            if actual['scenario_hash'] != reference['scenario_hash'] or not matches:
                raise AssertionError('Seed 1 did not reproduce original episode 30000')
            result['evaluation_protocol']['seed1_original_episode_verified'] = True
        write_json(destination, result)
        write_json(target / 'status.json', dict(state='running', method=method, completed=seed,
            total=20, elapsed_s=time.time()-started))
        print(json.dumps(dict(method=method, seed=seed, J=result['summary']['weighted_delay_mean'],
            energy_ok=result['summary']['energy_feasible_episodes'],
            return_=result['summary']['episode_return'])), flush=True)
    if trained and digest(Path(record['checkpoint'])) != record['checkpoint_sha256']:
        raise AssertionError('Checkpoint changed during evaluation')
    done = dict(state='complete', method=method, completed=20, total=20,
                elapsed_s=time.time()-started, source_sha256=EXPECTED_SOURCE,
                checkpoint_sha256=record['checkpoint_sha256'], no_training=True)
    write_json(target / 'status.json', done)
    write_json(target / 'job.done', done)


def driver(project, output, parallel, device):
    reporter = Path(__file__).with_name('report_seed20.py')
    if not reporter.is_file():
        raise FileNotFoundError('Keep report_seed20.py beside this evaluation driver: ' + str(reporter))
    configure_runtime(project)
    if output.exists():
        raise FileExistsError('Use a new directory; do not overwrite')
    if not output.is_relative_to((project / 'results' / BATCH).resolve()):
        raise ValueError('Output must remain inside selected batch')
    records = preflight(project)
    output.mkdir(parents=True)
    request = dict(batch=BATCH, methods=records, seeds=SEEDS, episode=EPISODE,
        created_epoch=time.time(), source_sha256=EXPECTED_SOURCE, physical_parameters_unchanged=True,
        training_seed_unchanged=True, checkpoint_selection='previously selected best; never reselect',
        actor_protocol='same deterministic conditional mean as original evaluation',
        scenario_split='additional development evaluation; final holdout untouched',
        independent_training_runs=1, episodes_per_method=20, parallel_workers=parallel, device=device)
    write_json(output / 'request.json', request)
    states = {method: 'queued' for method in METHODS}
    failures = {}
    started = time.time()

    def run_one(method):
        with (output / (method + '.log')).open('x', encoding='utf-8') as log:
            command = [sys.executable, str(Path(__file__).resolve()), '--project', str(project),
                       '--out', str(output), '--worker', method, '--device', device]
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=project, check=False)
        return method, result.returncode

    write_json(output / 'status.json', dict(state='running', methods=states))
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(run_one, method): method for method in METHODS}
        for future in concurrent.futures.as_completed(futures):
            method, code = future.result()
            states[method] = 'complete' if code == 0 else 'failed'
            if code:
                failures[method] = code
            write_json(output / 'status.json', dict(state='running', methods=states,
                failures=failures, elapsed_s=time.time()-started))
            print(method, states[method], 'exit', code, flush=True)
    if failures:
        raise RuntimeError('Evaluation failed; preserve logs: ' + repr(failures))
    subprocess.run([sys.executable, str(reporter),
                    '--root', str(output)], check=True, cwd=project)
    done = dict(state='complete', methods=states, episodes=160, seeds=SEEDS,
                elapsed_s=time.time()-started, no_training=True, paired_scenarios_verified=True)
    write_json(output / 'status.json', done)
    write_json(output / 'job.done', done)
    print(json.dumps(done), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Project containing results/ and the retained frozen runtime; default this project')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=METHODS)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or a CUDA device such as cuda:0')
    args = parser.parse_args()
    project, output = args.project.resolve(), args.out.resolve()
    try:
        configure_runtime(project)
        from priority_mec.experiments import resolve_device
        device = resolve_device(args.device)
        if args.worker:
            worker(project, output, args.worker, device)
        else:
            if not 1 <= args.parallel <= 4:
                raise ValueError('Concurrency must be between one and four')
            driver(project, output, args.parallel, device)
    except BaseException:
        traceback.print_exc()
        destination = output / args.worker if args.worker else output
        if destination.exists():
            write_json(destination / 'failure.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise


if __name__ == '__main__':
    main()
