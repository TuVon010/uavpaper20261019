"""Run an explicit frozen campaign sequentially; no SSH or credentials here."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--campaign', type=Path,
                        default=Path('configs/campaigns/wide_compute1000_w5_cutoff.json'),
                        help='Campaign JSON relative to the frozen runtime; default the current eight-group cutoff campaign')
    parser.add_argument('--validation', type=Path,
                        help='Saved server validation evidence; default project/SERVER_SERVICE_CUTOFF_VALIDATION.json')
    args = parser.parse_args()
    project, runtime = args.project.resolve(), args.runtime.resolve()
    if runtime.parent != project/'frozen' or not (runtime/'run.py').is_file():
        raise ValueError('Expected a complete runtime immediately below project/frozen')
    if Path(args.tag).name != args.tag or args.tag in {'.','..'}:
        raise ValueError('Invalid batch tag')
    sys.path.insert(0, str(runtime))
    from priority_mec.experiments import campaign_settings, recipes
    from priority_mec.io import source_manifest
    validation_path = args.validation or project/'SERVER_SERVICE_CUTOFF_VALIDATION.json'
    validation = json.loads(validation_path.read_text())
    if not validation.get('passed') or validation['source_sha256'] != source_manifest()['sha256']:
        raise ValueError('Validate this source before running the campaign')
    for name, digest in validation.get('runtime_files', {}).items():
        candidate = (runtime/name).resolve()
        if not candidate.is_relative_to(runtime) or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen file differs from validated deployment: '+name)
    batch = project/'results'/args.tag
    batch.mkdir(parents=True, exist_ok=True)
    lock = (batch/'campaign.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (batch/'campaign_state.json').exists():
        raise FileExistsError('Campaign already recorded; inspect it instead of duplicating jobs')
    available = recipes()
    campaign, config_paths = (campaign_settings(args.campaign, root=runtime)
                              if args.campaign else (None, {}))
    current_order = ('baseline_priority_greedy', 'baseline_greedy_solver',
        'pure_ppo', 'hybrid_ppo', 'hybrid_sac', 'hybrid_ddpg',
        'hybrid_ppo_ar_adaptive', 'hybrid_ppo_ar05')
    order = campaign['experiments'] if campaign else current_order
    catalog = {name:available[name] for name in order}
    groups = [dict(id=name, label=r['label'], kind=r['kind'], control=r['control'], run_dir=name)
              for name, r in catalog.items()]
    write(batch/'experiment_manifest.json', dict(groups=groups, tag=args.tag, smoke=False,
        campaign=campaign,
        protocol='Paired scenes; slot-start service; best and last evaluated separately.'))
    state = dict(state='running', started_at_utc=now(), driver_pid=os.getpid(),
        runtime=str(runtime), source_sha256=source_manifest()['sha256'], device=args.device,
        campaign=campaign, group_order=list(catalog), current_group=None,
        groups={name:dict(state='queued') for name in catalog}, comparison={})
    write(batch/'campaign_state.json', state)
    child = None
    stopping = [None]

    def stop(signum, _):
        stopping[0] = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    for name, recipe in catalog.items():
        if stopping[0] is not None:
            break
        out = batch/name
        state['current_group'] = name
        detail = state['groups'][name]
        detail.update(state='running', started_at_utc=now())
        command = [sys.executable, '-u', str(runtime/'run.py'), 'run', '--experiment', name,
                   '--out', str(out), '--tag', args.tag, '--device', args.device]
        for key, path in config_paths.items(): command += ['--'+key, str(path)]
        detail['command'] = command
        write(batch/'campaign_state.json', state)
        print(now(), 'START', name, flush=True)
        # Keep the group directory absent until run.py verifies non-overwrite.
        with (batch/(name+'.log')).open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
            detail['pid'] = child.pid
            write(batch/'campaign_state.json', state)
            code = child.wait()
        required = ['evaluation'] if recipe['kind'] == 'baseline' else ['evaluation_best','evaluation_last']
        complete = code == 0 and (out/'job.done').exists() and all(
            (out/folder/'summary.json').is_file() for folder in required)
        detail.update(state='complete' if complete else 'interrupted' if stopping[0] else 'failed',
                      return_code=code, finished_at_utc=now())
        write(batch/'campaign_state.json', state)
        print(now(), detail['state'].upper(), name, 'exit='+str(code), flush=True)
        child = None
    state['current_group'] = None
    all_complete = all(v['state'] == 'complete' for v in state['groups'].values())
    if all_complete and stopping[0] is None:
        for kind in ('best','last'):
            command = [sys.executable, '-u', str(runtime/'run.py'), 'compare',
                       '--batch', str(batch), '--checkpoint', kind]
            with (batch/('comparison_'+kind+'.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
            state['comparison'][kind] = dict(return_code=code, complete=code == 0)
            child = None
            write(batch/'campaign_state.json', state)
            if stopping[0]:
                break
    success = all_complete and len(state['comparison']) == 2 and all(
        v['complete'] for v in state['comparison'].values())
    state.update(state='complete' if success else 'interrupted' if stopping[0] else 'completed_with_errors',
                 finished_at_utc=now(), signal=stopping[0])
    write(batch/'campaign_state.json', state)
    write(batch/('campaign.done' if success else 'campaign.incomplete'), state)
    print(now(), state['state'].upper(), flush=True)
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
