"""Check the released code and preserved data without executing training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--full-data', action='store_true', help='Also hash all preserved result files')
    parser.add_argument('--load-models', action='store_true', help='Load the six best models on CPU; no rollout')
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = json.loads((project / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    mismatches = []
    for name, record in manifest['files'].items():
        path = (project / name).resolve()
        if not path.is_relative_to(project) or not path.is_file() or sha(path) != record['sha256']:
            mismatches.append(name)
    if mismatches:
        raise ValueError('Released files differ: ' + repr(mismatches))
    release = json.loads((project / 'CURRENT_RELEASE.json').read_text(encoding='utf-8'))
    batch = project / 'results' / release['training_batch']
    data = json.loads((project / release['data_manifest']).read_text(encoding='utf-8'))
    for name, record in data['files'].items():
        path = (batch / name).resolve()
        if not path.is_relative_to(batch) or not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('Preserved result missing/different: ' + name)
        if args.full_data and sha(path) != record['sha256']:
            raise ValueError('Preserved result hash differs: ' + name)
    loaded = []
    if args.load_models:
        sys.path.insert(0, str(project / release['frozen_runtime']))
        from priority_mec.training import load_learner
        for name, record in release['model_checkpoints'].items():
            if record['checkpoint']:
                learner, cfg, checkpoint = load_learner(project / record['checkpoint'], device='cpu')
                if checkpoint['step'] != record['checkpoint_step']:
                    raise ValueError('Wrong selected step: ' + name)
                loaded.append(name)
                del learner, checkpoint
    print(json.dumps(dict(passed=True, release_id=release['release_id'],
        managed_files_verified=len(manifest['files']), result_files_verified=len(data['files']),
        full_result_hashes=args.full_data, best_models_loaded=loaded), ensure_ascii=False))


if __name__ == '__main__':
    main()
