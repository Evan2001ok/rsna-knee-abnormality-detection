"""Create empty input/output directories and a local path config; never overwrite it."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', choices=('local', 'kaggle'), default='local')
    args = p.parse_args()
    config = ROOT / 'config/paths.json'
    if config.exists():
        raise SystemExit('config/paths.json already exists; edit it explicitly instead of overwriting it.')
    template = json.loads((ROOT / 'config/paths.example.json').read_text(encoding='utf-8'))
    # Both layouts use the included public RSNA source labels. Kaggle retains
    # original mounts for MRI, OAI, initialization and task-trained weights.
    value = template if args.layout == 'local' else {'prefix_map': {
        old: new for old, new in template['prefix_map'].items()
        if new.startswith('${PACKAGE}/labels/rsna/')
    }}
    for folder in ['data/rsna', 'data/oai', 'labels/rsna', 'labels/oai', 'work', 'code/oai/scores']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({'layout': args.layout, 'config': str(config), 'data_downloaded': False,
                      'next': 'Edit config/paths.json, supply external assets, then run --plan and stage checks.'}, indent=2))


if __name__ == '__main__':
    main()
