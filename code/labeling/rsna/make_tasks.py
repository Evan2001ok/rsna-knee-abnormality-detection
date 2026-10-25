import json, os, random, argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--montages',required=True);p.add_argument('--output',required=True);args=p.parse_args()
if Path(args.output).exists() and list(Path(args.output).glob('t_*.json')):raise ValueError('Use an empty output task folder')
man = json.load(open(str(Path(args.montages)/'manifest.json'), encoding='utf-8'))
man = [e for e in man if e['images']]; random.Random(0).shuffle(man)
IMG = str(Path(args.montages).resolve())
os.makedirs(args.output, exist_ok=True)
for j in range(0, len(man), 10):
    part = [{'uid': e['uid'], 'report': e['report'],
             'images': [{'slot': im['slot'], 'path': f"{IMG}/{im['file']}", 'orientation': im['orientation'],
                         'series_description': im['series_description']} for im in e['images']]}
            for e in man[j:j+10]]
    json.dump(part, open(str(Path(args.output)/f't_{j//10:04d}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
