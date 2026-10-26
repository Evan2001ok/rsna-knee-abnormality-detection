"""Insert placeholder rows for the 58 official Gold studies into the seven public RSNA label snapshots.

The official expert labels are competition data and are NOT published here. Each Gold study gets one
placeholder row: every target = PLACEHOLDER_VALUE, and in the three *_labels_used.csv snapshots
label_source = PLACEHOLDER_SOURCE. Training code overrides these rows with the user's own official
train.csv (load_vlm_labels: blended.loc[gold.index] = gold), so training targets are unchanged.

Existing rows are kept byte-for-byte; placeholder rows are merged in StudyInstanceUID order (the original
4,407-row snapshots were sorted the same way). labels/rsna/manifest.json is updated (rows, bytes, sha256,
placeholder note). Afterwards run tools/refresh_bundle.py to refresh the root manifest and notebook.

    python tools/add_gold_placeholders.py --train-csv /path/to/official/train.csv
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'labels/rsna'
PLACEHOLDER_VALUE = '0.5'
PLACEHOLDER_SOURCE = 'official_gold_placeholder'
N_GOLD = 58
N_STUDIES = 4407


def gold_ids(train_csv, targets):
    """UIDs whose 12 official targets are all present (= the official expert/Gold rows)."""
    with Path(train_csv).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = [t for t in ['StudyInstanceUID', *targets] if t not in reader.fieldnames]
        if missing:
            raise SystemExit(f'{train_csv}: missing columns {missing}')
        ids = [r['StudyInstanceUID'] for r in reader if all(r[t].strip() != '' for t in targets)]
    if len(ids) != len(set(ids)) or len(ids) != N_GOLD:
        raise SystemExit(f'Expected {N_GOLD} unique Gold studies in {train_csv}, found {len(ids)}')
    return set(ids)


def add_placeholders(path, gold, targets):
    raw = path.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    lines = raw.decode('utf-8').split(newline)
    if lines[-1] == '':
        lines.pop()
    header, body = lines[0], lines[1:]
    columns = next(csv.reader([header]))
    if columns[0] != 'StudyInstanceUID' or columns[1:13] != targets or columns[13:] not in ([], ['label_source']):
        raise SystemExit(f'{path.name}: unexpected header {columns}')
    has_source = columns[13:] == ['label_source']
    rows = {}
    for line in body:
        fields = next(csv.reader([line]))
        rows[fields[0]] = (fields, line)
    present = gold & set(rows)
    for uid in present:  # Refuse to keep anything except an exact placeholder for a Gold study.
        fields = rows[uid][0]
        if fields[1:13] != [PLACEHOLDER_VALUE] * 12 or (has_source and fields[13] != PLACEHOLDER_SOURCE):
            raise SystemExit(f'{path.name}: Gold study {uid} has a non-placeholder row; remove it first')
    added = 0
    for uid in sorted(gold - present):
        fields = [uid, *[PLACEHOLDER_VALUE] * 12, *([PLACEHOLDER_SOURCE] if has_source else [])]
        rows[uid] = (fields, ','.join(fields))
        added += 1
    if len(rows) != N_STUDIES:
        raise SystemExit(f'{path.name}: {len(rows)} studies after placeholders, expected {N_STUDIES}')
    for uid, (fields, _) in rows.items():
        if uid not in gold:
            assert all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in fields[1:13]), (path.name, uid)
    text = newline.join([header, *(rows[uid][1] for uid in sorted(rows))]) + newline
    path.write_bytes(text.encode('utf-8'))
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-csv', required=True, help='Your own official competition train.csv (read locally only)')
    a = ap.parse_args()
    manifest_path = FOLDER / 'manifest.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode('utf-8'))
    targets = manifest['targets']
    gold = gold_ids(a.train_csv, targets)
    report = {}
    for item in manifest['files']:
        path = FOLDER / item['file']
        report[item['file']] = add_placeholders(path, gold, targets)
        item.update(rows=N_STUDIES, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest['study_count'] = N_STUDIES
    manifest['training_gold_override_count'] = N_GOLD
    manifest['gold_placeholder'] = dict(
        value=float(PLACEHOLDER_VALUE), label_source=PLACEHOLDER_SOURCE, rows=N_GOLD,
        note=('The 58 official expert/Gold labels are competition data and are not published. Their rows hold '
              'placeholders only; training overrides them from your own official train.csv.'))
    text = json.dumps(manifest, ensure_ascii=False, indent=2)  # keep the file's original line endings
    if b'\r\n' in manifest_raw:
        text = text.replace('\n', '\r\n')
    if manifest_raw.endswith(b'\n'):
        text += '\r\n' if b'\r\n' in manifest_raw else '\n'
    manifest_path.write_bytes(text.encode('utf-8'))
    print(json.dumps(dict(gold_placeholders_added=report, studies=N_STUDIES,
                          next_step='python tools/refresh_bundle.py'), indent=2))


if __name__ == '__main__':
    main()
