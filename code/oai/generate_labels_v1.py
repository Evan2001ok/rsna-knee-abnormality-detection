"""Generate the approved seven-target OAI weak supervision, aligned to packed MRI."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'python_deps'))
import numpy as np
from audit_scores import read_nda

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RULES = {
    'Medial Meniscus': (['m_mmah_morph', 'm_mmbody_morph', 'm_mmph_morph'], 'meniscus', 1.0),
    'Lateral Meniscus': (['m_lmah_morph', 'm_lmbody_morph', 'm_lmph_morph'], 'meniscus', 1.0),
    'Medial OA': (['m_cm_cmf', 'm_cm_pmf', 'm_cm_amt', 'm_cm_cmt', 'm_cm_pmt'], 'oa', 0.5),
    'Lateral OA': (['m_cm_clf', 'm_cm_plf', 'm_cm_alt', 'm_cm_clt', 'm_cm_plt'], 'oa', 0.5),
    'PF OA': (['m_cm_alf', 'm_cm_amf', 'm_cm_lp', 'm_cm_mp'], 'oa', 0.5),
    'Effusion': (['m_eff_wk'], 'effusion', 1.0),
    'Synovitis': (['m_syic'], 'hoffa', 0.5),
}
OA_ALLOWED = {Decimal(v) for v in ['0', '1', '1.1', '2', '2.1', '2.2', '3', '3.1', '3.2', '3.3']}
OA_POSITIVE = {Decimal(v) for v in ['2.2', '3.2', '3.3']}

def vote(row, fields, mode):
    vals = []
    for field in fields:
        raw = row.get(field, '').strip()
        try:
            value = Decimal(raw) if raw else None
        except InvalidOperation:
            return None, 'invalid_code'
        allowed = OA_ALLOWED if mode == 'oa' else set(map(Decimal, range(9 if mode == 'meniscus' else 4)))
        if value is not None and value not in allowed:
            return None, 'invalid_code'
        vals.append(value)
    if mode == 'oa':
        positive = any(v in OA_POSITIVE for v in vals)
        negative = all(v == 0 for v in vals)
    elif mode == 'meniscus':
        positive = any(v is not None and v >= 2 for v in vals)
        negative = all(v in (0, 1) for v in vals)
    elif mode == 'effusion':
        positive = vals[0] in (2, 3)
        negative = vals[0] in (0, 1)
    else:
        positive = vals[0] in (2, 3)
        negative = vals[0] == 0
    if positive:
        return 1, 'positive'
    if negative:
        return 0, 'negative'
    return None, 'missing' if any(v is None for v in vals) else 'ambiguous_grade'

def surgery_ambiguity(rows):
    flags = []
    for r in rows:
        for field in ['m_comments', 'm_technical']:
            text = r.get(field, '')
            if re.search(r'meniscect|menisc.{0,30}(repair|resect)|(?:repair|resect).{0,30}menisc', text, re.I):
                # A later-visit operation is not evidence of baseline postoperative anatomy.
                future = bool(re.search(r'between\s+V[1-9]\d*\s+and\s+V[1-9]\d*', text, re.I))
                flags.append({'field': field, 'text': text, 'readprj': r.get('readprj'),
                              'mask_baseline_meniscus': not future})
    return flags

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default=Path('D:/OAI/kaggle_upload_224'))
    parser.add_argument('--output', type=Path, default=ROOT / 'labels_v1')
    args = parser.parse_args()
    index_path = args.images / 'knee_index.json'
    index = sorted(json.loads(index_path.read_text(encoding='utf-8')), key=lambda r: (r['part'], r['row']))
    ids = np.concatenate([np.load(args.images / f'part{p}_knee_ids.npy', allow_pickle=False) for p in range(2)])
    assert len(index) == len(ids) == 2399
    assert list(ids) == [r['knee_id'] for r in index]
    assert len(set(ids)) == len(ids)
    for p in range(2):
        entries = [r for r in index if r['part'] == p]
        assert [r['row'] for r in entries] == list(range(len(entries)))
        cache = np.load(args.images / f'part{p}_cache.npy', mmap_mode='r', allow_pickle=False)
        assert cache.shape[0] == len(entries)
    keys = {(r['src_subject_id'], r['side']) for r in index}
    source = ROOT / 'scores/oai_kmrisemiquant01.txt'
    groups = defaultdict(list)
    for source_row, row in enumerate(read_nda(source), start=3):
        key = (row['src_subject_id'], {'1': 'RIGHT', '2': 'LEFT'}.get(row['side']))
        if row['visit'] == 'V00' and key in keys:
            groups[key].append((source_row, row))
    assert set(groups) == keys
    n = len(index)
    labels = np.zeros((n, 12), dtype=np.float32)
    mask = np.zeros((n, 12), dtype=np.uint8)
    weight = np.zeros((n, 12), dtype=np.float32)
    status = np.full((n, 12), 'not_selected', dtype='<U24')
    provenance = []
    for i, entry in enumerate(index):
        rows = groups[(entry['src_subject_id'], entry['side'])]
        surgical = surgery_ambiguity([r for _, r in rows])
        per_target = {}
        for name, (fields, mode, w) in RULES.items():
            j = TARGETS.index(name)
            readings = []
            votes = set()
            reasons = []
            for line, row in rows:
                v, reason = vote(row, fields, mode)
                if v is not None:
                    votes.add(v)
                reasons.append(reason)
                readings.append({'source_line': line, 'readprj': row.get('readprj'), 'version': row.get('version'),
                    'raw_scores': {f: row.get(f, '') for f in fields}, 'vote': v, 'reason': reason})
            if mode == 'meniscus' and any(f['mask_baseline_meniscus'] for f in surgical):
                s = 'surgery_ambiguity'
            elif 'invalid_code' in reasons:
                s = 'invalid_code'
            elif len(votes) > 1:
                s = 'conflict'
            elif votes:
                v = next(iter(votes))
                s = 'positive' if v else 'negative'
                labels[i, j], mask[i, j], weight[i, j] = v, 1, w
            else:
                s = 'ambiguous_grade' if 'ambiguous_grade' in reasons else 'missing'
            status[i, j] = s
            per_target[name] = {'status': s, 'readings': readings}
        provenance.append({**entry, 'targets': per_target, 'surgery_review': surgical})
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'labels.npy', labels)
    np.save(args.output / 'label_mask.npy', mask)
    np.save(args.output / 'label_weight.npy', weight)
    np.save(args.output / 'label_status.npy', status)
    np.save(args.output / 'knee_ids.npy', ids)
    np.save(args.output / 'subject_ids.npy', np.array([r['src_subject_id'] for r in index]))
    for p in range(2):
        selected = np.array([r['part'] == p for r in index])
        for suffix, arr in [('labels', labels), ('label_mask', mask), ('label_weight', weight), ('knee_ids', ids)]:
            np.save(args.output / f'part{p}_{suffix}.npy', arr[selected])
    with (args.output / 'labels.csv').open('w', encoding='utf-8-sig', newline='') as f:
        columns = ['knee_id', 'src_subject_id', 'side', 'part', 'row'] + TARGETS
        columns += [f'{t}__mask' for t in TARGETS] + [f'{t}__weight' for t in TARGETS]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, entry in enumerate(index):
            record = {k: entry[k] for k in columns[:5]}
            record.update({t: int(labels[i, j]) if mask[i, j] else '' for j, t in enumerate(TARGETS)})
            record.update({f'{t}__mask': int(mask[i, j]) for j, t in enumerate(TARGETS)})
            record.update({f'{t}__weight': float(weight[i, j]) for j, t in enumerate(TARGETS)})
            writer.writerow(record)
    with (args.output / 'provenance.jsonl').open('w', encoding='utf-8') as f:
        for record in provenance:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    # Reload saved arrays and verify actual artifact alignment and masked loss safety.
    y = np.load(args.output / 'labels.npy', allow_pickle=False)
    m = np.load(args.output / 'label_mask.npy', allow_pickle=False)
    w = np.load(args.output / 'label_weight.npy', allow_pickle=False)
    assert y.shape == m.shape == w.shape == (2399, 12)
    assert np.isfinite(y).all() and set(np.unique(y)) <= {0, 1}
    assert set(np.unique(m)) <= {0, 1} and np.all(w[m == 0] == 0)
    assert np.all(y[m == 0] == 0) and np.all(w[m == 1] > 0)
    for p in range(2):
        part_ids = np.load(args.output / f'part{p}_knee_ids.npy', allow_pickle=False)
        assert np.array_equal(part_ids, np.load(args.images / f'part{p}_knee_ids.npy', allow_pickle=False))
    counts = {t: dict(Counter(status[:, j])) for j, t in enumerate(TARGETS)}
    for j, t in enumerate(TARGETS):
        assert sum(counts[t].values()) == n
        assert int(m[:, j].sum()) == counts[t].get('positive', 0) + counts[t].get('negative', 0)
    report = {'format_version': 'oai-seven-labels-conservative-v1', 'knees': n,
        'subjects': len({r['src_subject_id'] for r in index}), 'source_rows': sum(map(len, groups.values())),
        'target_order': TARGETS, 'counts': counts, 'valid_target_cells': int(m.sum()),
        'knees_with_any_label': int((m.sum(axis=1) > 0).sum()),
        'rules': {name: {'fields': fields, 'mode': mode, 'weight': w} for name, (fields, mode, w) in RULES.items()},
        'aggregation': 'Ignore missing/ambiguous readings when a definite vote exists; opposite definite votes conflict-mask; invalid codes mask. No majority vote.',
        'masked_value': 'NPY labels use finite zero placeholders with mask=0 and weight=0; CSV masked values are blank. Never train on placeholders.',
        'weak_supervision': ['Medial OA', 'Lateral OA', 'PF OA', 'Synovitis'],
        'source_sha256': sha(source), 'image_index_sha256': sha(index_path),
        'files_sha256': {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.suffix in ('.npy', '.csv', '.jsonl')},
        'verification': {'passed': True, 'unique_ids': True, 'exact_image_row_alignment': True,
                         'saved_arrays_reloaded': True, 'binary_finite_values': True, 'zero_masked_weights': True},
        'surgical_comment_flags': sum(len(r['surgery_review']) for r in provenance)}
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'README.txt').write_text(
        'OAI seven-label conservative weak supervision v1\n'
        'Target order is in summary.json. Rows match packed part0 then part1 MRI exactly.\n'
        'OA: any 2.2/3.2/3.3 positive; ALL required regions 0 negative; rest masked.\n'
        'Meniscus: any 2-8 positive; all three regions 0/1 negative; rest masked.\n'
        'Effusion: 2/3 positive, 0/1 negative. Hoffa: 2/3 positive, 0 negative, 1 masked.\n'
        'Other five targets always masked. OA and Hoffa are proxies, not equivalent clinical reference labels.\n'
        'Use label_weight * label_mask in elementwise BCE loss. Normalize by valid mask count;\n'
        'normalizing a single-target loss by sum of weights would cancel its fixed 0.5 weight.\n'
        'Keep subject_ids together for any subject-level split (both knees must stay together).\n'
        'provenance.jsonl stores source lines, reading projects, raw scores, and vote reasons.\n'
        'Meniscal surgical history screening is limited to available comments; absence is not proof of no surgery.\n'
        'Image series mask is distinct from label_mask; DESS/MPR slots are not RSNA T1 slots.\n'
        'No training code or Kaggle dataset has been modified.\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'counts': counts, 'valid_target_cells': int(m.sum()),
                      'knees_with_any_label': report['knees_with_any_label'], 'verification': report['verification']}, indent=2))

if __name__ == '__main__':
    main()
