"""Match downloaded visit-specific MRI indices against the existing four-label audit.

Outputs are candidates, not final labels: surgery review and duplicate selection
remain necessary before training or bulk download.
"""
import contextlib
import csv
import io
import json
import runpy
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THREE = ['SAG_IW_TSE', 'COR_IW_TSE', 'AX_MPR']
FIVE = THREE + ['COR_MPR', 'SAG_3D_DESS']

def main():
    with contextlib.redirect_stdout(io.StringIO()):
        audit = runpy.run_path(str(ROOT / 'investigate_followup.py'))
    statuses, basepairs = audit['statuses'], audit['basepairs']
    targets = audit['targets']
    base_subjects = {k[0] for k in basepairs}
    ready = json.loads((ROOT / 'score_audit/followup_index_download_status.json').read_text())
    VISITS = ready['packages']
    if set(VISITS) != {'V01','V03','V05','V06'}:
        raise ValueError('Expected all four user-authorized visit package IDs in index status')
    images = defaultdict(list)
    source_checks = {}
    for visit, package in VISITS.items():
        if visit not in ready['completed_visits']:
            continue
        table = ROOT / 'followup_indices' / visit / 'image03.txt'
        inventory = ROOT / f'tool_state/NDA/nda-tools/downloadcmd/packages/{package}/package_file_metadata_{package}.txt'
        with inventory.open(encoding='utf-8-sig', newline='') as f:
            sizes = {r['NDA_S3_URL']: int(r['FILE_SIZE']) for r in csv.DictReader(f)}
        rows = audit['read_nda'](table)
        wrong_visit, absent_size = 0, 0
        for r in rows:
            if r['visit'] != visit:
                wrong_visit += 1
                continue
            desc = r['image_description']
            if r['image_modality'] != 'MRI' or '_' not in desc:
                continue
            seq, side = desc.rsplit('_', 1)
            if seq not in FIVE or side not in ('LEFT', 'RIGHT'):
                continue
            url = r['image_file']
            if not url.endswith('.tar.gz'):
                continue
            size = sizes.get(url)
            if size is None:
                absent_size += 1
                continue
            key = (r['src_subject_id'], {'RIGHT': '1', 'LEFT': '2'}[side], visit)
            images[key].append({'sequence': seq, 'image_file': url, 'bytes': size,
                                'date': r['interview_date'], 'side': side})
        source_checks[visit] = {'index_rows': len(rows), 'wrong_visit_rows': wrong_visit,
                                'MRI_rows_missing_inventory_size': absent_size}

    candidates = {k for k, ss in statuses.items() if k[2] in source_checks
                  and any(s in ('positive', 'negative') for s in ss.values())}
    matched = candidates & images.keys()

    def summarize(keys):
        keys = set(keys)
        three_files = {r['image_file']: r['bytes'] for k in keys for r in images[k] if r['sequence'] in THREE}
        five_files = {r['image_file']: r['bytes'] for k in keys for r in images[k]}
        def complete(k, sequences):
            return set(sequences) <= {r['sequence'] for r in images[k]}
        return {'exams': len(keys), 'subjects': len({k[0] for k in keys}),
                'all_three_sequences': sum(complete(k, THREE) for k in keys),
                'all_five_sequences': sum(complete(k, FIVE) for k in keys),
                'three_sequence_archives_including_duplicates': len(three_files),
                'three_sequence_bytes_including_duplicates': sum(three_files.values()),
                'five_sequence_archives_including_duplicates': len(five_files),
                'five_sequence_bytes_including_duplicates': sum(five_files.values()),
                'multiple_archive_slots': sum(v > 1 for k in keys for v in Counter(r['sequence'] for r in images[k]).values()),
                'multiple_imaging_dates': sum(len({r['date'] for r in images[k]}) > 1 for k in keys),
                'labels': {t: dict(Counter(statuses[k][t] for k in keys)) for t in targets}}

    report = {'complete_all_four_indices': set(source_checks) == set(VISITS),
              'source_checks': source_checks, 'scored_candidates_in_loaded_visits': len(candidates),
              'candidates_without_selected_MRI': len(candidates - matched),
              'matched_total': summarize(matched),
              'by_visit': {v: summarize(k for k in matched if k[2] == v) for v in source_checks},
              'strata': {
                  'new_subjects': summarize(k for k in matched if k[0] not in base_subjects),
                  'existing_subject_new_knee': summarize(k for k in matched if k[0] in base_subjects and k[:2] not in basepairs),
                  'existing_baseline_knee_followup': summarize(k for k in matched if k[:2] in basepairs)},
              'limitations': ['Before visit-aware surgical exclusions and image quality checks.',
                              'Uses existing conservative four-target mapping, not revised thresholds.',
                              'Reported archive bytes include duplicate series; no bulk download started.',
                              'Repeated visits are not independent new participants.']}
    out = ROOT / 'score_audit'
    (out / 'followup_MRI_match_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fields = ['src_subject_id', 'side', 'visit', 'sequence', 'image_file', 'bytes', 'date'] + targets
    with (out / 'followup_MRI_candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for k in sorted(matched):
            for r in images[k]:
                writer.writerow({'src_subject_id': k[0], 'visit': k[2], **r, **statuses[k]})
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
