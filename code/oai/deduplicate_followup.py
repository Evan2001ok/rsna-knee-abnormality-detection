"""Conservative metadata deduplication; no archive download or deletion."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'score_audit/followup_deduplicated'
THREE = {'SAG_IW_TSE', 'COR_IW_TSE', 'AX_MPR'}
KEY = ('src_subject_id', 'side', 'visit')

def write_csv(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'score_audit/followup_MRI_candidates.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        original = list(reader)
    unique = {}
    removed = []
    for r in original:
        url = r['image_file']
        if url in unique:
            if r != unique[url]:
                raise ValueError('Conflicting metadata for one archive: ' + url)
            removed.append(r)
        else:
            unique[url] = r
    rows = list(unique.values())
    slots, exams = defaultdict(list), defaultdict(list)
    for r in rows:
        ek = tuple(r[k] for k in KEY)
        exams[ek].append(r)
        slots[ek + (r['sequence'],)].append(r)
    multi_slots = {k for k, rr in slots.items() if len(rr) > 1}
    multi_dates = {k for k, rr in exams.items() if len({r['date'] for r in rr}) > 1}
    ready, review = [], []
    for r in rows:
        ek = tuple(r[k] for k in KEY)
        reasons = []
        if ek in multi_dates:
            reasons.append('multiple_imaging_dates_in_exam')
        if ek + (r['sequence'],) in multi_slots:
            reasons.append('multiple_distinct_archives_in_slot')
        if reasons:
            review.append(dict(r, review_reason=';'.join(reasons)))
        else:
            ready.append(r)
    write_csv('unique_archive_candidates.csv', rows, fields)
    write_csv('removed_exact_duplicate_records.csv', removed, fields)
    write_csv('unambiguous_five_sequences.csv', ready, fields)
    write_csv('unambiguous_three_sequences.csv', [r for r in ready if r['sequence'] in THREE], fields)
    write_csv('requires_review.csv', review, fields + ['review_reason'])
    def stats(rr):
        return {'archives': len(rr), 'bytes': sum(int(r['bytes']) for r in rr),
                'exams': len({tuple(r[k] for k in KEY) for r in rr})}
    report = {'input_records': len(original), 'exact_duplicate_records_removed': len(removed),
              'distinct_exam_sequence_slots': len(slots), 'exams': len(exams),
              'multiple_archive_slots': len(multi_slots), 'multiple_date_exams': len(multi_dates),
              'all_candidates_five': stats(rows),
              'all_candidates_three': stats([r for r in rows if r['sequence'] in THREE]),
              'unambiguous_five': stats(ready),
              'unambiguous_three': stats([r for r in ready if r['sequence'] in THREE]),
              'requires_review_five': stats(review),
              'requires_review_three': stats([r for r in review if r['sequence'] in THREE]),
              'policy': ['Different visits and knee sides are retained.',
                         'Only identical file URL and matching metadata may be removed automatically.',
                         'Different archive URLs are not evidence of identical image pixels.',
                         'Entire multi-date exams are held for review to prevent mixing dates.',
                         'No arbitrary choice by file size, date, or archive ID.',
                         'Unambiguous CSVs are partial download lists; reviewed slots must be added later.',
                         'No bulk downloads, deletions, or label changes performed.']}
    assert len(ready) + len(review) == len(rows)
    assert len({tuple(r[k] for k in KEY) + (r['sequence'],) for r in ready}) == len(ready)
    assert len(exams) == 6250
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
