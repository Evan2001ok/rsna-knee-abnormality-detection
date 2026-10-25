"""Export the full unique three-sequence candidate download queue, not adjudicated labels.

Retains distinct archive candidates, visits and sides. QC, geometry/identity review,
and the original packer's conflict masking must still run before training.
"""
import argparse
import csv
import json
from pathlib import Path

SLOTS = {'SAG_IW_TSE', 'COR_IW_TSE', 'AX_MPR'}
VISITS = ('V01', 'V03', 'V05', 'V06')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expected-exams', type=int)
    p.add_argument('--expected-archives', type=int)
    args = p.parse_args()
    with Path(args.candidates).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not {'src_subject_id', 'side', 'visit', 'sequence', 'image_file', 'bytes', 'date'} <= set(fields or []):
            raise ValueError('Missing candidate columns')
        unique = {}
        for row in reader:
            if row['sequence'] not in SLOTS:
                continue
            if row['side'] not in ('LEFT', 'RIGHT') or row['visit'] not in VISITS or int(row['bytes']) <= 0:
                raise ValueError('Invalid identity/visit/size')
            key = row['image_file']
            if key in unique and unique[key] != row:
                raise ValueError('Conflicting metadata for one archive')
            unique[key] = row
    rows = sorted(unique.values(), key=lambda r: tuple(r[k] for k in ('src_subject_id', 'side', 'visit', 'sequence', 'image_file')))
    exams = {tuple(r[k] for k in ('src_subject_id', 'side', 'visit')) for r in rows}
    if not rows:
        raise ValueError('Empty three-sequence queue')
    if args.expected_exams is not None and len(exams) != args.expected_exams:
        raise ValueError('Exam count differs from requested snapshot')
    if args.expected_archives is not None and len(rows) != args.expected_archives:
        raise ValueError('Archive count differs from requested snapshot')
    out = Path(args.output)
    targets = [out / 'download_manifest_three_sequences.csv', *[out / (v + '_three_sequence_s3_links.txt') for v in VISITS], out / 'download_queue_summary.json']
    if any(f.exists() for f in targets):
        raise ValueError('Queue outputs already exist; use a separate output folder to preserve the previous snapshot')
    out.mkdir(parents=True, exist_ok=True)
    with targets[0].open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for visit, path in zip(VISITS, targets[1:5]):
        links = [r['image_file'] for r in rows if r['visit'] == visit]
        path.write_text(''.join(s + '\n' for s in links), encoding='utf-8')
    report = dict(exams=len(exams), archives=len(rows), bytes=sum(int(r['bytes']) for r in rows),
                  policy='Download candidates only; distinct archive/date candidates retained for original QC/task/packer adjudication; no label evidence generated')
    targets[-1].write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
