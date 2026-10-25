"""Audit a fixed snapshot of completed OAI archives without extracting or changing them."""
import csv, gzip, io, json, sys, tarfile, time
from pathlib import Path
from collections import Counter
import traceback

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / 'python_deps'))
import numpy as np
import pydicom

destination = Path('D:/OAI/baseline_2399/image03')
out = root / 'quality_check'
out.mkdir(exist_ok=True)
sequences = {'SAG_IW_TSE', 'COR_IW_TSE', 'SAG_3D_DESS', 'COR_MPR', 'AX_MPR'}

def check(path, meta):
    result = dict(path=str(path), subject=meta['src_subject_id'], side=meta['side'],
                  sequence=meta['sequence'], expected_bytes=int(meta['bytes']),
                  actual_bytes=path.stat().st_size, status='passed', warnings=[])
    try:
        if result['expected_bytes'] <= 1:
            result['warnings'].append('Manifest size is a placeholder; size cannot be verified')
        elif result['actual_bytes'] != result['expected_bytes']:
            raise ValueError('Archive size differs from manifest')
        count, positions, shapes, descriptions, patients, dates, uids = 0, [], set(), set(), set(), set(), set()
        constant_frames = 0
        # The gzip stream is exhausted after tar traversal to validate the CRC/trailer.
        with gzip.open(path, 'rb') as stream:
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    payload = archive.extractfile(member).read()
                    try:
                        ds = pydicom.dcmread(io.BytesIO(payload))
                    except pydicom.errors.InvalidDicomError:
                        continue
                    if 'PixelData' not in ds:
                        continue
                    pixels = ds.pixel_array
                    if pixels.ndim != 2:
                        raise ValueError(f'Unexpected pixel shape: {pixels.shape}')
                    count += 1
                    shapes.add(tuple(pixels.shape))
                    constant_frames += int(pixels.min() == pixels.max())
                    descriptions.add(str(getattr(ds, 'SeriesDescription', '')))
                    patients.add(str(getattr(ds, 'PatientID', '')))
                    dates.add(str(getattr(ds, 'StudyDate', '')))
                    uids.add(str(getattr(ds, 'SeriesInstanceUID', '')))
                    if hasattr(ds, 'ImageOrientationPatient') and hasattr(ds, 'ImagePositionPatient'):
                        orientation = np.asarray(ds.ImageOrientationPatient, float)
                        positions.append(round(float(np.dot(np.cross(orientation[:3], orientation[3:]),
                                                            np.asarray(ds.ImagePositionPatient, float))), 4))
            while stream.read(1024 * 1024):
                pass
        if not count:
            raise ValueError('No DICOM pixel frames found')
        result.update(gzip_crc_valid=True, dicom_frames=count, pixel_shapes=sorted(shapes),
                      series_descriptions=sorted(descriptions), patient_ids=sorted(patients),
                      study_dates=sorted(dates), series_uid_count=len(uids), constant_frames=constant_frames)
        expected_description = meta['sequence'] + '_' + meta['side']
        if any(description.upper() != expected_description for description in descriptions):
            result['warnings'].append('SeriesDescription differs from manifest sequence/side')
        if any(patient and meta['src_subject_id'] not in patient for patient in patients):
            result['warnings'].append('PatientID differs from manifest subject; review identification convention')
        expected_date = path.parent.name
        if any(date and date != expected_date for date in dates):
            result['warnings'].append('StudyDate differs from archive directory date')
        if len(shapes) > 1 or len(uids) > 1:
            result['warnings'].append('Mixed pixel shapes or multiple series UIDs')
        if positions and len(set(positions)) != len(positions):
            result['warnings'].append('Repeated physical slice positions')
        if constant_frames:
            result['warnings'].append('Constant intensity frame(s)')
        if result['warnings']:
            result['status'] = 'review'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    return result

def main():
    watching = '--watch' in sys.argv
    with (root / 'score_audit/all_mri_candidates.csv').open(encoding='utf-8', newline='') as f:
        candidates = list(csv.DictReader(f))
    selected = {}
    for meta in candidates:
        if meta['sequence'] not in sequences:
            continue
        path = destination / meta['image_file'].split('/', 4)[4]
        selected[str(path)] = (path, meta)
    snapshot = [(p, m) for p, m in selected.values() if p.is_file()]
    (out / 'snapshot.json').write_text(json.dumps([str(p) for p, _ in snapshot], indent=2), encoding='utf-8')
    records, started = {}, time.time()
    results_file = out / 'archive_results.jsonl'
    if watching and results_file.exists():
        for line in results_file.read_text(encoding='utf-8').splitlines():
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            if previous['path'] in selected:
                records[previous['path']] = previous
    def save(done=False):
        results = list(records.values())
        summary = dict(snapshot_archives=len(snapshot), checked=len(results),
                       expected_archives=len(selected), watch_mode=watching,
                       updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                       counts=dict(Counter(r['status'] for r in results)),
                       sequence_counts=dict(Counter(r['sequence'] for r in results)),
                       dicom_frames=sum(r.get('dicom_frames', 0) for r in results),
                       elapsed_seconds=round(time.time()-started, 1), complete=done,
                       scope=('Continuously checks completed archives; excludes .partial files.' if watching
                              else 'Completed archives present at audit start; later downloads are excluded.'),
                       issues=[r for r in results if r['status'] != 'passed'])
        tmp = out / 'summary.tmp'
        tmp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        tmp.replace(out / 'summary.json')
    save()
    with results_file.open('a' if watching else 'w', encoding='utf-8') as output:
        while True:
            if watching:
                snapshot = [(p, m) for p, m in selected.values() if p.is_file()]
            for path, meta in snapshot:
                stat = path.stat()
                previous = records.get(str(path))
                if previous and previous['actual_bytes'] == stat.st_size and (
                    'mtime_ns' not in previous or previous['mtime_ns'] == stat.st_mtime_ns
                ):
                    continue
                result = check(path, meta)
                result['mtime_ns'] = stat.st_mtime_ns
                records[str(path)] = result
                output.write(json.dumps(result) + '\n')
                output.flush()
                if len(records) % 10 == 0:
                    save()
            done = len(records) == len(selected) if watching else True
            save(done)
            if done:
                break
            time.sleep(30)
    print(json.dumps(dict(checked=len(records), counts=dict(Counter(r['status'] for r in records.values())))))

if __name__ == '__main__':
    main()
