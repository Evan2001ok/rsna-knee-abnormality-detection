"""Incremental follow-up caches using the original supplied pixel transforms."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import ast, csv, io, json, sys, tarfile, time, shutil
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'oai_torch_deps'))
sys.path.insert(0, str(ROOT.parent / 'python_deps'))
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
SLOTS = ['SAG_IW_TSE', 'COR_IW_TSE', 'AX_MPR']
planes = dict(zip(SLOTS, ['Sagittal', 'Coronal', 'Axial']))
STATUS = ROOT / 'preprocess_status_followup_dual'
STATUS.mkdir(exist_ok=True)
DEST = Path('D:/OAI/followup_6250_three_sequences')

def load_functions(resolution):
    attachment = (str(Path(__file__).resolve().parent / 'transforms/source224.py')
                  if resolution == 224 else
                  str(Path(__file__).resolve().parent / 'transforms/source384.py'))
    text = Path(attachment).read_text(encoding='utf-8-sig').split('def read_slot(', 1)[0]
    tree = ast.parse(text)
    names = {'_sample_indices', '_crop_mm', '_to_uint8', '_physical_slice_normal', '_canonicalize_side'}
    scope = dict(globals())
    scope['cache_root'] = Path(f'D:/OAI/preprocessed_followup_{resolution}_series')
    scope['cache_root'].mkdir(parents=True, exist_ok=True)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), attachment, 'exec'), scope)
    old = (ROOT / ('preprocess_downloads.py' if resolution == 224 else 'preprocess_downloads_384.py')).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == 'convert')
    code = ast.get_source_segment(old, node)
    code = code.replace("cache_root/meta['src_subject_id']/meta['side']/meta['sequence']", "cache_root/meta['sample_id']/meta['sequence']")
    # Original conservative geometry guards are retained; failures remain review items.
    exec(compile(code, '<original-converter>', 'exec'), scope)
    return scope['convert']

def records(path, key):
    result = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
                result[item[key]] = item
            except json.JSONDecodeError:
                continue
    return result

def main():
    with (ROOT / 'score_audit/followup_deduplicated/download_manifest_three_sequences.csv').open(encoding='utf-8-sig', newline='') as f:
        manifest = list(csv.DictReader(f))
    selected = {}
    for row in manifest:
        path = DEST / row['visit'] / 'image03' / row['image_file'].split('/', 4)[4]
        row['sample_id'] = '_'.join(row[k] for k in ('src_subject_id', 'side', 'visit'))
        selected[str(path)] = row
    convert = {r: load_functions(r) for r in (224, 384)}
    ledger = STATUS / 'series_results.jsonl'
    done = records(ledger, 'key')
    with ledger.open('a', encoding='utf-8') as log:
        while True:
            quality = records(ROOT / 'quality_check_followup/archive_results.jsonl', 'path')
            for path, qc in quality.items():
                if path not in selected or qc.get('status') == 'failed' or not qc.get('gzip_crc_valid'):
                    continue
                source = Path(path)
                if not source.exists() or source.stat().st_size != qc['actual_bytes']:
                    continue
                if qc.get('mtime_ns') != source.stat().st_mtime_ns:
                    continue
                for resolution in (224, 384):
                    key = f'{resolution}:{path}'
                    previous = done.get(key)
                    if previous and (previous['status'] != 'passed' or Path(previous['output']).exists()):
                        continue
                    if shutil.disk_usage('D:/').free < 20_000_000_000:
                        raise RuntimeError('Free disk below 20 GB reserve; processing stopped without deleting data')
                    meta = selected[path]
                    try:
                        result = convert[resolution](source, meta)
                        if result['empty_output_slices'] or result['distinct_sampled_slices'] != 24:
                            raise ValueError('Blank or repeated sampled slices')
                    except Exception as exc:
                        result = dict(source=path, status='review', error=f'{type(exc).__name__}: {exc}')
                    result.update(key=key, resolution=resolution, sample_id=meta['sample_id'], visit=meta['visit'])
                    done[key] = result
                    log.write(json.dumps(result) + '\n')
                    log.flush()
                    summary = dict(expected_series=len(selected), processed=len(done),
                        counts=dict(Counter(f"{r['resolution']}:{r['status']}" for r in done.values())),
                        cache_bytes=sum(r.get('output_bytes', 0) for r in done.values()),
                        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), complete=len(done)==2*len(selected))
                    tmp = STATUS / 'summary.tmp'
                    tmp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
                    for attempt in range(20):
                        try:
                            tmp.replace(STATUS / 'summary.json')
                            break
                        except PermissionError:
                            time.sleep(0.1)
                    else:
                        print('Status file temporarily locked; durable ledger retained', flush=True)
            current_quality = {p: q for p, q in quality.items() if p in selected
                and Path(p).is_file() and q.get('mtime_ns') == Path(p).stat().st_mtime_ns
                and q.get('actual_bytes') == Path(p).stat().st_size}
            if len(current_quality) == len(selected):
                break
            time.sleep(30)

if __name__ == '__main__':
    main()
