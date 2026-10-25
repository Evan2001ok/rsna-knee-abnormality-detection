"""Pack final three-slot follow-up shards. No original archives are modified."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.lib.format import open_memmap
ROOT = Path(__file__).resolve().parent
SLOTS = ['SAG_IW_TSE','COR_IW_TSE','AX_MPR']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,choices=(224,384),required=True)
    parser.add_argument('--check-ready',action='store_true')
    parser.add_argument('--release-generated-series',action='store_true',
        help='After each verified shard, remove only its generated intermediate arrays; preserve raw archives')
    args = parser.parse_args()
    cache_root = Path(f'D:/OAI/preprocessed_followup_{args.resolution}_series').resolve()
    output = Path(f'D:/OAI/kaggle_upload_followup6250_{args.resolution}')
    records = {}
    for line in (ROOT/'preprocess_status_followup_dual/series_results.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
            if r['resolution']==args.resolution:
                records[r['source']] = r
        except json.JSONDecodeError:
            pass
    with (ROOT/'score_audit/followup_deduplicated/download_manifest_three_sequences.csv').open(encoding='utf-8-sig',newline='') as f:
        manifest = list(csv.DictReader(f))
    exams = defaultdict(list)
    missing = []
    for r in manifest:
        source = str(Path('D:/OAI/followup_6250_three_sequences')/r['visit']/'image03'/r['image_file'].split('/',4)[4])
        if source not in records:
            missing.append(source)
        exams['_'.join(r[k] for k in ('src_subject_id','side','visit'))].append((r,records.get(source)))
    tasks = ROOT/'soft_followup6250_v3/tasks'
    ready = not missing and len(list(tasks.glob('*.json'))) == len(exams)
    if args.check_ready:
        print(json.dumps(dict(ready=ready,missing_series=len(missing),exams=len(exams),tasks=len(list(tasks.glob('*.json'))))))
        return
    if not ready:
        raise RuntimeError('Wait for all series processing and all evidence tasks before packing')
    output.mkdir(parents=True,exist_ok=True)
    task_map = {}
    for path in tasks.glob('*.json'):
        item = json.loads(path.read_text(encoding='utf-8'))
        task_map[item['knee_id']] = item
    ids = sorted(exams)
    parts, audit, conflicts = [], [], []
    for part,start in enumerate(range(0,len(ids),500)):
        batch = ids[start:start+500]
        parts.append(len(batch))
        target = output/f'part{part}_cache.npy'
        marker = output/f'part{part}_VERIFIED.json'
        if marker.exists():
            old = json.loads(marker.read_text())
            if target.stat().st_size != old['bytes'] or digest(target)!=old['sha256']:
                raise ValueError('Previously packed shard changed')
            for item in old.get('sidecars',[]):
                if digest(output/item['name']) != item['sha256']:
                    raise ValueError('Previously packed IDs/mask changed')
            conflicts.extend(old.get('conflicts',[]))
            continue
        required = len(batch)*3*24*args.resolution**2+2_000_000_000
        if shutil.disk_usage(output).free < required:
            raise RuntimeError('Insufficient space for next shard; preserve sources')
        tmp = output/f'part{part}_cache.partial.npy'
        a = open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(batch),3,24,args.resolution,args.resolution))
        a[:] = 0
        masks = np.zeros((len(batch),3),np.uint8)
        retained, releases = {}, []
        conflict_start = len(conflicts)
        for row,sid in enumerate(batch):
            task = task_map[sid]
            if not task['identity_timepoint_confirmed']:
                conflicts.append(dict(knee_id=sid,reason='identity/date unresolved; all slots masked'))
            for slot,seq in enumerate(SLOTS):
                candidates = [r for m,r in exams[sid] if m['sequence']==seq and r['status']=='passed']
                if not candidates or not task['identity_timepoint_confirmed']:
                    continue
                arrays = [np.load(r['output'],mmap_mode='r',allow_pickle=False) for r in candidates]
                if any(not np.array_equal(arrays[0],x) for x in arrays[1:]):
                    conflicts.append(dict(knee_id=sid,sequence=seq,reason='multiple nonidentical archive candidates; slot masked'))
                    del arrays
                    continue
                a[row,slot] = arrays[0]
                masks[row,slot] = 1
                retained[(row,slot)] = Path(candidates[0]['output'])
                releases.extend(Path(r['output']) for r in candidates)
                del arrays
            m = exams[sid][0][0]
            audit.append(dict(knee_id=sid,src_subject_id=m['src_subject_id'],side=m['side'],visit=m['visit'],part=part,row=row))
        a.flush()
        del a
        check = np.load(tmp,mmap_mode='r',allow_pickle=False)
        for (row,slot),source in retained.items():
            if not np.array_equal(check[row,slot],np.load(source,mmap_mode='r',allow_pickle=False)):
                raise ValueError('Packed pixel verification failed')
        del check
        tmp.replace(target)
        np.save(output/f'part{part}_knee_ids.npy',np.asarray(batch))
        np.save(output/f'part{part}_series_mask.npy',masks)
        sidecars = [dict(name=f'part{part}_{suffix}.npy',sha256=digest(output/f'part{part}_{suffix}.npy'))
                    for suffix in ('knee_ids','series_mask')]
        marker.write_text(json.dumps(dict(bytes=target.stat().st_size,sha256=digest(target),pixel_verified=True,
            sidecars=sidecars,conflicts=conflicts[conflict_start:])),encoding='utf-8')
        if args.release_generated_series:
            for source in set(releases):
                resolved = source.resolve()
                if not resolved.is_relative_to(cache_root) or resolved.suffix!='.npy':
                    raise ValueError('Intermediate deletion outside intended cache root refused')
                resolved.unlink(missing_ok=True)
        print(f'Verified {args.resolution} shard {part}: {len(batch)} exams',flush=True)
    # Reconstruct index even after resuming verified shards.
    audit = [dict(knee_id=sid,src_subject_id=exams[sid][0][0]['src_subject_id'],side=exams[sid][0][0]['side'],
                  visit=exams[sid][0][0]['visit'],part=i//500,row=i%500) for i,sid in enumerate(ids)]
    with (output/'index.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    (output/'manifest.json').write_text(json.dumps(dict(slot_order=SLOTS,parts=parts,checks=len(ids),
        resolution=args.resolution,crop_mm=160 if args.resolution==224 else 130,slices=24,central_fraction=.8,
        dtype='uint8',id_schema='subject_side_visit',label_join='knee_id plus visit'),indent=2),encoding='utf-8')
    (output/'packing_conflicts.json').write_text(json.dumps(conflicts,indent=2),encoding='utf-8')
    # Independently verify structural identity, validity masks and missing-slot zeros.
    verified_ids = []
    for part,n in enumerate(parts):
        a = np.load(output/f'part{part}_cache.npy',mmap_mode='r',allow_pickle=False)
        knee_ids = np.load(output/f'part{part}_knee_ids.npy',allow_pickle=False).astype(str)
        masks = np.load(output/f'part{part}_series_mask.npy',allow_pickle=False)
        if a.shape != (n,3,24,args.resolution,args.resolution) or a.dtype != np.uint8:
            raise ValueError('Final cache structure mismatch')
        if knee_ids.shape != (n,) or masks.shape != (n,3) or not np.isin(masks,[0,1]).all():
            raise ValueError('Final ID/mask structure mismatch')
        for row in range(n):
            for slot in range(3):
                if masks[row,slot]==0 and np.any(a[row,slot]):
                    raise ValueError('Masked slot contains nonzero data')
                if masks[row,slot]==1 and np.any(np.ptp(a[row,slot],axis=(1,2))==0):
                    raise ValueError('Valid slot has blank slice')
        verified_ids.extend(knee_ids.tolist())
        del a
    if verified_ids != ids or len(set(verified_ids)) != len(ids):
        raise ValueError('Final exam ordering mismatch')
    files = [dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in output.iterdir() if p.is_file() and p.name!='UPLOAD_READY.json']
    (output/'UPLOAD_READY.json').write_text(json.dumps(dict(verified=True,checks=len(ids),files=files),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
