"""Pack completed OAI caches into two knee-level shards for a private Kaggle dataset."""
import csv, hashlib, json, shutil, sys, time
from collections import defaultdict, Counter
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'python_deps'))
import numpy as np
from numpy.lib.format import open_memmap
slots=['SAG_IW_TSE','COR_IW_TSE','AX_MPR','COR_MPR','SAG_3D_DESS']
output=Path('D:/OAI/kaggle_upload_224')
status_path=root/'preprocess_status/packing_status.json'

def read_ledger(path,key):
    rows={}
    for line in path.read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        rows[r[key]]=r
    return rows

def progress(value):
    value['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    status_path.write_text(json.dumps(value,indent=2),encoding='utf-8')

quality=read_ledger(root/'quality_check/archive_results.jsonl','path')
prep=read_ledger(root/'preprocess_status/series_results.jsonl','source')
with (root/'score_audit/all_mri_candidates.csv').open(encoding='utf-8',newline='') as f:
    expected={str(Path('D:/OAI/baseline_2399/image03')/r['image_file'].split('/',4)[4])
              for r in csv.DictReader(f) if r['sequence'] in slots}
if len(expected)!=12053:
    raise SystemExit('Expected manifest count changed; review before packing')
missing_files=[p for p in expected if not Path(p).is_file()]
missing_quality=expected-set(quality)
bad_quality=[p for p in expected & set(quality) if quality[p]['status']=='failed']
pending_preprocess={p for p in expected & set(quality) if quality[p]['status']!='failed'}-set(prep)
readiness=dict(ready=not(missing_files or missing_quality or pending_preprocess or bad_quality),
               expected_files=len(expected),missing_files=len(missing_files),
               unchecked_files=len(missing_quality),pending_preprocess=len(pending_preprocess),
               corrupted_files=len(bad_quality),preprocess_skipped=sum(prep[p]['status']!='passed' for p in expected & set(prep)))
if '--check-ready' in sys.argv:
    print(json.dumps(readiness));sys.exit(0)
if not readiness['ready']:
    raise SystemExit(json.dumps(readiness))
output.mkdir(exist_ok=True)
if (output/'UPLOAD_READY.json').exists():
    print('Packing already complete: '+str(output/'UPLOAD_READY.json'));sys.exit(0)
need=2399*5*24*224*224
if shutil.disk_usage(output).free<need+5_000_000_000:
    raise SystemExit('Insufficient space for final shards and reserve')
with (root/'score_audit/major_moaks_baseline_knees.csv').open(encoding='utf-8',newline='') as f:
    knees=sorted((r['src_subject_id'],r['side']) for r in csv.DictReader(f))
assert len(knees)==2399 and len(set(knees))==2399
groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
for path,r in prep.items():
    if path in expected and r['status']=='passed':
        groups[(r['subject'],r['side'])][Path(path).parent.name][r['sequence']].append(r)
selections={}
for key in knees:
    dates=groups[key]
    if dates:
        date=sorted(dates,key=lambda d:(-len(dates[d]),d))[0]
        selections[key]=(date,{seq:sorted(rows,key=lambda r:(-r.get('retained_slices',r['source_slices']),r['source']))[0]
                              for seq,rows in dates[date].items()})
    else:
        selections[key]=(None,{})
index_rows=[]
file_manifest=[]
processed=0
for part,part_keys in enumerate((knees[:1200],knees[1200:])):
    tmp=output/f'part{part}_cache.partial.npy'
    cache=open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(part_keys),5,24,224,224))
    masks=np.zeros((len(part_keys),5),dtype=np.uint8)
    ids=[]
    for row,key in enumerate(part_keys):
        date,selected=selections[key]
        ids.append(key[0]+'_'+key[1])
        cache[row]=0
        selected_ids={}
        for slot,seq in enumerate(slots):
            if seq not in selected:continue
            record=selected[seq]
            a=np.load(record['output'],mmap_mode='r')
            if a.shape!=(24,224,224) or a.dtype!=np.uint8 or a.min()==a.max():
                raise ValueError('Invalid cache: '+record['output'])
            cache[row,slot]=a
            masks[row,slot]=1
            selected_ids[seq]=Path(record['source']).name
        index_rows.append(dict(knee_id=ids[-1],src_subject_id=key[0],side=key[1],visit='V00',
                               study_date=date,part=part,row=row,valid_slots=int(masks[row].sum()),
                               selected_archives=selected_ids))
        processed+=1
        if processed%50==0:
            cache.flush()
            progress(dict(stage='packing',knees_packed=processed,total_knees=2399))
    cache.flush();del cache
    tmp.replace(output/f'part{part}_cache.npy')
    np.save(output/f'part{part}_series_mask.npy',masks)
    np.save(output/f'part{part}_knee_ids.npy',np.asarray(ids,dtype='U14'))
    # Compare persisted tensors with selected inputs, including mask-zero slots.
    check=np.load(output/f'part{part}_cache.npy',mmap_mode='r')
    for row,key in enumerate(part_keys):
        _,selected=selections[key]
        for slot,seq in enumerate(slots):
            if masks[row,slot]:
                original=np.load(selected[seq]['output'],mmap_mode='r')
                if not np.array_equal(check[row,slot],original):
                    raise ValueError('Packed/source pixels differ')
            elif np.any(check[row,slot]):
                raise ValueError('Masked slot is not zero-filled')
    del check
progress(dict(stage='hashing',knees_packed=processed,total_knees=2399))
for path in sorted(output.glob('*.npy')):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    file_manifest.append(dict(name=path.name,bytes=path.stat().st_size,sha256=h.hexdigest()))
metadata=dict(format_version='oai-224px-24sl-central80-crop160-v1',knees=2399,
              slot_order=slots,parts=[1200,1199],dtype='uint8',shape_per_knee=[5,24,224,224],
              missing_slot_policy='Zero-filled cache with series_mask=0; never infer negative labels from missing images.',
              selection_policy='Single date per knee: most valid sequence types, earliest date tie-break; most retained slices per sequence, archive name tie-break.',
              full_five_slot_knees=sum(r['valid_slots']==5 for r in index_rows),
              knees_with_no_valid_slots=sum(r['valid_slots']==0 for r in index_rows),
              slot_coverage={seq:sum(seq in selections[k][1] for k in knees) for seq in slots},
              preprocess_skipped=readiness['preprocess_skipped'],files=file_manifest,
              labels_included=False,
              note='OAI sequence schema differs from RSNA fluid/T1 slots. DESS and MPR are not renamed as T1. Final label mapping is a separate artifact.')
(output/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
(output/'knee_index.json').write_text(json.dumps(index_rows,indent=2),encoding='utf-8')
(output/'preprocess_exclusions.json').write_text(json.dumps([dict(subject=Path(p).parts[-3],
    archive=Path(p).name,sequence=r['sequence'],error=r['error']) for p,r in prep.items()
    if p in expected and r['status']=='failed'],indent=2),encoding='utf-8')
(output/'README.md').write_text('''# OAI baseline knee image cache

Private research cache uploaded under the account holder's confirmed permission.
2,399 baseline knees; two disjoint shards of 1,200 and 1,199 knees.
Each cache: uint8 [N,5,24,224,224], central-80% sampling and 160mm crop.
Each shard has its own knee_ids and series_mask in exactly the same row order.
Use mmap_mode="r" and allow_pickle=False when loading NPY files.

Slot order: SAG_IW_TSE, COR_IW_TSE, AX_MPR, COR_MPR, SAG_3D_DESS.
These are actual OAI sequences, not the RSNA fluid/T1 slot definitions.
Missing/rejected sequence slots are zero-filled and marked series_mask=0.
Retain both knees of one participant in the same training/validation split.
No final disease labels are included. Associate future labels by knee_id,
or src_subject_id + side + V00. Missing images are not negative labels.
See manifest.json for hashes, knee_index.json for row mapping and selected archives,
and preprocess_exclusions.json for rejected series.
''',encoding='utf-8')
(output/'UPLOAD_READY.json').write_text(json.dumps(dict(verified=True,files=file_manifest,
    target_owner='minminandy',target_slug='oai-baseline-2399-224px-images',private=True,
    labels_included=False),indent=2),encoding='utf-8')
progress(dict(stage='ready_to_upload',knees_packed=processed,total_knees=2399,output=str(output)))
print(json.dumps(metadata))
