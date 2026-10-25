"""Incremental OAI series cache; exact pixel transforms from supplied preprocessing code."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import ast, csv, io, json, sys, tarfile, time
from pathlib import Path
from collections import Counter
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/'oai_torch_deps'))
sys.path.insert(0, str(root.parent/'python_deps'))
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

source = Path(__file__).resolve().parent / 'transforms/source224.py'
names = {'_sample_indices','_crop_mm','_to_uint8','_physical_slice_normal','_canonicalize_side'}
source_text = source.read_text(encoding='utf-8-sig')
# Pasted file includes notebook output after its code; only parse the transform section.
tree = ast.parse(source_text.split('def read_slot(',1)[0])
functions = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names], type_ignores=[])
exec(compile(functions,str(source),'exec'),globals())
destination = Path('D:/OAI/baseline_2399/image03')
cache_root = Path('D:/OAI/preprocessed_224_series')
cache_root.mkdir(exist_ok=True)
report_root = root/'preprocess_status'
report_root.mkdir(exist_ok=True)
planes = {'SAG_IW_TSE':'Sagittal','COR_IW_TSE':'Coronal','AX_MPR':'Axial',
          'COR_MPR':'Coronal','SAG_3D_DESS':'Sagittal'}
with (root/'score_audit/all_mri_candidates.csv').open(encoding='utf-8',newline='') as f:
    manifest = {str(destination/r['image_file'].split('/',4)[4]):r for r in csv.DictReader(f) if r['sequence'] in planes}
watch = '--watch' in sys.argv
limit = int(next((a.split('=',1)[1] for a in sys.argv if a.startswith('--limit=')), '20'))
ledger = report_root/'series_results.jsonl'
records = {}
if ledger.exists():
    for line in ledger.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
            records[r['source']] = r
        except json.JSONDecodeError:
            continue
started = time.time()

def convert(path, meta):
    start = time.time()
    datasets = []
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(archive.extractfile(member).read()))
            except pydicom.errors.InvalidDicomError:
                continue
            if 'PixelData' in ds:
                datasets.append((member.name,ds))
    if len(datasets)<2:
        raise ValueError('Too few DICOM slices')
    plane = planes[meta['sequence']]
    for _,ds in datasets:
        if str(getattr(ds,'PatientID','')) != meta['src_subject_id']:
            raise ValueError('PatientID does not match manifest')
        if str(getattr(ds,'SeriesDescription','')).upper()!=meta['sequence']+'_'+meta['side']:
            raise ValueError('Sequence/side does not match manifest')
        if not hasattr(ds,'ImageOrientationPatient') or not hasattr(ds,'ImagePositionPatient'):
            raise ValueError('Missing physical ordering geometry; manual review required')
    original_count=len(datasets)
    axis = {'Sagittal':0,'Coronal':1,'Axial':2}[plane]
    normals=[_physical_slice_normal(np.asarray(ds.ImageOrientationPatient,float),plane) for _,ds in datasets]
    expected=[n for n in normals if n is not None and abs(n[axis])>=0.9]
    if not expected:
        raise ValueError('No frames in expected plane')
    normal = expected[0]
    retained, excluded = [], []
    for item,n in zip(datasets,normals):
        if n is None or abs(n[axis])<0.9 or np.dot(n,normal)<0.999:
            excluded.append(item[0])
        else:
            retained.append(item)
    if len(excluded)>max(1,int(0.05*original_count)):
        raise ValueError('Too many inconsistent orientation frames')
    datasets=retained
    if normal is None:
        raise ValueError('Invalid orientation')
    if abs(normal[axis])<0.9:
        raise ValueError('Orientation inconsistent with expected plane')
    for _,ds in datasets:
        n = _physical_slice_normal(np.asarray(ds.ImageOrientationPatient,float),plane)
        if n is None or np.dot(n,normal)<0.999:
            raise ValueError('Orientation changes inside series')
        if plane in ('Coronal','Axial') and float(ds.ImageOrientationPatient[0])<0.9:
            raise ValueError('In-plane orientation differs from original flip assumption')
    datasets.sort(key=lambda item:(float(np.dot(np.asarray(item[1].ImagePositionPatient,float),normal)),
                                    float(getattr(item[1],'InstanceNumber',0)),item[0]))
    positions=[float(np.dot(np.asarray(ds.ImagePositionPatient,float),normal)) for _,ds in datasets]
    if len(set(round(p,4) for p in positions)) != len(positions):
        raise ValueError('Duplicate slice positions')
    idx = _sample_indices(len(datasets),24)
    slices=[]
    for i in idx:
        ds = datasets[int(i)][1]
        if str(getattr(ds,'PhotometricInterpretation',''))!='MONOCHROME2':
            raise ValueError('Photometric interpretation differs from supplied preprocessing')
        pixels=ds.pixel_array.astype(np.float32)
        slices.append(pixels*float(getattr(ds,'RescaleSlope',1) or 1)+float(getattr(ds,'RescaleIntercept',0) or 0))
    if len({s.shape for s in slices}) != 1:
        raise ValueError('Inconsistent pixel shapes')
    ds=datasets[0][1]
    spacing=np.asarray(ds.PixelSpacing,float)
    if len(spacing)!=2 or not np.allclose(spacing[0],spacing[1],rtol=0.001):
        raise ValueError('Non-square pixel spacing needs crop adaptation')
    vol,want=_crop_mm(np.stack(slices),float(spacing[0]),160.0)
    x=_to_uint8(vol,224,want)
    x=_canonicalize_side(x,'R' if meta['side']=='RIGHT' else 'L',plane,True)
    array=x.numpy()
    if array.shape!=(24,224,224) or array.dtype!=np.uint8 or array.max()==array.min():
        raise ValueError('Invalid or constant output cache')
    target=cache_root/meta['src_subject_id']/meta['side']/meta['sequence']/(path.stem.removesuffix('.tar')+'.npy')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.save(f,array)
    temporary.replace(target)
    return dict(source=str(path),output=str(target),status='passed',subject=meta['src_subject_id'],side=meta['side'],
                sequence=meta['sequence'],source_slices=original_count,retained_slices=len(datasets),
                excluded_orientation_members=excluded,sampled_indices=idx.tolist(),
                distinct_sampled_slices=len(set(idx)),shape=list(array.shape),dtype=str(array.dtype),
                output_bytes=target.stat().st_size,seconds=round(time.time()-start,3),
                empty_output_slices=int(np.sum(np.ptp(array,axis=(1,2))==0)))

def save():
    values=list(records.values())
    result=dict(mode='watch' if watch else 'pilot',updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                checked=len(values),counts=dict(Counter(r['status'] for r in values)),
                output_bytes=sum(r.get('output_bytes',0) for r in values),elapsed_seconds=round(time.time()-started,1),
                series_with_excluded_orientation_frames=sum(bool(r.get('excluded_orientation_members')) for r in values),
                excluded_orientation_frames=sum(len(r.get('excluded_orientation_members',[])) for r in values),
                cache_root=str(cache_root),workers=1,torch_threads=1,
                config=dict(img_size=224,crop_mm=160,slices=24,dtype='uint8',central_fraction=0.8),
                sequence_schema=list(planes),
                scope='Per-series incremental cache. OAI sequence names retained; not RSNA T1 slots. Final knee-level packing and labels pending.',
                issues=[r for r in values if r['status']!='passed'])
    tmp=report_root/'summary.tmp'
    tmp.write_text(json.dumps(result,indent=2),encoding='utf-8')
    tmp.replace(report_root/'summary.json')

converted=0
with ledger.open('a',encoding='utf-8') as log:
    while True:
        quality={}
        for line in (root/'quality_check/archive_results.jsonl').read_text(encoding='utf-8').splitlines():
            try:
                r=json.loads(line)
                quality[r['path']]=r
            except json.JSONDecodeError:
                continue
        pending=[(p,r) for p,r in quality.items() if p in manifest and r.get('gzip_crc_valid') and r['status']!='failed' and
                 (p not in records or ('--retry-failed' in sys.argv and records[p]['status']=='failed') or
                  (records[p]['status']=='passed' and not Path(records[p]['output']).exists()))]
        # Pilot covers all five sequences rather than only the first download group.
        if not watch:
            buckets={s:[] for s in planes}
            for p,r in pending:
                buckets[r['sequence']].append((p,r))
            pending=[]
            while any(buckets.values()):
                for bucket in buckets.values():
                    if bucket:
                        pending.append(bucket.pop(0))
        for path,quality_record in pending:
            meta=manifest[path]
            try:
                result=convert(Path(path),meta)
            except Exception as exc:
                result=dict(source=path,status='failed',sequence=meta['sequence'],error=f'{type(exc).__name__}: {exc}')
            records[path]=result
            log.write(json.dumps(result)+'\n')
            log.flush()
            converted+=1
            save()
            if not watch and converted>=limit:
                break
            time.sleep(0.1)
        save()
        if not watch or len(records)==len(manifest):
            break
        time.sleep(30)
print(json.dumps(dict(converted=converted,counts=dict(Counter(r['status'] for r in records.values())))))
