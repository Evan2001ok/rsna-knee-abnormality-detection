"""Prepare v3 follow-up evidence only; never includes old labels or predictions."""
import ast, csv, json, sys, datetime
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'python_deps'))
import numpy as np
from PIL import Image, ImageDraw
from audit_scores import read_nda
OUT = ROOT / 'soft_followup6250_v3'
(OUT / 'tasks').mkdir(parents=True, exist_ok=True)
(OUT / 'images').mkdir(exist_ok=True)
T = ['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']
tree = ast.parse((ROOT / 'generate_labels_v1.py').read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(a, ast.Name) and a.id=='RULES' for a in n.targets))
fields = {k: v[0] for k,v in ast.literal_eval(node.value).items()}
tree = ast.parse((ROOT / 'prepare_soft_full2399_v3.py').read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(a, ast.Name) and a.id=='definitions' for a in n.targets))
definitions = ast.literal_eval(node.value)
dictionary = {r['ElementName']:r for r in csv.DictReader((ROOT/'score_dictionary.csv').open(encoding='utf-8-sig',newline=''))}
scores = defaultdict(list)
for line, r in enumerate(read_nda(ROOT/'scores/oai_kmrisemiquant01.txt'),3):
    if r.get('side') in ('1','2'):
        scores[(r['src_subject_id'], {'1':'RIGHT','2':'LEFT'}[r['side']],r['visit'])].append((line,r))
manifest = defaultdict(list)
for r in csv.DictReader((ROOT/'score_audit/followup_deduplicated/download_manifest_three_sequences.csv').open(encoding='utf-8-sig',newline='')):
    sid = '_'.join(r[k] for k in ('src_subject_id','side','visit'))
    manifest[sid].append(r)
records = {}
quality = {}
for line in (ROOT/'quality_check_followup/archive_results.jsonl').read_text(encoding='utf-8').splitlines():
    try:
        q = json.loads(line)
        quality[q['path']] = q
    except json.JSONDecodeError:
        pass
for line in (ROOT/'preprocess_status_followup_dual/series_results.jsonl').read_text(encoding='utf-8').splitlines():
    try:
        r = json.loads(line)
        if r['resolution']==384:
            records[r['source']] = r
    except json.JSONDecodeError:
        pass
prepared = 0
for case_index, (sid, sources) in enumerate(sorted(manifest.items())):
    path = OUT/'tasks'/f'{case_index:04d}.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('input_version') == 2:
        continue
    source_paths = [str(Path('D:/OAI/followup_6250_three_sequences')/r['visit']/'image03'/r['image_file'].split('/',4)[4]) for r in sources]
    if not all(p in records for p in source_paths):
        continue
    identity_issues = []
    for meta,p in zip(sources,source_paths):
        q = quality.get(p,{})
        expected_date = datetime.datetime.strptime(meta['date'],'%m/%d/%Y').strftime('%Y%m%d')
        if q.get('patient_ids') != [meta['src_subject_id']] or q.get('study_dates') != [expected_date]:
            identity_issues.append(dict(archive=Path(p).name,reason='DICOM identity/date not confirmed against manifest'))
    if len({r['date'] for r in sources}) != 1:
        identity_issues.append(dict(reason='Multiple imaging dates within nominal visit; do not combine without adjudication'))
    first = sources[0]
    key = tuple(first[k] for k in ('src_subject_id','side','visit'))
    readings = scores.get(key, [])
    image = Image.new('RGB',(984,30+len(sources)*366),'#151515')
    draw = ImageDraw.Draw(image)
    draw.text((4,4),sid+' | same-visit follow-up | sampled cache views',fill='white')
    image_meta = []
    for slot,(meta, source_path) in enumerate(zip(sources,source_paths)):
        rec = records[source_path]
        y = 30+366*slot
        draw.text((4,y),meta['sequence']+' | '+Path(source_path).name,fill='white')
        available = rec['status']=='passed' and Path(rec['output']).exists()
        tile_info = dict(sequence=meta['sequence'],series_available=available,archive=Path(source_path).name,
                         manifest_date=meta['date'],geometry_status=rec['status'])
        if available:
            a = np.load(rec['output'],mmap_mode='r',allow_pickle=False)
            zs = np.rint(np.linspace(0,23,12)).astype(int).tolist()
            sampled = rec['sampled_indices'][:]
            if first['side']=='RIGHT' and meta['sequence'].startswith('SAG'):
                sampled = sampled[::-1]
            for j,z in enumerate(zs):
                x, yy = (j%6)*164, y+17+(j//6)*174
                image.paste(Image.fromarray(a[z]).resize((160,160),Image.Resampling.LANCZOS).convert('RGB'),(x,yy))
                draw.text((x+3,yy+160),f'cache{z+1}/raw{sampled[z]+1}',fill='yellow')
            tile_info.update(cache_slice_indices_1based=[z+1 for z in zs],
                original_retained_slice_indices_1based=[sampled[z]+1 for z in zs],
                orientation='Canonicalized: SAG medial-to-lateral; COR/AX image-right lateral; no unverified superior/anterior labels',
                coverage='12 of 24 cached slices; central80%; crop130mm resized384; not full diagnostic volume')
        else:
            draw.text((4,y+45),'UNAVAILABLE / GEOMETRY REVIEW: DO NOT INTERPRET BLANK AREA',fill='red')
            tile_info['limitation'] = rec.get('error','unavailable')
        image_meta.append(tile_info)
    img = OUT/'images'/f'{case_index:04d}_{sid}.jpg'
    image.save(img,quality=95)
    original = {}
    all_fields = set()
    comments = []
    for t in T:
        original[t] = []
        for line,r in readings:
            raw = {f:r.get(f,'') for f in fields[t]}
            all_fields.update(raw)
            original[t].append(dict(source_line=line,readprj=r.get('readprj'),version=r.get('version'),raw_scores=raw))
    for line,r in readings:
        comment = {f:r.get(f,'') for f in ('m_comments','m_technical') if r.get(f,'')}
        if comment:
            comments.append(dict(source_line=line,**comment))
    task = dict(case_index=case_index,knee_id=sid,subject_id=key[0],side=key[1],visit=key[2],image_visit=key[2],
        input_version=2,identity_audit_version=1,identity_timepoint_confirmed=not identity_issues,identity_issues=identity_issues,
        images=[dict(file=str(img.resolve()),sequences=image_meta)],original_readings=original,
        score_dictionary={f:dictionary.get(f,{'ElementName':f,'limitation':'Missing dictionary entry; do not invent meaning'}) for f in sorted(all_fields)},
        reader_comments=comments,target_definitions=definitions,
        scoring_cautions={'missing_scores':'Use only same-visit readings; do not borrow baseline or another follow-up',
            'cartilage':'Same working interpretation as baseline v3: code components encode area of any loss and full-thickness loss, NOT depth percentage. Local dictionary lacks full value notes; preserve this limitation.',
            'meniscus':'Same working interpretation as baseline v3: 0 normal, 1 intrasubstance signal; 2-5 tear morphologies; 6-8 maceration. Do not infer subtype if dictionary omits it.',
            'hoffa':'Same working interpretation as baseline v3: 0-3 increasing nonspecific signal abnormality; not a direct synovial-thickening measurement.',
            'mpr':'AX_MPR is derived DESS, not an independent acquisition or T1',
            'multiple_archives':'Separate archives are shown separately; do not treat them as independent reader agreement'})
    path.write_text(json.dumps(task,indent=2),encoding='utf-8')
    prepared += 1
summary = dict(candidate_exams=len(manifest),newly_prepared=prepared,total_tasks=len(list((OUT/'tasks').glob('*.json'))),
    scoring_started=bool(list((OUT/'results').glob('*.jsonl'))),
    result_files=len(list((OUT/'results').glob('*.jsonl'))),
    prompt=str(ROOT/'oai_vlm_soft_mapping_prompt_v3_seven_targets.txt'))
(OUT/'preparation_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
