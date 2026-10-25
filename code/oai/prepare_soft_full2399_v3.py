import csv,json,sys,random,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'python_deps'))
from PIL import Image,ImageDraw
OUT=ROOT/'soft_full2399_v3'; OUT.mkdir(exist_ok=True)
(OUT/'images').mkdir(exist_ok=True); (OUT/'tasks').mkdir(exist_ok=True)
SOURCE=Path('D:/OAI/kaggle_upload_384')
T=['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']
records=[json.loads(l) for l in (ROOT/'labels_v1/provenance.jsonl').read_text().splitlines()]
dictionary={r['ElementName']:r for r in csv.DictReader((ROOT/'score_dictionary.csv').open(encoding='utf-8-sig',newline=''))}
arrays=[];masks=[];lookup={}
manifest=json.loads((SOURCE/'manifest.json').read_text())
for p,n in enumerate(manifest['parts']):
    arrays.append(np.load(SOURCE/f'part{p}_cache.npy',mmap_mode='r'))
    masks.append(np.load(SOURCE/f'part{p}_series_mask.npy'))
    for row,k in enumerate(np.load(SOURCE/f'part{p}_knee_ids.npy').astype(str)):lookup[k]=(p,row)
ledger={}
for line in (ROOT/'preprocess_status_384/series_results.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['status']=='passed':ledger[(r['subject'],r['side'],r['sequence'],Path(r['source']).name)]=r
records.sort(key=lambda r:r['knee_id'])
selected=records
assert len(selected)==2399
start=int(sys.argv[1]);end=int(sys.argv[2])
selected=records[start:end]
audit=[]
definitions={
 'definition_version':'user-supplied-imgrep-recipe-not-independently-reverified',
 'source':'User attachment a7956007-b268-4699-b5f6-43d614935aa0; working definitions, not a verified official citation',
 'Medial Meniscus':'Medial meniscal surface-reaching tear on >=2 original slices OR qualifying abnormal morphology; intrasubstance degeneration negative; assess postoperative ambiguity.',
 'Medial OA':'Medial tibiofemoral high-grade cartilage loss >50% thickness over roughly >=1 cm; borderline negative.',
 'Effusion':'Joint effusion beyond normal physiologic fluid; use supplied working definition, no independently verified quantitative cutoff; effusion-synovitis composite is not pure effusion.',
 'Lateral Meniscus':'Surface-reaching tear on >=2 original slices OR qualifying abnormal morphology; intrasubstance degeneration negative; assess postoperative ambiguity.',
 'Lateral OA':'Lateral tibiofemoral high-grade cartilage loss >50% thickness over roughly >=1 cm; borderline negative.',
 'PF OA':'Patellar/trochlear high-grade cartilage loss >50% thickness over roughly >=1 cm; borderline negative.',
 'Synovitis':'Synovial thickening/inflammation; effusion alone insufficient; Hoffa signal is nonspecific proxy.'}
cases=[]
for i,r in enumerate(selected,start):
    k=r['knee_id'];p,row=lookup[k]
    image=Image.new('RGB',(984,1868),'#151515');draw=ImageDraw.Draw(image)
    draw.text((4,4),k+' | V00 | five sequences | sampled cache tiles; not a full diagnostic volume',fill='white')
    metadata=[]
    for s,seq in enumerate(manifest['slot_order']):
        y=26+s*366;draw.text((4,y),seq+' | slice order: see task JSON',fill='white')
        zs=np.rint(np.linspace(0,23,12)).astype(int).tolist()
        rec=ledger.get((r['src_subject_id'],r['side'],seq,r['selected_archives'].get(seq,'')))
        available=bool(masks[p][row,s]) and rec is not None
        if not available:
            draw.text((4,y+50),'MISSING SEQUENCE - do not interpret zeros',fill='red')
            metadata.append({'sequence':seq,'series_available':False})
            continue
        sampled=rec['sampled_indices'][:]
        if r['side']=='RIGHT' and seq.startswith('SAG'):sampled=sampled[::-1]
        for j,z in enumerate(zs):
            x=(j%6)*164; yy=y+17+(j//6)*174
            image.paste(Image.fromarray(arrays[p][row,s,z]).resize((160,160),Image.Resampling.LANCZOS).convert('RGB'),(x,yy))
            draw.text((x+3,yy+160),f'cache{z+1} / raw{sampled[z]+1}',fill='yellow')
        metadata.append({'sequence':seq,'series_available':True,'cache_slice_indices_1based':[z+1 for z in zs],
                         'original_retained_slice_indices_1based':[sampled[z]+1 for z in zs],
                         'source_slices':rec['source_slices'],'coverage':'12 of 24 cached slices; central80 only',
                         'orientation':'Canonicalized: SAG order medial->lateral; COR/AX image-right lateral. In-plane superior/anterior labels not verified here.',
                         'spatial_scale':'130mm intended crop resized to384; display160px; do not measure lesion length from this montage alone'})
    path=OUT/'images'/f'{i:02d}_{k}.jpg';image.save(path,quality=95)
    readings={};fields=set()
    for t in T:
        readings[t]=[{kk:vv for kk,vv in rr.items() if kk in ['source_line','readprj','version','raw_scores']} for rr in r['targets'][t]['readings']]
        for rr in readings[t]:fields.update(rr['raw_scores'])
    case={'case_index':i,'knee_id':k,'subject_id':r['src_subject_id'],'side':r['side'],'visit':'V00','image_visit':'V00',
          'images':[{'file':str(path.resolve()),'sequences':metadata}], 'original_readings':readings,
          'score_dictionary':{f:dictionary[f] for f in sorted(fields)},
          'scoring_cautions':{'cartilage':'Code components encode area of any loss and full-thickness loss, NOT depth percentage. Local dictionary lacks full value notes; preserve this limitation.',
                              'meniscus':'0 normal, 1 intrasubstance signal; 2-5 tear morphologies; 6-8 maceration. Do not infer subtype if dictionary omits it.',
                              'hoffa':'0-3 increasing nonspecific signal abnormality; not a direct synovial-thickening measurement'},
          'reader_comments':r.get('surgery_review',[]),'target_definitions':definitions}
    cases.append(case)
for c in cases:
    (OUT/'tasks'/f"{c['case_index']:04d}.json").write_text(json.dumps(c,indent=2),encoding='utf-8')
print(json.dumps({'prepared_range':[start,end],'cases':len(cases)}))
