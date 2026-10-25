"""Prepare validated JSON rows for final machine CSV export, only when all cases exist."""
import csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'soft_followup6250_v3'
TARGETS=['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']

def main():
    expected=sorted((OUT/'tasks').glob('*.json'))
    missing=[p.stem for p in expected if not (OUT/'results'/f'{p.stem}.jsonl').exists()]
    if missing:
        report=dict(ready=False,expected=len(expected),missing_count=len(missing),missing=missing)
        (OUT/'final_merge_status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='missing'}))
        return
    subprocess.run([sys.executable,str(ROOT/'validate_followup_soft_results.py')],check=True,cwd=ROOT.parent)
    results={}
    for p in expected:
        r=json.loads((OUT/'results'/f'{p.stem}.jsonl').read_text(encoding='utf-8'))
        if r['knee_id'] in results:
            raise ValueError('Duplicate result ID')
        results[r['knee_id']]=r
    cache_index={}
    for resolution in (224,384):
        directory=Path(f'D:/OAI/kaggle_upload_followup6250_{resolution}')
        rows=list(csv.DictReader((directory/'index.csv').open(encoding='utf-8',newline='')))
        if len(rows)!=6250 or len({r['knee_id'] for r in rows})!=6250 or set(results)!=set(r['knee_id'] for r in rows):
            raise ValueError('Cache/label identity set mismatch')
        cache_index[resolution]=rows
    if cache_index[224]!=cache_index[384]:
        raise ValueError('224/384 index mismatch')
    headers=['knee_id','src_subject_id','side','visit','part','row',*TARGETS,*[t+'__mask' for t in TARGETS]]
    merged=[]
    evidence=[]
    for row in cache_index[224]:
        r=results[row['knee_id']]
        if r['visit']!=row['visit']:
            raise ValueError('Visit mismatch')
        data={k:row[k] for k in headers[:6]}
        data['part'],data['row']=int(data['part']),int(data['row'])
        for target in TARGETS:
            data[target]=r['targets'][target]['p']
            data[target+'__mask']=r['targets'][target]['mask']
        merged.append([data[k] for k in headers])
        evidence.append(r)
    payload=dict(headers=headers,rows=merged,checks=6250,join='knee_id plus visit',
                 prompt_version='oai-soft-mapping-v3-seven-targets')
    (OUT/'final_label_rows.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    with (OUT/'OAI_soft_followup6250_v3.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(merged)
    # Keep original per-case evidence intact; this is a separate ordered copy.
    with (OUT/'OAI_soft_followup6250_v3_evidence.jsonl').open('w',encoding='utf-8') as f:
        for r in evidence:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report=dict(ready=True,checks=len(merged),valid_scores=sum(v['mask'] for r in evidence for v in r['targets'].values()),
                abstentions=sum(v['mask']==0 for r in evidence for v in r['targets'].values()),
                csv_export_pending=False)
    (OUT/'final_merge_status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':
    main()
