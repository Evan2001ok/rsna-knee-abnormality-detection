"""Strictly merge report/VLM JSONL into a complete training CSV; Gold overrides by UID."""
import argparse,csv,json,math
from pathlib import Path
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def prob(x):
    if isinstance(x,dict):
        keys=[k for k in ('prob','grade','p') if k in x]
        if len(keys)!=1:raise ValueError('Expected one probability field')
        x=x[keys[0]]
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('Invalid probability')
    return x
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--train-csv',required=True);ap.add_argument('--results',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    with open(a.train_csv,encoding='utf-8-sig',newline='') as f:train=list(csv.DictReader(f))
    ids=[r['StudyInstanceUID'] for r in train]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate official UID')
    gold={r['StudyInstanceUID']:r for r in train if all(r[t].strip() for t in T)}
    expected=set(ids)-set(gold);scores={};evidence=[]
    paths=sorted(Path(a.results).glob('*.jsonl'))
    for p in paths:
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            if not line.strip():continue
            try:
                r=json.loads(line);uid=r.get('uid',r.get('StudyInstanceUID'))
                if uid not in expected:raise ValueError('Unexpected UID (Gold should not be rescored)')
                if uid in scores:raise ValueError('Duplicate result UID')
                scores[uid]={t:prob(r[t]) for t in T};evidence.append(r)
            except Exception as e:raise ValueError(f'{p.name}:{i}: {e}') from e
    missing=expected-set(scores)
    if missing:raise ValueError(f'Missing {len(missing)} non-Gold results; no output written')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['StudyInstanceUID',*T]);w.writeheader()
        for uid in ids:
            row={t:prob(float(gold[uid][t])) for t in T} if uid in gold else scores[uid]
            w.writerow({'StudyInstanceUID':uid,**row})
    with output.with_suffix('.evidence.jsonl').open('w',encoding='utf-8') as f:
        for row in evidence:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(ids),'gold_overridden':len(gold),'scored':len(scores),'output':str(output)}))
if __name__=='__main__':main()
