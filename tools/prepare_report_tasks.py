"""Prepare non-Gold report tasks; no model calls and no access to images or old labels."""
import argparse,csv,json
from pathlib import Path
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=10);a=p.parse_args()
    if a.batch_size<1:p.error('batch-size must be positive')
    with open(a.train_csv,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ids=[r['StudyInstanceUID'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate UID')
    selected=[{'uid':r['StudyInstanceUID'],'report':r['Report']} for r in rows if not all(r[t].strip() for t in T)]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('task_*.json')):raise ValueError('Use an empty task output directory')
    for i in range(0,len(selected),a.batch_size):(out/f'task_{i//a.batch_size:04d}.json').write_text(json.dumps(selected[i:i+a.batch_size],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'non_gold':len(selected),'tasks':(len(selected)+a.batch_size-1)//a.batch_size}))
if __name__=='__main__':main()
