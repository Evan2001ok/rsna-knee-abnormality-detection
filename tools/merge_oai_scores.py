"""Merge seven-target v3 results without reading legacy hard labels or predictions."""
import argparse,csv,json,math
from pathlib import Path
T=['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True,help='Cache knee_index.json or index.csv');p.add_argument('--results',required=True);p.add_argument('--tasks',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    idx=Path(a.index)
    if idx.suffix=='.json':rows=json.loads(idx.read_text(encoding='utf-8'))
    else:
        with idx.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ids=[r['knee_id'] for r in rows]
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Empty/duplicate cache index')
    tasks={}
    for f in sorted(Path(a.tasks).glob('*.json')):
        task=json.loads(f.read_text(encoding='utf-8'));k=task['knee_id']
        if k in tasks:raise ValueError('Duplicate task ID')
        tasks[k]=task
    if set(tasks)!=set(ids):raise ValueError('Task/cache ID set differs')
    for row in rows:
        task=tasks[row['knee_id']]
        if task['visit']!=row['visit'] or task.get('side',row['side'])!=row['side'] or str(task.get('subject_id',row['src_subject_id']))!=str(row['src_subject_id']):raise ValueError('Cache/task identity or visit mismatch')
    results={}
    for f in sorted(Path(a.results).glob('*.jsonl')):
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            r=json.loads(line);k=r['knee_id']
            if k not in tasks or k in results:raise ValueError('Unknown/duplicate result ID')
            if r.get('prompt_version')!='oai-soft-mapping-v3-seven-targets' or not r.get('definition_version'):raise ValueError('Missing v3 prompt/definition provenance')
            if r['visit']!=tasks[k]['visit'] or set(r['targets'])!=set(T):raise ValueError('Visit or target mismatch')
            for target in T:
                v=r['targets'][target];m=v['mask'];score=v['p']
                required={'p','mask','evidence_source','raw_fields_used','image_refs','finding','definition_alignment','limitation','reader_conflict','review_needed'}
                if not required.issubset(v) or not isinstance(v['raw_fields_used'],list) or not isinstance(v['image_refs'],list) or type(v['reader_conflict']) is not bool or type(v['review_needed']) is not bool:raise ValueError('Incomplete v3 target schema')
                if type(m) is not int or m not in (0,1) or (score is None)!=(m==0):raise ValueError('Invalid p/mask')
                if m and (isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1):raise ValueError('Invalid score')
                if v['evidence_source'] not in ('scores_and_images','scores_only','image_only','insufficient') or not v['finding']:raise ValueError('Missing evidence')
                if not v['definition_alignment'] or (m==0 and not v['limitation']) or (m==1 and v['evidence_source']=='insufficient'):raise ValueError('Missing definition/abstention explanation')
                if not tasks[k].get('identity_timepoint_confirmed',True) and m:raise ValueError('Identity mismatch requires abstention')
            results[k]=r
    if set(results)!=set(ids):raise ValueError(f'Missing {len(set(ids)-set(results))} results; no output written')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    fields=['knee_id','src_subject_id','side','visit','part','row',*T,*[t+'__mask' for t in T]]
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            r=results[row['knee_id']]
            if row['visit']!=r['visit']:raise ValueError('Cache/result visit mismatch')
            value={k:row[k] for k in fields[:6]}
            for t in T:value[t]=r['targets'][t]['p'];value[t+'__mask']=r['targets'][t]['mask']
            w.writerow(value)
    out.with_suffix('.evidence.jsonl').write_text(''.join(json.dumps(results[k],ensure_ascii=False)+'\n' for k in ids),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'output':str(out)}))
if __name__=='__main__':main()
