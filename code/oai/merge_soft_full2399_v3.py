"""Validate and merge independently recorded visual mapping shards."""
import csv, json, hashlib, math
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'soft_full2399_v3'
T=['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']
def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    index=json.loads((D/'index.json').read_text(encoding='utf-8'))
    byid={r['knee_id']:r for r in index}; collected={};errors=[];counts={}
    for p in sorted((D/'results').glob('agent*.jsonl')):
        n=0
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            try:
                r=json.loads(line);k=r['knee_id'];assert k in byid
                assert k not in collected, 'duplicate ID'
                assert r['visit']=='V00'
                assert set(r['targets'])==set(T)
                for t in T:
                    x=r['targets'][t];v=x['p'];m=x['mask']
                    assert m in [0,1] and (v is None)==(m==0)
                    assert v is None or isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1
                    assert x['finding'] and x['evidence_source'] in ['scores_and_images','image_only','insufficient']
                collected[k]=r;n+=1
            except Exception as e:errors.append({'file':p.name,'line':n+1,'error':str(e)})
        counts[p.name]=n
    status={'completed':len(collected),'total':2399,'per_agent':counts,'errors':errors,
            'missing_ids':[k for k in byid if k not in collected]}
    (D/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if len(collected)!=2399 or errors:
        print(json.dumps({k:v for k,v in status.items() if k!='missing_ids'}));return
    source=R/'labels_v1/labels.csv';h=hashlib.sha256(source.read_bytes()).hexdigest()
    assert h=='c26133460de48c2310310a83f9251f7a0e3cae7a36577ad795a26d188d4e184b'
    with source.open(encoding='utf-8-sig',newline='') as f:old={r['knee_id']:r for r in csv.DictReader(f)}
    rows=[];comp=[];preds=[]
    for c in index:
        k=c['knee_id'];p=collected[k]
        task=json.loads((D/'tasks'/f"{c['index']:04d}.json").read_text(encoding='utf-8'))
        assert task['knee_id']==k and task['visit']==task['image_visit']=='V00'
        assert all(Path(im['file']).is_file() for im in task['images'])
        p['case_index']=c['index'];p['scorer']='current Codex model and 12 delegated agents, in-session visual scoring'
        p['probabilities_calibrated']=False;p['prompt_version']='v3-seven-targets'
        p['definition_version']='user-provided working recipe, not independently verified official criteria'
        row={'knee_id':k,'visit':'V00','src_subject_id':task['subject_id']}
        for t in T:
            x=p['targets'][t];v=x['p'];raw=task['original_readings'][t]
            x['original_readings']=raw;x['image_refs']=task['images']
            x['raw_fields_used']=sorted({f for r in raw for f,a in r['raw_scores'].items() if a!=''})
            nonempty=[r['raw_scores'] for r in raw if any(a!='' for a in r['raw_scores'].values())]
            x['reader_conflict']=len({json.dumps(a,sort_keys=True) for a in nonempty})>1
            x['review_needed']=bool(x['review_needed'] or x['reader_conflict'])
            row[t]='' if v is None else v;row[t+'__mask']=x['mask']
            o=old[k];m=int(o[t+'__mask']);oldv=float(o[t]) if m else None
            comp.append({'knee_id':k,'target':t,'old_label':'' if oldv is None else oldv,'old_mask':m,
                         'new_p':'' if v is None else v,'new_mask':x['mask'],
                         'newly_evaluable':int(not m and v is not None),
                         'binary_disagreement_at_0_5':int(m and v is not None and (v>=.5)!=(oldv>=.5))})
        rows.append(row);preds.append(p)
    writecsv(D/'OAI_soft_labels_2399_v3.csv',rows);writecsv(D/'review_comparison.csv',comp)
    (D/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in preds),encoding='utf-8')
    stats=[]
    for t in T:
        r=[x for x in comp if x['target']==t];ps=[x['new_p'] for x in r if x['new_mask']]
        s={'target':t,'old_0':sum(x['old_mask'] and x['old_label']==0 for x in r),
           'old_1':sum(x['old_mask'] and x['old_label']==1 for x in r),'old_mask':sum(not x['old_mask'] for x in r),
           'new_lt_0_1':sum(v<.1 for v in ps),'new_0_1_to_0_3':sum(.1<=v<.3 for v in ps),
           'new_0_3_to_0_5':sum(.3<=v<.5 for v in ps),'new_ge_0_5':sum(v>=.5 for v in ps),
           'new_mask':2399-len(ps),'newly_evaluable':sum(x['newly_evaluable'] for x in r),
           'binary_disagreements_at_0_5':sum(x['binary_disagreement_at_0_5'] for x in r)}
        stats.append(s)
    writecsv(D/'distribution_comparison.csv',stats)
    summary={'cases':2399,'agents':12,'structural_validation_passed':True,'per_target':stats,
             'original_labels_sha256':h,'original_labels_modified':False,
             'total_scored_cells':sum(2399-x['new_mask'] for x in stats),
             'limitations':['Uncalibrated model estimates, no expert-reference accuracy evaluation.',
                            'Sampled cached montage, no full-volume lesion extent measurement.',
                            'Existing pilot results reused when same ID, explicitly recorded in shard metadata.',
                            'Seven targets scored; training target selection remains separate.']}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==h
    with (D/'OAI_soft_labels_2399_v3.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    assert len(rr)==2399 and len({x['knee_id'] for x in rr})==2399
    for r in rr:
        for t in T:assert (r[t]=='')==(r[t+'__mask']=='0')
    (D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
