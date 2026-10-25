"""Validate evidence-backed v3 results; no score generation or imputation."""
import json,math
from pathlib import Path
ROOT = Path(__file__).resolve().parent/'soft_followup6250_v3'
TARGETS = ['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']
required = {'p','mask','evidence_source','raw_fields_used','image_refs','finding',
            'definition_alignment','limitation','reader_conflict','review_needed'}
results,errors = {},[]
for path in sorted((ROOT/'results').glob('*.jsonl')):
    for line_number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        try:
            r = json.loads(line)
            task_path = ROOT/'tasks'/f'{int(path.stem):04d}.json'
            task = json.loads(task_path.read_text(encoding='utf-8'))
            assert (r['knee_id'],r['visit']) == (task['knee_id'],task['visit']), 'ID/visit mismatch'
            assert r['prompt_version']=='oai-soft-mapping-v3-seven-targets', 'Wrong prompt version'
            assert r['definition_version']==task['target_definitions']['definition_version'], 'Definition mismatch'
            assert set(r['targets'])==set(TARGETS), 'Expected seven targets'
            key = r['knee_id'],r['visit']
            assert key not in results, 'Duplicate result'
            allowed_fields = set(task['score_dictionary'])
            for target,v in r['targets'].items():
                assert required.issubset(v), 'Missing output fields'
                assert v['mask'] in (0,1), 'Invalid mask'
                assert v['evidence_source'] in ('scores_and_images','scores_only','image_only','insufficient')
                used_fields = set()
                for item in v['raw_fields_used']:
                    if isinstance(item,str):
                        used_fields.add(item)
                    elif isinstance(item,dict) and 'raw_scores' in item:
                        used_fields.update(item['raw_scores'])
                        assert item in task['original_readings'][target], 'Unprovided reader/score evidence'
                    elif isinstance(item,dict) and 'field' in item and 'value' in item:
                        used_fields.add(item['field'])
                        assert any(rr.get('source_line')==item.get('source_line')
                            and rr.get('readprj')==item.get('readprj')
                            and rr.get('version')==item.get('version')
                            and rr['raw_scores'].get(item['field'])==item['value']
                            for rr in task['original_readings'][target]), 'Unprovided field/value evidence'
                    else:
                        raise AssertionError('Unsupported raw-field reference format')
                assert used_fields.issubset(allowed_fields), 'Unprovided raw field'
                target_fields = {f for rr in task['original_readings'][target] for f in rr['raw_scores']}
                assert used_fields.issubset(target_fields), 'Raw evidence belongs to a different target'
                provided_images = {image['file']: image for image in task['images']}
                for ref in v['image_refs']:
                    if isinstance(ref,str):
                        assert any(Path(name).name in ref for name in provided_images), 'Unprovided image reference'
                    elif isinstance(ref,dict):
                        assert ref.get('file') in provided_images, 'Unprovided image file'
                        provided = provided_images[ref['file']]['sequences']
                        sequences = {meta['sequence'] for meta in provided}
                        if 'sequence' in ref:
                            assert ref['sequence'] in sequences, 'Unprovided sequence reference'
                        if 'sequences' in ref:
                            assert set(ref['sequences']).issubset(sequences), 'Unprovided sequence list'
                        if 'archive' in ref:
                            assert any(meta.get('archive')==ref['archive'] and meta['sequence']==ref.get('sequence') for meta in provided), 'Unprovided archive reference'
                    else:
                        raise AssertionError('Invalid image reference type')
                if v['mask']==1:
                    assert not isinstance(v['p'],bool) and isinstance(v['p'],(int,float)) and math.isfinite(v['p']) and 0<=v['p']<=1
                    assert task['identity_timepoint_confirmed'], 'Unresolved identity cannot receive valid target'
                    assert v['evidence_source']!='insufficient'
                    if v['evidence_source'] in ('scores_and_images','image_only'):
                        assert v['image_refs'], 'Image evidence needs actual reference'
                else:
                    assert v['p'] is None
                assert isinstance(v['reader_conflict'],bool) and isinstance(v['review_needed'],bool)
            results[key]=r
        except Exception as exc:
            errors.append(dict(file=str(path),line=line_number,error=str(exc)))
report = dict(task_count=len(list((ROOT/'tasks').glob('*.json'))),completed_cases=len(results),
    valid_scores=sum(v['mask']==1 for r in results.values() for v in r['targets'].values()),
    abstentions=sum(v['mask']==0 for r in results.values() for v in r['targets'].values()),errors=errors,
    scope='Schema/identity consistency only, not clinical accuracy or calibrated probabilities')
(ROOT/'result_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
if errors:
    raise SystemExit(1)
