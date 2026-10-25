"""Read-only score audit. MRI matching and surgical adjudication remain pending."""
import ast, csv, json, re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from audit_scores import read_nda

root = Path(__file__).resolve().parent
tree = ast.parse((root / 'generate_labels_v1.py').read_text())
selected = [n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('RULES','OA_ALLOWED','OA_POSITIVE') for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name == 'vote')]
exec(compile(ast.Module(body=selected, type_ignores=[]), '<existing rules>', 'exec'))
targets = ['Lateral Meniscus', 'Lateral OA', 'PF OA', 'Synovitis']
groups = defaultdict(list)
for r in read_nda(root / 'scores/oai_kmrisemiquant01.txt'):
    if r['side'] in ('1', '2'):
        groups[(r['src_subject_id'], r['side'], r['visit'])].append(r)
basepairs = set()
with (root / 'labels_v1/labels.csv').open(encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        basepairs.add((r['src_subject_id'], {'RIGHT':'1','LEFT':'2'}[r['side']]))
result = {}
statuses = {}
for key, rows in groups.items():
    ss = {}
    for t in targets:
        fields, mode, _ = RULES[t]
        readings = [vote(r, fields, mode) for r in rows]
        votes = {v for v, _ in readings if v is not None}
        reasons = {why for _, why in readings}
        if 'invalid_code' in reasons: s = 'invalid_code'
        elif len(votes) > 1: s = 'conflict'
        elif votes: s = 'positive' if next(iter(votes)) else 'negative'
        else: s = 'ambiguous_grade' if 'ambiguous_grade' in reasons else 'missing'
        ss[t] = s
    statuses[key] = ss
for visit in sorted({k[2] for k in groups}):
    keys = [k for k in groups if k[2] == visit]
    eligible = [k for k in keys if any(s in ('positive','negative') for s in statuses[k].values())]
    result[visit] = {'scored_exams':len(keys), 'any_four_eligible_before_surgery_and_MRI':len(eligible),
        'eligible_from_existing_baseline_pairs':sum(k[:2] in basepairs for k in eligible),
        'subjects':len({k[0] for k in eligible}),
        'targets':{t:dict(Counter(statuses[k][t] for k in keys)) for t in targets}}
follow = [k for k in statuses if k[2] != 'V00' and any(s in ('positive','negative') for s in statuses[k].values())]
base_subjects = {k[0] for k in basepairs}
strata = {}
for name, selected_keys in {
    'new_subjects': [k for k in follow if k[0] not in base_subjects],
    'existing_subject_new_knee': [k for k in follow if k[0] in base_subjects and k[:2] not in basepairs],
    'existing_baseline_knee_followup': [k for k in follow if k[:2] in basepairs],
}.items():
    strata[name] = {'exams':len(selected_keys), 'subjects':len({k[0] for k in selected_keys}),
        'targets':{t:dict(Counter(statuses[k][t] for k in selected_keys)) for t in targets}}
changes = {}
for t in targets:
    paired = [(statuses[(k[0],k[1],'V00')][t], statuses[k][t]) for k in follow if k[:2] in basepairs and (k[0],k[1],'V00') in statuses]
    changes[t] = dict(Counter(a+' -> '+b for a,b in paired if a in ('positive','negative') and b in ('positive','negative')))
report = {'by_visit':result, 'followup_eligible_before_surgery_and_MRI':len(follow),
    'followup_strata':strata,
    'followup_eligible_from_existing_baseline_pairs':sum(k[:2] in basepairs for k in follow),
    'followup_eligible_new_subjects':len({k[0] for k in follow} - {k[0] for k in basepairs}),
    'followup_targets':{t:dict(Counter(statuses[k][t] for k in statuses if k[2]!='V00')) for t in targets},
    'paired_baseline_followup_label_changes':changes,
    'limitations':['Before MRI availability matching; not a download-ready cohort.', 'Uses existing conservative MOAKS rules, not revised thresholds.', 'Meniscal surgical comments need visit-aware review; baseline-specific surgery rule not reused on followups.', 'Repeated visits are correlated, not independent new participants.']}
(root/'score_audit/followup_four_target_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
