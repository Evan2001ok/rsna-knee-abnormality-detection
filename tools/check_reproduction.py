"""Offline source/dependency/entry-point checks and synthetic label-pipeline tests.

Uses only the Python standard library. Never imports models or reads real MRI/labels.
"""
import argparse
import ast
import csv
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import shutil

from run_pipeline import PathsOnly, load_mapping, transformed, reject_mixed_label_inputs

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
OAI = ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report')
    args = ap.parse_args()
    checks, errors, modules = [], [], set()

    def check(value, label):
        (checks if value else errors).append(label)

    def run(arguments, success=True):
        p = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                           text=True, encoding='utf-8', capture_output=True)
        if success and p.returncode:
            raise AssertionError(p.stderr or p.stdout)
        if not success and not p.returncode:
            raise AssertionError('Expected invalid input to be rejected')
        return p

    trees = []
    for folder in ['code', 'tools']:
        for p in sorted((ROOT / folder).rglob('*.py')):
            tree = ast.parse(p.read_text(encoding='utf-8-sig'))
            trees.append((p.relative_to(ROOT).as_posix(), tree))
    embedded = 0
    for label, tree in trees[:]:
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                try:
                    value = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    continue
                candidates = [value] if isinstance(value, str) else list(value.values()) if isinstance(value, dict) else []
                for source in candidates:
                    if isinstance(source, str) and len(source) > 1000 and source.lstrip().startswith(('from ', 'import ')):
                        trees.append((label + '::embedded', ast.parse(source)))
                        embedded += 1
    local_modules = {p.stem for d in ['code', 'tools'] for p in (ROOT / d).rglob('*.py')}
    for label, tree in trees:
        for n in ast.walk(tree):
            imports = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            modules.update(s.split('.')[0] for s in imports)
    third_party = modules - sys.stdlib_module_names - local_modules - {'__future__'}
    distributions = {'PIL': 'Pillow', 'cv2': 'opencv-python', 'sklearn': 'scikit-learn', 'NDATools': 'nda-tools'}
    declared = set()
    for p in (ROOT / 'requirements').glob('*.txt'):
        declared.update(re.split(r'[<>=!~\[;]', s.strip())[0].lower().replace('_', '-')
                        for s in p.read_text().splitlines() if s.strip() and not s.startswith('#'))
    # Installed separately as a matching CUDA pair; explicitly documented in ENVIRONMENT.md.
    declared.update({'torch', 'torchvision'})
    for module in sorted(third_party):
        distribution = distributions.get(module, module).lower().replace('_', '-')
        check(distribution in declared, 'Dependency declared: ' + module)

    prompts = ['code/labeling/rsna/report_v1_prompt.txt', 'code/labeling/rsna/report_v2_prompt.txt',
               'code/labeling/rsna/vlm_image_report_prompt.txt', 'code/oai/oai_vlm_soft_mapping_prompt_v3_seven_targets.txt']
    for name in prompts:
        check((ROOT / name).is_file() and (ROOT / name).stat().st_size > 100, 'Operational prompt: ' + name)
    for p in ROOT.glob('*.md'):
        for href in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if not href.startswith(('http:', 'https:', '#')):
                check((p.parent / href.split('#')[0]).exists(), 'Documentation link: ' + href)

    mapping = load_mapping(ROOT / 'config/paths.example.json')
    for p in sorted((ROOT / 'code/train').glob('*.py')):
        result = run(['tools/run_pipeline.py', '--script', p.relative_to(ROOT), '--paths',
                      'config/paths.example.json', '--folds', '0', '--plan'])
        plan = json.loads(result.stdout)
        check(plan['execution'] is False and plan['folds'] == [0] and plan['path_changes'], 'Training dry-run: ' + p.name)
    for name, count in [('cache224.py', 2), ('cache384.py', 4)]:
        for part in range(count):
            result = run(['tools/run_pipeline.py', '--script', 'code/rsna/' + name, '--paths',
                          'config/paths.example.json', '--part', part, '--plan'])
            check(json.loads(result.stdout)['part'] == part, 'Cache dry-run: ' + name + '/' + str(part))
    for name in ['ensemble6.py', 'ensemble8.py']:
        run(['tools/run_pipeline.py', '--script', 'code/inference/' + name, '--paths', 'config/paths.example.json', '--plan'])
        check(True, 'Inference dry-run: ' + name)

    samples = ["p=f'D:/OAI/preprocessed_followup_{resolution}_series'",
               "p=f'D:/OAI/kaggle_upload_followup6250_{resolution}'"]
    for src in samples:
        tree = ast.fix_missing_locations(PathsOnly(mapping).visit(ast.parse(src)))
        namespace = {'resolution': 384}
        exec(compile(tree, '<synthetic-path>', 'exec'), namespace)
        check(namespace['p'].startswith(str(ROOT).replace('\\', '/') + '/data/oai/'), 'Dynamic path mapped: ' + src)
    for name in ['watch_followup_dual_cache.py', 'pack_followup_caches.py', 'prepare_final_followup_label_rows.py']:
        tree, _ = transformed(ROOT / 'code/oai' / name, mapping)
        leading = [n.values[0].value for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
                   and n.values and isinstance(n.values[0], ast.Constant) and isinstance(n.values[0].value, str)]
        check(not any(s.startswith('D:/OAI') for s in leading), 'Real dynamic OAI paths mapped: ' + name)

    scratch = ROOT / 'work'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rsna-repro-', dir=scratch) as tmp:
        tmp = Path(tmp)
        official = tmp / 'train.csv'
        with official.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['StudyInstanceUID', 'Report', *TARGETS])
            w.writeheader()
            w.writerow({'StudyInstanceUID': 'synthetic-report', 'Report': 'Synthetic fixture only.'})
            w.writerow({'StudyInstanceUID': 'synthetic-gold', 'Report': 'Excluded from scoring.', **{t: 1 for t in TARGETS}})
        run(['tools/prepare_report_tasks.py', '--train-csv', official, '--output', tmp / 'tasks'])
        tasks = json.loads(next((tmp / 'tasks').glob('*.json')).read_text())
        check([r['uid'] for r in tasks] == ['synthetic-report'], 'Report tasks exclude Gold')
        scores = tmp / 'scores'; scores.mkdir()
        row = {'uid': 'synthetic-report', **{t: {'grade': 0.25, 'stated': True, 'evidence': 'Fixture'} for t in TARGETS}}
        (scores / 'results.jsonl').write_text(json.dumps(row) + '\n')
        merged = tmp / 'rsna.csv'
        run(['tools/merge_rsna_scores.py', '--train-csv', official, '--results', scores, '--output', merged])
        values = list(csv.DictReader(merged.open(encoding='utf-8-sig')))
        check(len(values) == 2 and float(values[0]['ACL']) == 0.25 and float(values[1]['ACL']) == 1, 'RSNA strict merge and Gold override')
        row['ACL']['grade'] = 2
        (scores / 'results.jsonl').write_text(json.dumps(row) + '\n')
        run(['tools/merge_rsna_scores.py', '--train-csv', official, '--results', scores, '--output', tmp / 'invalid.csv'], success=False)
        check(not (tmp / 'invalid.csv').exists(), 'RSNA invalid probability rejected without output')

        case = dict(knee_id='synthetic_LEFT_V01', src_subject_id='synthetic', side='LEFT', visit='V01', part=0, row=0)
        index = tmp / 'index.json'; index.write_text(json.dumps([case]))
        tasks = tmp / 'oai_tasks'; tasks.mkdir()
        (tasks / '0.json').write_text(json.dumps({**case, 'subject_id': 'synthetic', 'identity_timepoint_confirmed': True}))
        result = dict(knee_id=case['knee_id'], visit='V01', prompt_version='oai-soft-mapping-v3-seven-targets', definition_version='synthetic-test-only',
                      targets={t: dict(p=0.3, mask=1, evidence_source='scores_and_images', raw_fields_used=[], image_refs=[], finding='Synthetic fixture', definition_alignment='Synthetic fixture', limitation='', reader_conflict=False, review_needed=False) for t in OAI})
        scores = tmp / 'oai_scores'; scores.mkdir()
        result['targets']['Synovitis'].update(p=None, mask=0, evidence_source='insufficient', limitation='Missing fixture evidence')
        (scores / '0.jsonl').write_text(json.dumps(result) + '\n')
        out = tmp / 'oai.csv'
        run(['tools/merge_oai_scores.py', '--index', index, '--tasks', tasks, '--results', scores, '--output', out])
        row = next(csv.DictReader(out.open(encoding='utf-8-sig')))
        check(row['Synovitis'] == '' and row['Synovitis__mask'] == '0' and row['visit'] == 'V01', 'OAI missing stays masked, identity preserved')
        result['visit'] = 'V00'
        (scores / '0.jsonl').write_text(json.dumps(result) + '\n')
        run(['tools/merge_oai_scores.py', '--index', index, '--tasks', tasks, '--results', scores, '--output', tmp / 'invalid_oai.csv'], success=False)
        check(not (tmp / 'invalid_oai.csv').exists(), 'OAI mismatched visit rejected without output')

        candidates=tmp/'candidates.csv'
        fields=['src_subject_id','side','visit','sequence','image_file','bytes','date']
        entry=dict(src_subject_id='synthetic',side='LEFT',visit='V01',sequence='SAG_IW_TSE',image_file='synthetic-a.tar.gz',bytes=123,date='01/01/2000')
        with candidates.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            w.writerows([entry,entry,dict(entry,image_file='synthetic-b.tar.gz',date='01/02/2000'),dict(entry,sequence='SAG_3D_DESS',image_file='synthetic-other.tar.gz')])
        queue=tmp/'queue'
        cmd=['tools/prepare_oai_download_manifest.py','--candidates',candidates,'--output',queue,'--expected-exams','1','--expected-archives','2']
        run(cmd)
        actual=list(csv.DictReader((queue/'download_manifest_three_sequences.csv').open(encoding='utf-8-sig')))
        check(len(actual)==2 and len({r['date'] for r in actual})==2,'OAI queue removes exact records but retains distinct/date candidates')
        before=(queue/'download_manifest_three_sequences.csv').read_bytes()
        run(cmd,success=False)
        check((queue/'download_manifest_three_sequences.csv').read_bytes()==before,'OAI queue preserves existing snapshot')

        mini=tmp/'workspace'
        (mini/'tools').mkdir(parents=True);(mini/'config').mkdir()
        shutil.copyfile(ROOT/'tools/init_workspace.py',mini/'tools/init_workspace.py')
        shutil.copyfile(ROOT/'config/paths.example.json',mini/'config/paths.example.json')
        run([mini/'tools/init_workspace.py','--layout','local'])
        before=(mini/'config/paths.json').read_bytes()
        check((mini/'data/oai').is_dir() and (mini/'labels/rsna').is_dir(),'First-time workspace creates required parent directories')
        run([mini/'tools/init_workspace.py','--layout','local'],success=False)
        check((mini/'config/paths.json').read_bytes()==before,'First-time workspace refuses to overwrite config')

        kaggle=tmp/'kaggle-workspace'
        (kaggle/'tools').mkdir(parents=True);(kaggle/'config').mkdir()
        shutil.copyfile(ROOT/'tools/init_workspace.py',kaggle/'tools/init_workspace.py')
        shutil.copyfile(ROOT/'config/paths.example.json',kaggle/'config/paths.example.json')
        run([kaggle/'tools/init_workspace.py','--layout','kaggle'])
        configured=json.loads((kaggle/'config/paths.json').read_text())['prefix_map']
        check(len(configured)==4 and all(v.startswith('${PACKAGE}/labels/rsna/') for v in configured.values()),
              'Kaggle initialization maps all four public RSNA source labels only')

        raw=tmp/'raw.csv';raw.write_text('StudyInstanceUID,ACL\nfixture,0.2\n')
        raw2=tmp/'raw2.csv';shutil.copyfile(raw,raw2)
        mixed=tmp/'renamed-targets.csv';mixed.write_text('StudyInstanceUID,ACL,label_source\nfixture,0.2,vlm07_llm03\n')
        def source_tree(first,second,third):
            return ast.parse('class Cfg:\n'+''.join(f'    {name}: object = Path({str(value)!r})\n'
                for name,value in zip(['scored_path','v1_path','v2_path'],[first,second,third])))
        for tree,label in [(source_tree(mixed,raw,raw2),'Renamed mixed-target CSV is rejected before training'),
                           (source_tree(raw,raw,raw2),'Duplicate source file mapping is rejected before training')]:
            rejected=False
            try:reject_mixed_label_inputs(tree)
            except ValueError:rejected=True
            check(rejected,label)

        pack_tree=ast.parse((ROOT/'tools/refresh_bundle.py').read_text())
        eligible=next(n for n in pack_tree.body if isinstance(n,ast.FunctionDef) and n.name=='eligible')
        sandbox={'ROOT':mini}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[eligible],type_ignores=[])),'<packaging-allowlist>','exec'),sandbox)
        for name in ['labels/oai/patient.csv','results/oof/a.csv','code/oai/score_audit/cohort.csv','code/oai/scores/raw.txt','.env','config/paths.json','config/local-secrets.json','code/oai/soft_followup6250_v3/results/writer.py','model.pt']:
            p=mini/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic only')
            check(not sandbox['eligible'](p),'Publication excludes local asset: '+name)

    report = dict(passed=not errors, check_count=len(checks), checks=checks, errors=errors,
                  third_party_import_roots=sorted(third_party), embedded_sources_parsed=embedded,
                  real_data_used=False, gpu_execution_tested=False, remote_scoring_tested=False)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
