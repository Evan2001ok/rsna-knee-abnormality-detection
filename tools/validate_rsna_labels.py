"""Validate the seven frozen public RSNA label snapshots, without ML dependencies."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit_training_recipes(paths=None, execute_loaders=False):
    """Check actual training defaults after path mapping; optionally execute only label loaders."""
    from run_pipeline import load_mapping, transformed, training_label_paths, reject_mixed_label_inputs
    mapping = load_mapping(paths or ROOT / 'config/paths.example.json')
    manifest = json.loads((ROOT / 'labels/rsna/manifest.json').read_text(encoding='utf-8'))
    hashes = {item['file']: item['sha256'] for item in manifest['files']}
    targets = manifest['targets']
    records = []
    for source in sorted((ROOT / 'code/train').glob('*.py')):
        tree, _ = transformed(source, mapping)
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Cfg')
        defaults = {n.target.id: n.value for n in cfg.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        vlm = 'rsna_vlm_opus_original.csv' if source.stem in ('conv_new', 'dino_new') else 'rsna_vlm_original.csv'
        expected = dict(scored_path=vlm, v1_path='rsna_llm_v1_original.csv', v2_path='rsna_llm_v2_original.csv')
        paths_actual = training_label_paths(tree)
        reject_mixed_label_inputs(tree)
        for key, filename in expected.items():
            assert hashlib.sha256(Path(paths_actual[key]).read_bytes()).hexdigest() == hashes[filename], (source.name, key, 'Wrong frozen source or modified CSV')
        weight = .5 if source.stem == 'dino_new' else .7
        assert ast.literal_eval(defaults['vlm_weight']) == weight
        assert ast.literal_eval(defaults['v1_weight']) == .7
        source_targets = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'TARGETS' for t in n.targets))
        assert ast.literal_eval(source_targets) == targets, source.name
        reference = 'dino_new_labels_used.csv' if source.stem == 'dino_new' else 'conv_new_labels_used.csv' if source.stem == 'conv_new' else 'conv_labels_used.csv'
        records.append(dict(script=source.name, source_files=expected, resolved_paths=paths_actual,
                            vlm_weight=weight, v1_weight=.7, reference_snapshot=reference))
    assert len(records) == 10
    if execute_loaders:
        # Execute the original two label functions, never model imports/training.
        # The metadata fixture inherits saved Gold values; it is not a new check
        # against the separately obtained official train.csv or fold metadata.
        import logging
        import tempfile
        from types import SimpleNamespace
        import numpy as np
        import pandas as pd
        reference = pd.read_csv(ROOT / 'labels/rsna/conv_labels_used.csv', dtype={'StudyInstanceUID': str})
        official_fixture = reference[['StudyInstanceUID', *targets]].copy()
        official_fixture.loc[reference['label_source'] != 'gold', targets] = np.nan
        (ROOT / 'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='rsna-label-loaders-', dir=ROOT / 'work') as temporary:
            metadata = Path(temporary) / 'train.csv'
            official_fixture.to_csv(metadata, index=False)
            for record in records:
                source = ROOT / 'code/train' / record['script']
                tree, _ = transformed(source, mapping)
                functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in ('read_scored_labels', 'load_vlm_labels')]
                assert len(functions) == 2
                namespace = dict(np=np, pd=pd, Path=Path, TARGETS=targets, Cfg=SimpleNamespace,
                                 log=logging.getLogger('rsna-label-audit'))
                exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), 'exec'), namespace)
                cfg = SimpleNamespace(**{k: Path(v) for k, v in record['resolved_paths'].items()},
                                      vlm_weight=record['vlm_weight'], v1_weight=.7,
                                      train_csv=metadata, n_gold_expected=58, label_desc='audit')
                actual, gold, _ = namespace['load_vlm_labels'](cfg)
                saved = pd.read_csv(ROOT / 'labels/rsna' / record['reference_snapshot'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
                delta = float(np.max(np.abs(actual[targets].to_numpy() - saved.reindex(actual.index)[targets].to_numpy())))
                assert delta < 1e-12 and len(gold) == 58, (source.name, delta)
                record['executed_loader_max_error'] = delta
    return dict(passed=True, training_entries=records, original_label_loaders_executed=execute_loaders,
                official_metadata_independently_verified=False, model_training_executed=False)


def audit():
    folder = ROOT / 'labels/rsna'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    tables, gold_ids, deltas = {}, None, {}
    targets = manifest['targets']
    assert len(manifest['files']) == 7 and len(targets) == 12
    for item in manifest['files']:
        path = folder / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path.name
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            expected = ['StudyInstanceUID', *targets]
            if item['file'] in manifest['recipes']:
                expected.append('label_source')
            assert reader.fieldnames == expected, path.name
        table = {r['StudyInstanceUID']: r for r in rows}
        assert len(rows) == len(table) == manifest['study_count'] == 4407
        assert all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1
                   for r in rows for t in targets), path.name
        tables[path.name] = table
    ids = set(next(iter(tables.values())))
    assert all(set(table) == ids for table in tables.values())
    for name, recipe in manifest['recipes'].items():
        table = tables[name]
        gold = {uid for uid, row in table.items() if row['label_source'] == 'gold'}
        assert len(gold) == manifest['training_gold_override_count'] == 58
        assert all(row['label_source'] in ('gold', 'vlm07_llm03') for row in table.values())
        if gold_ids is None:
            gold_ids = gold
        assert gold == gold_ids
        weight = recipe['vlm_weight']
        delta = max(abs(float(row[t]) - (
            weight * float(tables[recipe['vlm_source']][uid][t]) + (1-weight) * (
                .7 * float(tables['rsna_llm_v1_original.csv'][uid][t]) +
                .3 * float(tables['rsna_llm_v2_original.csv'][uid][t]))))
            for uid, row in table.items() if uid not in gold for t in targets)
        assert delta < 1e-12, (name, delta)
        assert all(float(table[uid][t]) == float(tables['conv_labels_used.csv'][uid][t])
                   for uid in gold for t in targets)
        deltas[name] = delta
    return dict(passed=True, files=7, studies=4407, targets=12, gold_overrides=58,
                max_blend_errors=deltas, clinical_correctness_tested=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', help='Audit a configured path map; default uses paths.example.json')
    parser.add_argument('--training-loaders', action='store_true', help='Also run the ten original label loaders against a Gold metadata fixture; needs numpy/pandas, no GPU')
    parser.add_argument('--report')
    args = parser.parse_args()
    result = audit()
    result['training_recipe_audit'] = audit_training_recipes(args.paths, args.training_loaders)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
