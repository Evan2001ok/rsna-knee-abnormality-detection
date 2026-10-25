"""Validate recovered sources and fold/path adapter behavior without importing models."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from run_pipeline import PathsOnly, load_mapping, transformed

ROOT = Path(__file__).resolve().parents[1]


def signature(s):
    return ast.dump(ast.parse(s), include_attributes=False)


def final_call(tree):
    guards = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert len(guards) == 1, 'Expected one main guard'
    calls = [n for n in ast.walk(guards[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_all_folds']
    assert len(calls) == 1
    return calls[0]


def fold_value(call):
    return call.args[1] if len(call.args) >= 2 else next(k.value for k in call.keywords if k.arg == 'folds')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--notebook', help='Optional original Notebook4 for independent extraction checks')
    ap.add_argument('--report')
    a = ap.parse_args()
    provenance = json.loads((ROOT / 'config/training_source_provenance.json').read_text(encoding='utf-8'))
    checks = []
    def check(value, label):
        if not value:
            raise AssertionError(label)
        checks.append(label)

    mapping = load_mapping(ROOT / 'config/paths.example.json')
    for record in provenance['files']:
        path = ROOT / record['file']
        text = path.read_text(encoding='utf-8')
        tree = ast.parse(text)
        check(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], 'Source hash: '+path.name)
        check(sum(isinstance(n, ast.FunctionDef) and n.name == 'main' for n in tree.body) == 1, 'Single training program: '+path.name)
        original_call = ast.dump(final_call(tree), include_attributes=False)
        unselected, _ = transformed(path, [])
        check(ast.dump(final_call(unselected), include_attributes=False) == original_call, 'Default fold call preserved: '+path.name)
        selected, changes = transformed(path, mapping, folds=(0,))
        compile(selected, str(path), 'exec')  # Compiles only; never execute it.
        check(ast.literal_eval(fold_value(final_call(selected))) == (0,), 'Actual main selects only fold0: '+path.name)
        check(bool(changes), 'Path mappings found: '+path.name)

    # Regression: original sources use implicit, explicit-keyword and positional folds.
    for src in ['run_all_folds()', 'run_all_folds(Cfg(), folds=(0,1,2,3))', 'run_all_folds(Cfg(), (0,1,2,3))']:
        t = ast.parse(src)
        selected = ast.fix_missing_locations(PathsOnly([], folds=(2, 3)).visit(t))
        call = selected.body[0].value
        check(ast.literal_eval(fold_value(call)) == (2, 3), 'Fold adapter case: '+src)
        if 'Cfg()' in src:
            check(ast.unparse(call.args[0]) == 'Cfg()', 'Preserve supplied Cfg argument: '+src)

    # Verify source-level recipe constraints; these are not numerical forward tests.
    files = {n: (ROOT / 'code/train' / n).read_text(encoding='utf-8') for n in ['conv224.py', 'conv_gray384.py', 'dino3.py']}
    check('img_size: int = 224' in files['conv224.py'] and 'crop_mm: float = 160.0' in files['conv224.py'], '224 native cache/crop')
    check('single_grayscale_slice_repeated_to_3_channels' in files['conv_gray384.py'], 'Grayscale representation metadata retained')
    check("convnext_tiny.dinov3_lvd1689m" in files['dino3.py'] and 'epochs: int = 16' in files['dino3.py'], 'DINOv3 model and duration retained')
    check('backbone_lr: float = 2e-5' in files['dino3.py'], 'DINOv3 split learning rate retained')

    paths = json.loads((ROOT / 'config/paths.example.json').read_text(encoding='utf-8'))['prefix_map']
    check(paths['/kaggle/input/datasets/minminandy/oai-baseline-2399-224px-images'].endswith('/kaggle_upload_224'), 'Exact224 OAI image mapping')

    if a.notebook:
        nb_path = Path(a.notebook)
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        check(hashlib.sha256(nb_path.read_bytes()).hexdigest() == provenance['source_notebook_sha256'], 'Original notebook hash')
        source224 = ''.join(nb['cells'][19]['source'])
        check(files['conv224.py'] == source224, '224 extraction text-exact')
        gray = ''.join(nb['cells'][8]['source'])
        tree = ast.parse(gray)
        guards = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
        check(len(guards) == 2, 'Original grayscale cell contains two programs')
        lines = gray.splitlines(keepends=True)
        first, second = ''.join(lines[:guards[0].end_lineno]), ''.join(lines[guards[0].end_lineno:])
        check(signature(first) == signature(second) == signature(files['conv_gray384.py']), 'Both grayscale copies and packaged source AST-identical')

    result = dict(passed=True, checks=checks, check_count=len(checks),
                  original_notebook_checked=bool(a.notebook),
                  gpu_training_tested=False, model_code_imported=False,
                  final_team_inference_included=False,
                  dino3_source=provenance['dino3_fixed_version_url'])
    if a.report:
        Path(a.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
