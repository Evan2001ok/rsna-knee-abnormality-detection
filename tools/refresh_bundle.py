"""Refresh the source-reading notebook, hashes and optional ZIP; never run pipeline stages."""
import argparse
import ast
import hashlib
import json
import shlex
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_NOTE = 'October25 supplement: entry points, source recovery, fold adapter or regenerated support metadata'


def eligible(path):
    parts = path.relative_to(ROOT).parts
    rel = path.relative_to(ROOT).as_posix()
    if rel in {'labels/rsna/conv_new_labels_used.csv', 'labels/rsna/rsna_vlm_original.csv', 'labels/rsna/rsna_llm_v2_original.csv', 'labels/rsna/rsna_vlm_opus_original.csv', 'labels/rsna/dino_new_labels_used.csv', 'labels/rsna/README.md', 'labels/rsna/manifest.json', 'labels/rsna/rsna_llm_v1_original.csv', 'labels/rsna/conv_labels_used.csv'}:
        return path.is_file()
    if not path.is_file() or any(p in {'__pycache__', '.git', '.venv', 'data', 'work', 'labels', 'results', 'score_audit', 'scores', 'tool_state'} for p in parts):
        return False
    # Allowlist publication material. A local data restore must not enter a new ZIP.
    if parts[0]=='code':
        source_location=(len(parts)==2 and path.name=='oai_manifest.py') or (len(parts)==3 and parts[1] in {'train','inference','rsna','oai'}) or (len(parts)==4 and tuple(parts[1:3]) in {('labeling','rsna'),('oai','transforms')})
        return (source_location and path.suffix=='.py') or path.relative_to(ROOT).as_posix() in {
            'code/oai/score_dictionary.csv','code/oai/oai_vlm_soft_mapping_prompt_v3_seven_targets.txt',
            'code/labeling/rsna/report_v1_prompt.txt','code/labeling/rsna/report_v2_prompt.txt',
            'code/labeling/rsna/vlm_image_report_prompt.txt'}
    if parts[0]=='tools':return path.suffix=='.py'
    if parts[0]=='config':return path.name in {'paths.example.json','nda_packages.example.json','resource_inventory.json',
        'eight_model_manifest.json','source_urls.json','training_source_provenance.json','reproduction_source_provenance.json','external_resources.json'}
    if parts[0]=='docs':return path.suffix=='.md'
    if parts[0]=='requirements':return path.suffix=='.txt'
    return len(parts)==1 and path.name in {'README.md','REPRODUCE.md','ENVIRONMENT.md','AI_USAGE.md',
        'PUBLIC_RELEASE.md','REPRODUCIBILITY_AUDIT.md','.gitignore','.gitattributes','our_pipeline.ipynb',
        'requirements_reference.txt','source_supplement_validation.json','verification.json','manifest.json'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', help='Optional output archive path, outside package')
    a = ap.parse_args()
    codepaths = sorted(p for folder in ['code', 'tools'] for p in (ROOT / folder).rglob('*')
                       if eligible(p) and p.suffix != '.csv' and 'score_audit' not in p.parts)
    cells = []
    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
    def code(text):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))
    md('# RSNA Knee: Essential source reading edition\n\nRead README, AI_USAGE and REPRODUCE first. Cells restore source text only; they do not train, download or score. Copy the repository for configs and docs; frozen RSNA labels are included separately; OAI labels, OOF, splits and cohort manifests are external. Includes recovered224, graycopy384 and original DINOv3 training sources. Final team inference is not included.\n')
    dirs = sorted({(Path('rsna_core_restored') / p.relative_to(ROOT).parent).as_posix() for p in codepaths})
    code('from pathlib import Path\nfor p in '+repr(dirs)+':\n    Path(p).mkdir(parents=True, exist_ok=True)\n')
    for p in codepaths:
        rel = p.relative_to(ROOT)
        md('### `'+rel.as_posix()+'`')
        code('%%writefile '+shlex.quote((Path('rsna_core_restored') / rel).as_posix())+'\n'+p.read_text(encoding='utf-8-sig'))
    for i, cell in enumerate(cells):
        cell['id'] = f'core-{i:03d}'
    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': dict(name='python3', display_name='Python3', language='python'),
        'language_info': dict(name='python')}, cells=cells)
    (ROOT / 'our_pipeline.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

    python_files = sorted(p for folder in ['code', 'tools'] for p in (ROOT / folder).rglob('*.py') if eligible(p))
    for p in python_files:
        ast.parse(p.read_text(encoding='utf-8-sig'))
    for c in cells:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        if src.startswith('%%writefile '):
            header, body = src.split('\n', 1)
            target = Path(shlex.split(header)[1]).relative_to('rsna_core_restored')
            assert body == (ROOT / target).read_text(encoding='utf-8-sig')
        else:
            ast.parse(src)
    old_verification = json.loads((ROOT / 'verification.json').read_text(encoding='utf-8'))
    old_verification.update(python_files_syntax_ok=len(python_files), restored_source_files=len(codepaths),
                            notebook_cells=len(cells), source_bodies_exact=True,
                            training_source_supplement='config/training_source_provenance.json',
                            supplement_checks='source_supplement_validation.json',
                            gpu_training_tested=False)
    (ROOT / 'verification.json').write_text(json.dumps(old_verification, indent=2), encoding='utf-8')

    original = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    old = {r['file']: r for r in original['files']}
    supplements = json.loads((ROOT / 'config/training_source_provenance.json').read_text(encoding='utf-8'))
    new_sources = {r['file']: r for r in supplements['files']}
    records = []
    for p in sorted(ROOT.rglob('*')):
        if not eligible(p) or p == ROOT / 'manifest.json':
            continue
        rel = p.relative_to(ROOT).as_posix()
        row = dict(old.get(rel, new_sources.get(rel, dict(file=rel, source='October25 source supplement / package support', change='Added operational documentation, validation or source provenance'))))
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        notes = list(dict.fromkeys(row.get('change', '').split('; ')))  # also collapses notes repeated by earlier runs
        if rel in old and old[rel]['sha256'] != actual and SUPPLEMENT_NOTE not in notes:
            notes.append(SUPPLEMENT_NOTE)
        row['change'] = '; '.join(notes)
        row.update(file=rel, sha256=actual, bytes=p.stat().st_size)
        records.append(row)
    manifest = dict(files=records, scope='public pipeline with seven frozen RSNA label snapshots; OAI labels, OOF, splits, cohort manifests and trained weights excluded; final team inference excluded')
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    result = dict(manifest_files=len(records), restored_source_files=len(codepaths),
                  python_files_syntax_ok=len(python_files), source_bodies_exact=True)
    if a.zip:
        archive = Path(a.zip).resolve()
        if archive.is_relative_to(ROOT):
            raise ValueError('Write the ZIP outside the package')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for row in records:
                z.write(ROOT / row['file'], row['file'])
            z.write(ROOT / 'manifest.json', 'manifest.json')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for row in records:
                assert hashlib.sha256(z.read(row['file'])).hexdigest() == row['sha256']
        result.update(archive=str(archive), archive_bytes=archive.stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
