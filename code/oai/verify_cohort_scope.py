"""Choose baseline knees from raw score availability, independently of sequence availability."""
import csv
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from audit_scores import read_nda

root = Path(__file__).resolve().parent
(root / 'score_audit').mkdir(parents=True, exist_ok=True)
scores = [r for r in read_nda(root / 'scores/oai_kmrisemiquant01.txt') if r['visit'] == 'V00']
groups = defaultdict(list)
for r in scores:
    side = {'1': 'RIGHT', '2': 'LEFT'}.get(r['side'])
    if side:
        groups[(r['src_subject_id'].strip(), side)].append(r)
fields = [f for f in scores[0] if f.startswith('m_cm_') or f in {
    'm_lmah_morph','m_lmbody_morph','m_lmph_morph','m_mmah_morph','m_mmbody_morph','m_mmph_morph',
    'm_acl_tear','m_eff_wk','m_syic','m_popcys'}]
cohort = {k for k, rs in groups.items() if any(r.get(f, '').strip() for r in rs for f in fields)}
all_mri = defaultdict(lambda: defaultdict(list))
for r in read_nda(root / 'baseline/image03.txt'):
    desc = r['image_description']
    if r['image_modality'] != 'MRI' or not desc.endswith(('_LEFT','_RIGHT')):
        continue
    seq, side = desc.rsplit('_', 1)
    all_mri[(r['src_subject_id'].strip(),side)][seq].append(r)
package = int(os.environ.get('OAI_BASELINE_PACKAGE_ID', '1249823'))
with (root / f'tool_state/NDA/nda-tools/downloadcmd/packages/{package}/package_file_metadata_{package}.txt').open(encoding='utf-8-sig', newline='') as f:
    sizes = {r['NDA_S3_URL']: int(r['FILE_SIZE']) for r in csv.DictReader(f)}
seqs = sorted({s for k in cohort for s in all_mri[k]})
coverage = {}
manifest = []
for seq in seqs:
    selected = [(k, r) for k in sorted(cohort) for r in all_mri[k].get(seq, [])]
    coverage[seq] = {'knees': sum(bool(all_mri[k].get(seq)) for k in cohort),
                     'series': len(selected), 'bytes': sum(sizes[r['image_file']] for k,r in selected),
                     'duplicate_knees': sum(len(all_mri[k].get(seq,[]))>1 for k in cohort)}
    for k, r in selected:
        manifest.append({'src_subject_id':k[0],'side':k[1],'sequence':seq,
                         'image_file':r['image_file'],'bytes':sizes[r['image_file']],
                         'duplicate_series_for_knee_sequence':len(all_mri[k][seq])})
sets = {
    'SAG_IW_only':['SAG_IW_TSE'],
    'two_IW':['SAG_IW_TSE','COR_IW_TSE'],
    'two_IW_and_DESS':['SAG_IW_TSE','COR_IW_TSE','SAG_3D_DESS'],
    'five_sequences':['SAG_IW_TSE','COR_IW_TSE','SAG_3D_DESS','COR_MPR','AX_MPR']}
set_summary = {}
for name, sequences in sets.items():
    selected = [r for r in manifest if r['sequence'] in sequences]
    set_summary[name] = {'sequences':sequences,
                         'complete_knees':sum(all(all_mri[k].get(s) for s in sequences) for k in cohort),
                         'series':len(selected), 'GB':sum(r['bytes'] for r in selected)/1e9}
    unique_groups = [all_mri[k][s] for k in cohort for s in sequences if all_mri[k].get(s)]
    set_summary[name].update({
        'unique_knee_sequence_groups':len(unique_groups),
        'one_series_per_group_GB_min':sum(min(sizes[r['image_file']] for r in rs) for rs in unique_groups)/1e9,
        'one_series_per_group_GB_max':sum(max(sizes[r['image_file']] for r in rs) for rs in unique_groups)/1e9})
    if name == 'five_sequences':
        (root / 'score_audit/multisequence_candidate_s3_links.txt').write_text(
            '\n'.join(sorted({r['image_file'] for r in selected}))+'\n', encoding='utf-8')
availability = {}
for name, fs in {
    'lateral_cartilage':['m_cm_clf','m_cm_plf','m_cm_alt','m_cm_clt','m_cm_plt'],
    'PF_cartilage':['m_cm_alf','m_cm_amf','m_cm_lp','m_cm_mp'],
    'lateral_meniscus':['m_lmah_morph','m_lmbody_morph','m_lmph_morph'],
    'medial_meniscus':['m_mmah_morph','m_mmbody_morph','m_mmph_morph'],
    'ACL':['m_acl_tear'], 'effusion':['m_eff_wk'], 'Hoffa':['m_syic'], 'Baker':['m_popcys']}.items():
    availability[name] = {'any_field_knees':sum(any(r.get(f,'').strip() for r in groups[k] for f in fs) for k in cohort),
                          'complete_fields_in_one_reading_knees':sum(any(all(r.get(f,'').strip() for f in fs) for r in groups[k]) for k in cohort)}
result = {'selection':'V00 + unilateral knee + any recorded major MOAKS field, independent of image sequence and positivity threshold',
          'baseline_scored_knees_all_systems':len(groups), 'major_MOAKS_knees':len(cohort),
          'major_MOAKS_participants':len({k[0] for k in cohort}),
          'with_any_MRI':sum(bool(all_mri[k]) for k in cohort),
          'without_SAG_IW':[list(k) for k in sorted(cohort) if not all_mri[k].get('SAG_IW_TSE')],
          'score_availability':availability,'sequence_coverage':coverage,'download_sets':set_summary,
          'all_indexed_side_specific_MRI_GB':sum(r['bytes'] for r in manifest)/1e9,
          'note':'All sizes are compressed package file sizes and include duplicate series. Five-sequence list is our candidate set, not a verified reproduction of the author. No images downloaded.'}
out=root / 'score_audit'
(out / 'cohort_scope_verification.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
with (out / 'all_mri_candidates.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
with (out / 'major_moaks_baseline_knees.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['src_subject_id','side','score_rows','projects'])
    for k in sorted(cohort):
        w.writerow([*k,len(groups[k]),';'.join(sorted({r['readprj'] for r in groups[k]}))])
assert len({r['image_file'] for r in manifest}) == len(manifest)
print(json.dumps(result,indent=2))
