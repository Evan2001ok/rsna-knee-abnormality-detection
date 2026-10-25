"""Read-only, offline package audit. Does not import training code, score images or use a GPU."""
import argparse,ast,csv,hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
T=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',action='store_true',help='Require separately supplied private labels, splits and OOF')
    ap.add_argument('--dependencies',action='store_true')
    ap.add_argument('--dependency-group',choices=('pipeline','nda','all'),default='pipeline')
    ap.add_argument('--external',action='store_true')
    ap.add_argument('--recipe',choices=[p.stem for p in sorted((ROOT/'code/train').glob('*.py'))]+['ensemble6','ensemble8'],help='Restrict external-asset inventory to one training or inference recipe')
    ap.add_argument('--paths',default=str(ROOT/'config/paths.json'));ap.add_argument('--report');a=ap.parse_args()
    if a.recipe and not a.external:ap.error('--recipe requires --external')
    report={'offline':True,'errors':[],'external_inputs_checked':False,'gpu_training_tested':False}
    def check(condition,message):
        if not condition:report['errors'].append(message)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        p=ROOT/r['file'];check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'File hash: '+r['file'])
    for p in [*(ROOT/'code').rglob('*.py'),*(ROOT/'tools').glob('*.py')]:
        try:ast.parse(p.read_text(encoding='utf-8-sig'))
        except SyntaxError as e:check(False,f'Syntax: {p.relative_to(ROOT)}:{e.lineno}')
    required=['report_v1_prompt.txt','report_v2_prompt.txt','vlm_image_report_prompt.txt']
    for f in required:check((ROOT/'code/labeling/rsna'/f).is_file(),'Missing prompt '+f)
    check((ROOT/'code/oai/oai_vlm_soft_mapping_prompt_v3_seven_targets.txt').is_file(),'Missing OAI v3 prompt')
    from validate_rsna_labels import audit as audit_rsna_labels, audit_training_recipes
    try:
        report['published_rsna_labels']=audit_rsna_labels()
        report['training_label_paths_and_recipes']=audit_training_recipes()
    except (AssertionError,ValueError,KeyError,OSError) as error:check(False,'Published RSNA label audit: '+str(error))
    report['package_hashes_checked']=len(manifest['files'])
    report['private_assets_checked']=a.assets
    asset_paths=['results/splits.csv',*[f'results/oof/{n}.csv' for n in ['conv','eff','dino','dino3','rad','tiny','conv_new','dino_new','A_original_six_equal','B_new_two_equal','AB_70_30']],
                 'labels/oai/OAI_soft_labels_2399_v3.csv','labels/oai/OAI_soft_followup6250_v3.csv']
    missing_assets=[s for s in asset_paths if not (ROOT/s).is_file()] if a.assets else []
    for s in missing_assets:check(False,'External historical asset required: '+s)
    report['private_assets_checked']=a.assets and not missing_assets
    if a.assets and not missing_assets:
        split=readcsv(ROOT/'results/splits.csv');ids=[r['StudyInstanceUID'] for r in split]
        check(len(ids)==len(set(ids))==4407,'Split UID count');check(sum(int(r['is_gold']) for r in split)==58,'Gold count')
        predictions={}
        for name in ['conv','eff','dino','dino3','rad','tiny','conv_new','dino_new','A_original_six_equal','B_new_two_equal','AB_70_30']:
            rows=readcsv(ROOT/f'results/oof/{name}.csv');mapping={r['StudyInstanceUID']:r for r in rows};predictions[name]=mapping
            check(len(mapping)==len(rows) and set(mapping)==set(ids),'OOF IDs '+name)
            for r in split:
                v=mapping[r['StudyInstanceUID']]
                check(v['fold']==r['fold'] and v['is_gold']==r['is_gold'],'OOF split '+name)
                check(all(math.isfinite(float(v[t])) and 0<=float(v[t])<=1 for t in T),'OOF probability '+name)
        for uid in ids:
            for t in T:
                aa=sum(float(predictions[n][uid][t]) for n in ['conv','eff','dino','dino3','rad','tiny'])/6
                bb=sum(float(predictions[n][uid][t]) for n in ['conv_new','dino_new'])/2
                check(abs(float(predictions['A_original_six_equal'][uid][t])-aa)<1e-12,'Blend A')
                check(abs(float(predictions['B_new_two_equal'][uid][t])-bb)<1e-12,'Blend B')
                check(abs(float(predictions['AB_70_30'][uid][t])-(.7*aa+.3*bb))<1e-12,'Blend AB')
        for name,count in [('OAI_soft_labels_2399_v3.csv',2399),('OAI_soft_followup6250_v3.csv',6250)]:
            rows=readcsv(ROOT/'labels/oai'/name);check(len(rows)==len({r['knee_id'] for r in rows})==count,'OAI identities '+name)
            for r in rows:
                for t in ['Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis']:
                    mask=int(r[t+'__mask']);p=r[t]
                    check(mask in (0,1),'OAI mask')
                    check((p.strip()=='' or p.lower() in ('nan','null')) if mask==0 else (math.isfinite(float(p)) and 0<=float(p)<=1),'OAI score/mask')
        report['aligned_oof_studies']=len(ids)
    if a.dependencies:
        modules=['numpy','pandas','pydicom','PIL','cv2','torch','torchvision','timm','sklearn','tqdm','matplotlib','safetensors'] if a.dependency_group in ('pipeline','all') else []
        if a.dependency_group in ('nda','all'):modules.append('NDATools')
        found={n:importlib.util.find_spec(n) is not None for n in modules};report['dependency_modules_available']=found
        report['missing_dependencies']=[n for n,v in found.items() if not v]
        # Availability is not ABI compatibility or CUDA execution validation.
    if a.external:
        from run_pipeline import load_mapping,transformed
        cfg=json.loads((ROOT/'config/resource_inventory.json').read_text(encoding='utf-8'));pairs=load_mapping(a.paths)
        check_paths=[]
        for item in cfg['inputs']:
            if a.recipe:
                if a.recipe.startswith('ensemble'):
                    keys={'conv','eff','dino','dino3','rad','tiny'}|({'conv_new','dino_new'} if a.recipe=='ensemble8' else set())
                    if not (item['kind']=='trained_checkpoint' and set(item['used_by']) & keys):continue
                elif item['kind']=='trained_checkpoint' or a.recipe+'.py' not in item['used_by']:continue
            s=item['path'];new=s
            for old,value in pairs:
                if s==old or s.startswith(old+'/'):new=value+s[len(old):];break
            check_paths.append({'original':s,'resolved':new,'exists':Path(new).exists(),'kind':item['kind']})
        if a.recipe:
            base='/kaggle/input/competitions/rsna-knee-abnormality-detection'
            required=['test_series.csv','sample_submission.csv','test_series'] if a.recipe.startswith('ensemble') else ['train.csv']
            for name in required:
                s=base+'/'+name;new=s
                for old,value in pairs:
                    if s==old or s.startswith(old+'/'):new=value+s[len(old):];break
                check_paths.append({'original':s,'resolved':new,'exists':Path(new).exists(),'kind':'competition_input'})
        report['recipe']=a.recipe
        report['external_inputs_checked']=True;report['external_paths']=check_paths
        report['missing_external_inputs']=[r for r in check_paths if not r['exists']]
    report['passed']=not(report['errors'] or report.get('missing_dependencies') or report.get('missing_external_inputs'));report['errors']=list(dict.fromkeys(report['errors']))
    if a.report:Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['errors'] or report.get('missing_dependencies') or report.get('missing_external_inputs'):raise SystemExit(1)
if __name__=='__main__':main()
