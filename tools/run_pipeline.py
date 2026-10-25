"""Apply explicit path mappings at runtime; preserve all computational source nodes."""
import argparse,ast,csv,json,os,sys,types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load_mapping(filename):
    cfg=json.loads(Path(filename).read_text(encoding='utf-8'))
    pairs=[]
    for old,new in cfg['prefix_map'].items():
        new=new.replace('${PACKAGE}',str(ROOT)).replace('\\','/')
        if '${' in new or '<' in new:raise ValueError('Fill placeholder in path configuration: '+old)
        pairs.append((old.replace('\\','/').rstrip('/'),new.rstrip('/')))
    return sorted(pairs,key=lambda x:len(x[0]),reverse=True)

class PathsOnly(ast.NodeTransformer):
    def __init__(self,pairs,part=None,folds=None):self.pairs=pairs;self.part=part;self.folds=folds;self.changes=[]
    def visit_JoinedStr(self,node):
        # A path may be f'D:/OAI/cache_{resolution}'. Map its leading literal,
        # retaining every expression, format specifier and pixel operation.
        if node.values and isinstance(node.values[0],ast.Constant) and isinstance(node.values[0].value,str):
            first=node.values[0];s=first.value.replace('\\','/')
            for old,new in self.pairs:
                if s==old or s.startswith(old+'/'):
                    self.changes.append([first.value,new+s[len(old):]])
                    first.value=new+s[len(old):]
                    break
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            s=node.value.replace('\\','/')
            for old,new in self.pairs:
                if s==old or s.startswith(old+'/'):
                    self.changes.append([node.value,new+s[len(old):]])
                    return ast.copy_location(ast.Constant(new+s[len(old):]),node)
        return node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if self.part is not None and any(isinstance(t,ast.Name) and t.id=='PART_ID' for t in node.targets):node.value=ast.Constant(self.part)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if self.folds is not None and isinstance(node.func,ast.Name) and node.func.id=='run_all_folds':
            requested=ast.Tuple(elts=[ast.Constant(f) for f in self.folds],ctx=ast.Load())
            # Some recovered sources explicitly call run_all_folds(Cfg(), folds=...).
            # Override only fold selection, preserving the supplied configuration.
            if len(node.args)>=2:
                node.args[1]=requested
                node.keywords=[k for k in node.keywords if k.arg!='folds']
            else:
                node.keywords=[k for k in node.keywords if k.arg!='folds']
                node.keywords.append(ast.keyword(arg='folds',value=requested))
        return node

def transformed(path,mapping,part=None,folds=None):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));patch=PathsOnly(mapping,part,folds)
    return ast.fix_missing_locations(patch.visit(tree)),patch.changes

def training_label_paths(tree):
    """Read only the three literal label-input defaults from the mapped source AST."""
    cfg=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Cfg')
    result={}
    for node in cfg.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in {'scored_path','v1_path','v2_path'}:
            value=node.value
            if not (isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='Path' and len(value.args)==1):
                raise ValueError('Cannot validate label path default: '+node.target.id)
            result[node.target.id]=str(ast.literal_eval(value.args[0]))
    if len(result)!=3:raise ValueError('Expected exactly three source label paths')
    return result

def reject_mixed_label_inputs(tree):
    """Prevent reference training-target exports from being blended a second time."""
    paths=training_label_paths(tree)
    if len({str(Path(p).resolve()) for p in paths.values()})!=3:
        raise ValueError('The VLM, V1 and V2 inputs must be three distinct source files.')
    for field,path in paths.items():
        with Path(path).open(encoding='utf-8-sig',newline='') as handle:
            columns=next(csv.reader(handle),[])
        if 'label_source' in columns or Path(path).name in {'conv_labels_used.csv','conv_new_labels_used.csv','dino_new_labels_used.csv'}:
            raise ValueError('Already mixed training labels cannot be used as '+field+': '+path+
                             '. Map the original VLM/V1/V2 sources; see labels/rsna/README.md.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--script',required=True,help='Package-relative code file')
    ap.add_argument('--paths',default=str(ROOT/'config/paths.json'))
    ap.add_argument('--part',type=int);ap.add_argument('--folds',help='e.g. 0 or 0,1,2,3')
    ap.add_argument('--plan',action='store_true',help='Print path replacements only; do not import/execute source')
    ap.add_argument('script_args',nargs=argparse.REMAINDER)
    a=ap.parse_args();path=(ROOT/a.script).resolve()
    if not path.is_relative_to((ROOT/'code').resolve()):ap.error('Script must be inside package code/')
    mapping=load_mapping(a.paths);folds=tuple(map(int,a.folds.split(','))) if a.folds else None
    if folds is not None and (not folds or any(f not in range(4) for f in folds) or len(set(folds))!=len(folds)):ap.error('folds must be distinct 0..3')
    if a.part is not None and (path.name not in ('cache224.py','cache384.py') or a.part not in range(2 if path.name=='cache224.py' else 4)):ap.error('Invalid cache part')
    tree,changes=transformed(path,mapping,a.part,folds)
    if a.plan:
        print(json.dumps({'script':str(path),'path_changes':changes,'part':a.part,'folds':folds,'execution':False},ensure_ascii=False,indent=2));return
    if path.parent==(ROOT/'code/train').resolve():reject_mixed_label_inputs(tree)
    # All path mapping occurs before dataclass defaults/global values are evaluated.
    # No edits to convolution, normalization, attention, losses or optimizer code.
    mod=types.ModuleType('__main__');mod.__file__=str(path);sys.modules['__main__']=mod
    sys.path[:0]=[str(path.parent),str(ROOT/'code/oai')]
    sys.argv=[str(path),*([*a.script_args[1:]] if a.script_args[:1]==['--'] else a.script_args)]
    os.chdir(ROOT)
    exec(compile(tree,str(path),'exec'),mod.__dict__)
if __name__=='__main__':main()
