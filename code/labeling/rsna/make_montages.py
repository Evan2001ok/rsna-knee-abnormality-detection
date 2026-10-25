import glob, json, os, numpy as np, pandas as pd, pydicom
from PIL import Image, ImageDraw, ImageFont

import argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--data-root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
COMP=str(Path(args.data_root).resolve())
OUT=str(Path(args.output).resolve()); os.makedirs(OUT,exist_ok=True)
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
     'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
tr  = pd.read_csv(f'{COMP}/train.csv', dtype={'StudyInstanceUID': str})
ser = pd.read_csv(f'{COMP}/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
# slot: (平面, Fluid_Sensitive, Fat_Suppression, 网格列, 网格行)
SLOTS = {'SAG_FS': ('Sagittal', 1, 1, 4, 4), 'COR_FS': ('Coronal', 1, 1, 4, 3),
         'AX_FS': ('Axial', 1, 1, 4, 3), 'COR_T1': ('Coronal', 0, 0, 4, 3)}
TILE = 320
try:
    FONT=ImageFont.truetype('DejaVuSans.ttf',18); FONT_S=ImageFont.truetype('DejaVuSans.ttf',16)
except OSError:
    FONT=FONT_S=ImageFont.load_default()
NAMES = {'R': 'patient RIGHT', 'L': 'patient LEFT', 'A': 'ANTERIOR', 'P': 'POSTERIOR',
         'H': 'HEAD (superior)', 'F': 'FEET (inferior)'}

def files(uid, sid): return sorted(glob.glob(f'{COMP}/train_series/{uid}/{sid}/*.dcm'))

def pick_series(uid, plane, fluid, fs):
    s = ser[(ser.StudyInstanceUID == uid) & (ser.Anatomical_Plane == plane)]
    ex = s[(s.Fluid_Sensitive == fluid) & (s.Fat_Suppression == fs)]
    if len(ex) == 0 and fluid == 1: ex = s[s.Fluid_Sensitive == 1]   # 没有压脂就退到"液体敏感"
    if len(ex) == 0: return None
    n = {sid: len(files(uid, sid)) for sid in ex.SeriesInstanceUID}   # 同类取层数最多的序列
    return max(n, key=n.get) if max(n.values()) > 0 else None

def axis_letter(v):   # DICOM LPS：+x 患者左，+y 后，+z 头
    k = int(np.argmax(np.abs(v))); return [('R','L'), ('A','P'), ('F','H')][k][int(v[k] > 0)]

def to_tile(a, lo, hi):
    a = (np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a); s = TILE / max(im.size)
    im = im.resize((max(1, round(im.size[0]*s)), max(1, round(im.size[1]*s))), Image.BILINEAR)
    c = Image.new('L', (TILE, TILE), 0); c.paste(im, ((TILE-im.size[0])//2, (TILE-im.size[1])//2)); return c

def make_study(uid, report):
    entry = {'uid': uid, 'report': report, 'images': []}
    for slot, (plane, fluid, fs, gc, gr) in SLOTS.items():
        sid = pick_series(uid, plane, fluid, fs)
        if sid is None: continue
        hs = [(f, pydicom.dcmread(f, stop_before_pixels=True, force=True)) for f in files(uid, sid)]
        iop = np.array([float(x) for x in hs[0][1].ImageOrientationPatient]); r, c = iop[:3], iop[3:]
        nrm = np.cross(r, c)
        hs.sort(key=lambda h: float(np.dot([float(x) for x in h[1].ImagePositionPatient], nrm)))  # 按层位置排序
        n = len(hs); k = gc * gr
        idx = np.unique(np.linspace(int(.08*(n-1)), int(.92*(n-1)), min(k, n)).round().astype(int))  # 8%–92% 等距抽层
        arrs = [pydicom.dcmread(hs[i][0], force=True).pixel_array for i in idx]
        st = np.concatenate([a.ravel() for a in arrs]); lo, hi = np.percentile(st, 1), np.percentile(st, 99.5)
        m = Image.new('L', (gc*TILE, gr*TILE + 64), 0); d = ImageDraw.Draw(m)
        for t, a in enumerate(arrs):
            x, y = (t % gc)*TILE, 64 + (t // gc)*TILE
            m.paste(to_tile(a, lo, hi), (x, y)); d.text((x+6, y+4), f'#{t+1}', fill=255, font=FONT_S)
        desc = str(getattr(hs[0][1], 'SeriesDescription', '') or '')
        orient = (f'image right = {NAMES[axis_letter(r)]}; image down = {NAMES[axis_letter(c)]}; '
                  f'slices #1->#{len(idx)} go toward {NAMES[axis_letter(nrm)]}')
        d.text((8, 6), f'{slot}  ({desc[:40]})  {len(idx)} of {n} slices', fill=255, font=FONT)
        d.text((8, 34), orient, fill=255, font=FONT_S)
        fn = f'{uid.split(".")[-1]}_{slot}.jpg'; m.convert('RGB').save(f'{OUT}/{fn}', quality=90)
        entry['images'].append({'slot': slot, 'file': fn, 'orientation': orient, 'series_description': desc})
    return entry

todo = tr[~tr[T].notna().all(axis=1)]            # 非金标病例（金标行 12 列都有值）
manifest = [make_study(u, rep) for u, rep in zip(todo.StudyInstanceUID, todo.Report)]
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), ensure_ascii=False)
