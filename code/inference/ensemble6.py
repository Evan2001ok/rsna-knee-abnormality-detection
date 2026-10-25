#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RSNA: original SIX-model equal-probability ensemble, one standalone file.

直接把本文件全部放进 Kaggle 的一个 Python cell 运行，或运行同目录交付的单 cell ipynb。
需要挂载下面 MODEL_SPECS 中的六个模型 Dataset，以及比赛数据。无需训练缓存或标签 CSV。
保留核对后的前向实现；每套先 TTA/fold logit mean -> sigmoid，再六套概率等权平均。
默认精简 TTA：仅原Conv使用原图和+12度，其余五套只有原图。
精简设置尚未验证线上分数；不声称等效于原三视图配置。
模型逐套运行；每套四折由原后端分配到可用 GPU，完成后释放。
CPU 端仅缓存一套 384 图像；DINO 自行 resize392，Rad 自行 resize224/graycopy24。
断点复用只接受输入、代码、配置、权重文件标识和 CSV 校验都匹配的结果。
本文件内嵌所需全部实现，不依赖本地审计文件，也不下载预训练权重。
输出 /kaggle/working/submission.csv；没有提交 Kaggle 的网络操作。
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd

# ------------------------ 可编辑配置 ------------------------
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
OUTPUT_DIR = Path('/kaggle/working')
# /kaggle/temp 的缓存不作为 Notebook 输出保存。无需挂载训练图像缓存。
CACHE_ROOT = Path('/kaggle/temp/rsna_six_model_shared384_v1')
PREDICTION_DIR = OUTPUT_DIR / 'six_model_predictions'
RESUME = True
USE_MULTI_GPU = True
BATCH_STUDIES = 2
CPU_THREADS = 1
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
BACKBONE_CHUNK_SIZE = 0
FP16_FIRST_BATCH_CHECK = True
SAVE_FOLD_LOGITS = True
# None = 使用下方精简配置；(0.0,) = 所有模型关闭 TTA。
# 两次 A / 0.7A+0.3B 对照时，原六套应保持同一配置。
TTA_OVERRIDE = None
MODEL_TTA_ANGLES = {
    'conv': (0.0, 12.0),
    'eff': (0.0,),
    'dino': (0.0,),
    'dino3': (0.0,),
    'rad': (0.0,),
    'tiny': (0.0,),
}

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
           'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
           "Baker's", 'Contusion', 'Fracture']
MODEL_SPECS = [{'run_root': '/kaggle/input/datasets/minminandy/distillstage1conv384/rsna_convnext_tiny_384_oai8649_mix58_42_seven', 'dataset_url': 'https://www.kaggle.com/datasets/minminandy/distillstage1conv384', 'model_name': 'convnext_tiny.fb_in22k_ft_in1k_384', 'feature_dim': 768, 'model_input_size': 384, 'tokens_per_slot': 11, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'key': 'conv', 'dataset_slug': 'distillstage1conv384', 'run_name': 'rsna_convnext_tiny_384_oai8649_mix58_42_seven', 'audited_dataset_version_reference': '20697207'}, {'run_root': '/kaggle/input/datasets/minminandy/distillstage1effs384/rsna_effv2s_384_oai8649_mix58_42_seven', 'dataset_url': 'https://www.kaggle.com/datasets/minminandy/distillstage1effs384', 'model_name': 'tf_efficientnetv2_s.in21k_ft_in1k', 'feature_dim': 1280, 'model_input_size': 384, 'tokens_per_slot': 11, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'key': 'eff', 'dataset_slug': 'distillstage1effs384', 'run_name': 'rsna_effv2s_384_oai8649_mix58_42_seven', 'audited_dataset_version_reference': '20678490'}, {'run_root': '/kaggle/input/datasets/zhuyifanss/distillstage1dinov2s384/rsna_dinov2s_392_oai_mix_unfreeze9', 'dataset_url': 'https://www.kaggle.com/datasets/zhuyifanss/distillstage1dinov2s384', 'model_name': 'vit_small_patch14_dinov2.lvd142m', 'feature_dim': 384, 'model_input_size': 392, 'tokens_per_slot': 11, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'key': 'dino', 'dataset_slug': 'distillstage1dinov2s384', 'run_name': 'rsna_dinov2s_392_oai_mix_unfreeze9', 'audited_dataset_version_reference': '20712330'}, {'run_root': '/kaggle/input/datasets/lihu6666/dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a/rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr', 'dataset_url': 'https://www.kaggle.com/datasets/lihu6666/dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a', 'model_name': 'convnext_tiny.dinov3_lvd1689m', 'feature_dim': 768, 'model_input_size': 384, 'tokens_per_slot': 11, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'key': 'dino3', 'dataset_slug': 'dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a', 'run_name': 'rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr', 'audited_dataset_version_reference': '20728663'}, {'run_root': '/kaggle/input/datasets/zhuyifanss/distillstage1radimageresnet224/rsna_radimagenet50_cache384_input224_oai_graycopy24', 'dataset_url': 'https://www.kaggle.com/datasets/zhuyifanss/distillstage1radimageresnet224', 'model_name': 'radimagenet_resnet50', 'feature_dim': 2048, 'model_input_size': 224, 'tokens_per_slot': 24, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'key': 'rad', 'dataset_slug': 'distillstage1radimageresnet224', 'run_name': 'rsna_radimagenet50_cache384_input224_oai_graycopy24', 'audited_dataset_version_reference': '20708093'}, {'run_root': '/kaggle/input/datasets/minminandy/distillstage1tinyvit384/rsna_tinyvit21m384_oai8649_mix58_42_seven', 'dataset_url': 'https://www.kaggle.com/datasets/minminandy/distillstage1tinyvit384', 'model_name': 'tiny_vit_21m_384.dist_in22k_ft_in1k', 'feature_dim': 576, 'model_input_size': 384, 'tokens_per_slot': 11, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'key': 'tiny', 'dataset_slug': 'distillstage1tinyvit384', 'run_name': 'rsna_tinyvit21m384_oai8649_mix58_42_seven', 'audited_dataset_version_reference': '20697665'}]

# 下方内嵌代码分别隔离全局常量，防止 DINO392/Rad224 或不同归一化相互覆盖。
# 所有模型类、序列构造、checkpoint 校验及精度检查来自已核对的对应推理代码。
_COMMON_SOURCE = r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/minminandy/distillstage1conv384/rsna_convnext_tiny_384_oai8649_mix58_42_seven')
RUN_DIR_NAME = 'rsna_convnext_tiny_384_oai8649_mix58_42_seven'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (-12.0, 0.0, 12.0)
EXPECTED_MODEL_NAME = 'convnext_tiny.fb_in22k_ft_in1k_384'
EXPECTED_FEATURE_DIM = 768
EXPECTED_ARCH = 'shared_convnext_tiny_graycopy_5expert_labelwise_spatiotemporal_attention_v3'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
IMG_SIZE = 384
CROP_MM = 130.0
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = 2
N_TOKENS = (SLICES_PER_SLOT - TOKEN_WINDOW) // TOKEN_STRIDE + 1
IN_CHANS = GROUP_SIZE
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True

def get_devices():
    if not torch.cuda.is_available():
        return [torch.device('cpu')]
    n = torch.cuda.device_count() if USE_MULTI_GPU else 1
    return [torch.device(f'cuda:{i}') for i in range(n)]

def find_col(df, names):
    lower = {c.lower().strip(): c for c in df.columns}
    for name in names:
        key = name.lower().strip()
        if key in lower:
            return lower[key]
    raise KeyError(f'找不到列 {names}；实际 columns = {list(df.columns)}')

def to_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(int(x))
    return str(x).strip().lower() in {'1', 'true', 'yes', 'y', 't'}

def normalize_plane(x):
    s = str(x).strip().lower()
    if 'sag' in s:
        return 'Sagittal'
    if 'cor' in s:
        return 'Coronal'
    if 'axi' in s or 'tra' in s:
        return 'Axial'
    return None

def _fmt(v, spec='.5f'):
    if v is None:
        return 'NA'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return 'NaN' if not np.isfinite(f) else format(f, spec)

def resolve_checkpoints(run_dir):
    found, missing = ([], [])
    for fold in FOLDS:
        p = run_dir / f'fold{fold}' / CKPT_NAME
        (found if p.exists() else missing).append((fold, p))
    if missing:
        lines = '\n'.join((f'  fold{f}: {p}' for f, p in missing))
        avail = sorted((str(p) for p in run_dir.glob('fold*/*.pt'))) if run_dir.exists() else []
        raise FileNotFoundError(f'以下 fold 的 checkpoint 不存在:\n{lines}\n\nrun_dir = {run_dir}\n实际存在的 .pt:\n' + ('\n'.join((f'  {p}' for p in avail)) or '  (无)'))
    return found

def load_series_csv(csv_path):
    df = pd.read_csv(csv_path, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    out = pd.DataFrame({'StudyInstanceUID': df[find_col(df, COLUMN_ALIASES['study'])].astype(str), 'SeriesInstanceUID': df[find_col(df, COLUMN_ALIASES['series'])].astype(str), 'plane': df[find_col(df, COLUMN_ALIASES['plane'])].map(normalize_plane), 'fluid': df[find_col(df, COLUMN_ALIASES['fluid'])].map(to_bool), 'fatsat': df[find_col(df, COLUMN_ALIASES['fatsat'])].map(to_bool)})
    return out.dropna(subset=['plane']).reset_index(drop=True)

def dicom_files(root, study_uid, series_uid):
    folder = root / str(study_uid) / str(series_uid)
    return sorted(folder.glob('*.dcm')) if folder.exists() else []
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

def _physical_slice_normal(iop, plane):
    normal = np.cross(iop[:3], iop[3:6])
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-08:
        return None
    normal = normal / norm
    plane_axis = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
    axis = plane_axis.get(plane, int(np.argmax(np.abs(normal))))
    if normal[axis] < 0:
        normal = -normal
    return normal

def sorted_dicom_files(root, study_uid, series_uid, plane):
    files = dicom_files(root, study_uid, series_uid)
    if len(files) < 2:
        return (files, False)
    records = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=True, specific_tags=_ORDER_TAGS)
            n = getattr(ds, 'InstanceNumber', None)
            instance = float(n) if n is not None else float('inf')
            ipp_raw = getattr(ds, 'ImagePositionPatient', None)
            iop_raw = getattr(ds, 'ImageOrientationPatient', None)
            ipp = np.asarray([float(v) for v in ipp_raw], dtype=np.float64) if ipp_raw is not None and len(ipp_raw) >= 3 else None
            iop = np.asarray([float(v) for v in iop_raw], dtype=np.float64) if iop_raw is not None and len(iop_raw) >= 6 else None
            records.append((p, instance, ipp, iop))
        except Exception:
            records.append((p, float('inf'), None, None))

    def instance_fallback():
        return [rec[0] for rec in sorted(records, key=lambda r: (r[1], r[0].name))]
    if any((ipp is None or iop is None for _, _, ipp, iop in records)):
        return (instance_fallback(), False)
    normal = _physical_slice_normal(records[0][3], plane)
    if normal is None:
        return (instance_fallback(), False)
    positioned = []
    for p, instance, ipp, _iop in records:
        position = float(np.dot(ipp, normal))
        if not np.isfinite(position):
            return (instance_fallback(), False)
        positioned.append((position, instance, p.name, p))
    return ([p for _, _, _, p in sorted(positioned)], True)

def probe_series(root, row):
    study, series = (str(row.StudyInstanceUID), str(row.SeriesInstanceUID))
    files = dicom_files(root, study, series)
    out = {'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'n_slices': len(files), 'px': np.nan, 'laterality': None, 'x_pos': np.nan, 'x_raw': np.nan, 'slice_dir': 1}
    if not files:
        return out
    try:
        ds = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True, force=True)
        spacing = getattr(ds, 'PixelSpacing', None)
        if spacing is not None:
            out['px'] = float(spacing[0])
        side = getattr(ds, 'ImageLaterality', None) or getattr(ds, 'Laterality', None)
        if side:
            side = str(side).strip().upper()
            if side and side[0] in ('L', 'R'):
                out['laterality'] = side[0]
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp is not None:
            ipp_v = np.array([float(v) for v in ipp])
            out['x_raw'] = float(ipp_v[0])
            try:
                iop_m = np.array([float(v) for v in ds.ImageOrientationPatient]).reshape(2, 3)
                ps = [float(v) for v in ds.PixelSpacing]
                center = ipp_v + iop_m[0] * ps[1] * (int(ds.Columns) - 1) / 2.0 + iop_m[1] * ps[0] * (int(ds.Rows) - 1) / 2.0
                out['x_pos'] = float(center[0])
            except Exception:
                pass
        if len(files) >= 3:
            iop = [float(v) for v in ds.ImageOrientationPatient]
            nv = np.cross(iop[:3], iop[3:])
            nv = nv / (np.linalg.norm(nv) + 1e-09)
            proj, inst = ([], [])
            for f in (files[0], files[-1]):
                d2 = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                proj.append(float(np.dot([float(v) for v in d2.ImagePositionPatient], nv)))
                inst.append(float(d2.InstanceNumber))
            d_inst = inst[1] - inst[0]
            if d_inst != 0:
                out['slice_dir'] = int(np.sign((proj[1] - proj[0]) * np.sign(d_inst)))
    except Exception:
        pass
    return out

def add_dicom_header_info(root, series_df, tag=''):
    rows = list(series_df.itertuples(index=False))
    print(f'读取 {len(rows)} 个 {tag}series 的 DICOM header...')
    with ThreadPoolExecutor(max_workers=HEADER_WORKERS) as pool:
        info = list(tqdm(pool.map(lambda r: probe_series(root, r), rows), total=len(rows), desc=f'{tag}DICOM headers', unit='series', mininterval=0.5))
    merged = series_df.merge(pd.DataFrame(info), on=['StudyInstanceUID', 'SeriesInstanceUID'], how='left')
    merged['n_slices'] = pd.to_numeric(merged['n_slices'], errors='coerce').fillna(0).astype(int)
    merged['slice_dir'] = pd.to_numeric(merged['slice_dir'], errors='coerce').fillna(1).astype(int)
    n_side = merged['laterality'].notna().sum()
    print(f'laterality tag (series 级): {n_side}/{len(merged)} ({n_side / max(len(merged), 1):.1%})  <- 兜底前')
    return merged[merged['n_slices'] > 0].reset_index(drop=True)

def resolve_laterality(series_df):
    out = {}
    n_tag = n_ctr = n_sag = n_none = 0

    def side_of(x):
        if not np.isfinite(x) or abs(x) < LAT_MIN_OFFSET_MM:
            return None
        return 'R' if x < 0 else 'L'
    for study, g in series_df.groupby('StudyInstanceUID'):
        tag = g['laterality'].dropna()
        if len(tag):
            out[str(study)] = str(tag.iloc[0])
            n_tag += 1
            continue
        xs = g['x_pos'].dropna()
        if len(xs):
            s = side_of(float(xs.median()))
            if s is not None:
                out[str(study)] = s
                n_ctr += 1
                continue
        xr = g.loc[g['plane'] == 'Sagittal', 'x_raw'].dropna()
        if len(xr):
            s = side_of(float(xr.median()))
            if s is not None:
                out[str(study)] = s
                n_sag += 1
                continue
        out[str(study)] = None
        n_none += 1
    total = max(len(out), 1)
    print(f'laterality (study 级): tag {n_tag} ({100 * n_tag / total:.1f}%) | 中心x {n_ctr} ({100 * n_ctr / total:.1f}%) | 矢状兜底 {n_sag} ({100 * n_sag / total:.1f}%) | 判不出 {n_none} ({100 * n_none / total:.1f}%) -> 覆盖 {100 * (total - n_none) / total:.1f}%  (训练端 {100 * TRAIN_LAT_COVERAGE:.1f}%)')
    return out

def attach_side(study_ids, slot_map, side_map, slot_index):
    jobs = []
    for i, sid in enumerate(study_ids):
        for name, rec in slot_map.get(str(sid), {}).items():
            rec = dict(rec)
            rec['laterality'] = side_map.get(str(sid))
            jobs.append((i, slot_index[name], rec))
    return jobs

def pick_slots(series_df):
    result = {}
    for study, g in series_df.groupby('StudyInstanceUID'):
        chosen = {}
        for name, plane, fluid, fatsat in SLOTS:
            cand = g[(g['plane'] == plane) & (g['fluid'] == fluid) & (g['fatsat'] == fatsat)]
            if len(cand) == 0 and fluid is False:
                cand = g[(g['plane'] == plane) & ~g['fatsat']]
            if len(cand):
                chosen[name] = cand.sort_values('n_slices', ascending=False).iloc[0].to_dict()
        result[str(study)] = chosen
    return result

def _sample_indices(n, k):
    f = float(CENTRAL_FRACTION)
    lo = int((1.0 - f) / 2.0 * (n - 1))
    hi = int((1.0 + f) / 2.0 * (n - 1))
    if hi - lo >= k - 1:
        return np.linspace(lo, hi, k).astype(int)
    span = min(n - 1, k - 1)
    lo = (n - 1 - span) // 2
    return np.linspace(lo, lo + span, k).astype(int)

def _read_volume(files, idx):
    slices = []
    for i in idx:
        try:
            ds = pydicom.dcmread(str(files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
            intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            slices.append(a * slope + intercept)
        except Exception:
            return None
    if any((s.shape != slices[0].shape for s in slices)):
        return None
    return np.stack(slices)

def _crop_mm(vol, px, crop_mm):
    try:
        px = float(px)
    except Exception:
        px = float('nan')
    h, w = vol.shape[1:]
    if not (np.isfinite(px) and px > 0):
        s = min(h, w)
        return (vol[:, h // 2 - s // 2:h // 2 + s // 2, w // 2 - s // 2:w // 2 + s // 2], None)
    want = int(round(crop_mm / px))
    if want <= 16:
        return (vol, None)
    cy, cx, half = (h // 2, w // 2, want // 2)
    return (vol[:, max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)], want)

def _to_uint8(vol, img_size, pad_to=None):
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / max(float(p99 - p1), 1e-06), 0, 1)
    if pad_to is not None and vol.shape[1:] != (pad_to, pad_to):
        k, h, w = vol.shape
        out = np.zeros((k, pad_to, pad_to), dtype=vol.dtype)
        out[:, (pad_to - h) // 2:(pad_to - h) // 2 + h, (pad_to - w) // 2:(pad_to - w) // 2 + w] = vol
        vol = out
    x = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False, antialias=True)
    return x.squeeze(0).mul(255).round().clamp(0, 255).to(torch.uint8)

def _canonicalize_side(x, side, plane, physical_ordered, slice_dir=1):
    if plane == 'Sagittal':
        if physical_ordered:
            flip = side == 'R'
        else:
            flip = (int(slice_dir) > 0) ^ (side == 'R')
        return torch.flip(x, dims=[0]) if flip else x
    if side == 'R' and plane in ('Coronal', 'Axial'):
        return torch.flip(x, dims=[-1])
    return x

def read_slot(root, rec):
    files, physical_ordered = sorted_dicom_files(root, str(rec['StudyInstanceUID']), str(rec['SeriesInstanceUID']), rec.get('plane'))
    if not files:
        return None
    vol = _read_volume(files, _sample_indices(len(files), SLICES_PER_SLOT))
    if vol is None:
        return None
    vol, want = _crop_mm(vol, rec.get('px', np.nan), CROP_MM)
    x = _to_uint8(vol, IMG_SIZE, want)
    x = _canonicalize_side(x, rec.get('laterality'), rec.get('plane'), physical_ordered, rec.get('slice_dir', 1))
    return x.numpy()

def get_test_study_ids(series_df):
    for path in (SAMPLE_SUBMISSION_CSV, TEST_CSV):
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        try:
            uid_col = find_col(df, COLUMN_ALIASES['study'])
        except KeyError:
            continue
        ids = df[uid_col].astype(str).drop_duplicates().to_numpy()
        if len(ids):
            return ids
    return series_df['StudyInstanceUID'].astype(str).drop_duplicates().sort_values().to_numpy()

def cache_self_check(cache, mask):
    filled = mask > 0.5
    if not filled.any():
        print('自检: 没有任何 slot 被填充，跳过。')
        return
    n_zero = 0
    for i in range(cache.shape[0]):
        for k in range(cache.shape[1]):
            if filled[i, k] and int(np.asarray(cache[i, k]).max()) == 0:
                n_zero += 1
    print(f'自检: mask=1 但内容全零的 slot {n_zero} 个 (应为 0)')
    first = np.asarray(cache[:, :, 0])[filled].mean()
    mid = np.asarray(cache[:, :, SLICES_PER_SLOT // 2])[filled].mean()
    last = np.asarray(cache[:, :, -1])[filled].mean()
    flag = '' if mid > first and mid > last else '   <-- 异常，中层不是最亮'
    ref = f'(训练端 {TRAIN_BRIGHTNESS[0]} / {TRAIN_BRIGHTNESS[1]} / {TRAIN_BRIGHTNESS[2]})' if TRAIN_BRIGHTNESS is not None else '(本版无参考值，只看中层是否最亮)'
    print(f'自检: 平均亮度 首层 {first:.1f} / 中层 {mid:.1f} / 末层 {last:.1f} {ref}{flag}')

def _report_sampling(jobs):
    short, all_unique = ([], [])
    for _, _, r in jobs:
        n = int(r.get('n_slices', 0) or 0)
        if n <= 0:
            continue
        u = len(np.unique(_sample_indices(n, SLICES_PER_SLOT)))
        all_unique.append(u)
        if u < SLICES_PER_SLOT:
            f = float(CENTRAL_FRACTION)
            lo = int((1.0 - f) / 2.0 * (n - 1))
            hi = int((1.0 + f) / 2.0 * (n - 1))
            old = len(np.unique(np.linspace(lo, hi, SLICES_PER_SLOT).astype(int)))
            short.append((u, old))
    mean_u = float(np.mean(all_unique)) if all_unique else float('nan')
    print(f'取样: 平均唯一层 {mean_u:.2f} / {SLICES_PER_SLOT} (训练端 part0 {TRAIN_SAMPLING_MEAN_UNIQUE} / {SLICES_PER_SLOT})')
    if not short:
        print(f'取样: 所有 selected series 都取满 {SLICES_PER_SLOT} 个唯一层')
        return
    print(f'取样: {len(short)} 条 selected series 仍取不满 {SLICES_PER_SLOT} 个唯一层 (中位唯一层 {int(np.median([u for u, _ in short]))}，不放宽时为 {int(np.median([o for _, o in short]))})')

def build_test_cache():
    series_df = add_dicom_header_info(TEST_DICOM_DIR, load_series_csv(TEST_SERIES_CSV), tag='test ')
    study_ids = get_test_study_ids(series_df)
    side_map = resolve_laterality(series_df)
    slot_map = pick_slots(series_df)
    slot_index = {name: i for i, (name, *_) in enumerate(SLOTS)}
    shape = (len(study_ids), N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
    gib = np.prod(shape) / 1024 ** 3
    print(f'test cache shape: {shape} ({gib:.2f} GiB uint8)')
    if gib > 15.0:
        print(f'  !! test cache 约 {gib:.1f} GiB，注意 /kaggle/working 的配额')
    cache = np.lib.format.open_memmap(TEST_CACHE_PATH, mode='w+', dtype=np.uint8, shape=shape)
    mask = np.zeros((len(study_ids), N_SLOTS), dtype=np.float32)
    jobs = attach_side(study_ids, slot_map, side_map, slot_index)
    print(f'准备读取 {len(jobs)} 个 selected test series...')
    n_sag = sum((1 for _, _, r in jobs if r.get('plane') == 'Sagittal'))
    n_sag_flip = sum((1 for _, _, r in jobs if r.get('plane') == 'Sagittal' and r.get('laterality') == 'R'))
    print(f'矢状位 {n_sag} 条，其中需翻转层序 {n_sag_flip} 条 ({100 * n_sag_flip / max(n_sag, 1):.1f}%)  (训练端 {100 * TRAIN_SAG_FLIP_RATE:.1f}%)')
    _report_sampling(jobs)

    def worker(job):
        i, k, rec = job
        return (i, k, read_slot(TEST_DICOM_DIR, rec))
    with ThreadPoolExecutor(max_workers=PIXEL_WORKERS) as pool:
        for i, k, x in tqdm(pool.map(worker, jobs), total=len(jobs), desc='Test DICOM -> cache', unit='series', mininterval=0.5):
            if x is not None:
                cache[i, k] = x
                mask[i, k] = 1.0
    cache.flush()
    np.save(TEST_MASK_PATH, mask)
    np.save(TEST_IDS_PATH, study_ids)
    print('各 slot 填充率 (test vs train):')
    n_study = max(len(study_ids), 1)
    for i, (name, *_) in enumerate(SLOTS):
        got = float(mask[:, i].mean())
        ref = TRAIN_SLOT_FILL[name]
        flag = ''
        if abs(got - ref) > 0.15 and n_study >= 50:
            flag = '   <-- 差异 >15pp，查 pick_slots'
        print(f'  {name:14s} {got:6.1%}   (train {ref:5.1%}){flag}')
    if n_study < 50:
        print(f'  (test 只有 {n_study} 个 study，填充率差异无统计意义，不告警)')
    cache_self_check(cache, mask)
    del cache
    return (np.load(TEST_CACHE_PATH, mmap_mode='r'), mask, study_ids)

def load_or_build_test_cache():
    if TEST_CACHE_PATH.exists() and TEST_MASK_PATH.exists() and TEST_IDS_PATH.exists():
        print('发现已有 test cache，mmap 读取，跳过 DICOM preprocessing。')
        print(f'  {TEST_CACHE_DIR}')
        print('  !! 形状检查查不出预处理版本差异 —— 改过 read_slot / laterality / _sample_indices 就必须删掉这个目录重建。')
        print('     (本版预处理端一字未改，沿用上一版 cache 是对的。)')
        cache = np.load(TEST_CACHE_PATH, mmap_mode='r')
        want = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if tuple(cache.shape[1:]) != want:
            raise RuntimeError(f'test cache 形状 {tuple(cache.shape[1:])} 与当前配置 {want} 不符 —— 删掉 {TEST_CACHE_DIR} 重新构建')
        mask = np.load(TEST_MASK_PATH)
        study_ids = np.load(TEST_IDS_PATH, allow_pickle=True).astype(str)
        if mask.shape != (len(cache), N_SLOTS):
            raise RuntimeError(f'test mask shape {mask.shape} 与预期 {(len(cache), N_SLOTS)} 不符 —— 删掉 {TEST_CACHE_DIR} 重新构建')
        if len(study_ids) != len(cache):
            raise RuntimeError(f'test ids {len(study_ids)} 与 cache {len(cache)} 长度不一致 —— 删掉 {TEST_CACHE_DIR} 重新构建')
        return (cache, mask, study_ids)
    print('没有 test cache，从 test DICOM 构建。')
    return build_test_cache()

def check_mask(mask, study_ids):
    n_slot = mask.sum(axis=1)
    empty = np.flatnonzero(n_slot == 0)
    if len(empty):
        print(f'\n!! WARNING: {len(empty)} 个 study 没有任何 slot，预测等于 fusion_bias')
        for sid in np.asarray(study_ids)[empty][:10]:
            print(f'     {sid}')
    print(f'mean present slots/study: {float(n_slot.mean()):.3f} (min {int(n_slot.min())}, max {int(n_slot.max())})')
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug',)
'''

_BACKEND_SOURCES = {
'conv': r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/minminandy/distillstage1conv384/rsna_convnext_tiny_384_oai8649_mix58_42_seven')
RUN_DIR_NAME = 'rsna_convnext_tiny_384_oai8649_mix58_42_seven'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (-12.0, 0.0, 12.0)
EXPECTED_MODEL_NAME = 'convnext_tiny.fb_in22k_ft_in1k_384'
EXPECTED_FEATURE_DIM = 768
EXPECTED_ARCH = 'shared_convnext_tiny_graycopy_5expert_labelwise_spatiotemporal_attention_v3'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
IMG_SIZE = 384
CROP_MM = 130.0
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = 2
N_TOKENS = (SLICES_PER_SLOT - TOKEN_WINDOW) // TOKEN_STRIDE + 1
IN_CHANS = GROUP_SIZE
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

class TestStudyDataset(Dataset):

    def __init__(self, cache, mask):
        self.cache = cache
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, i):
        imgs = torch.from_numpy(np.array(self.cache[i]))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i])).float()
        return (imgs, m)

def rotate_grayscale_batch(x, angle_deg):
    if float(angle_deg) == 0.0:
        return x
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation TTA expects square [N,1,H,W] grayscale images.')
    with torch.autocast(device_type=x.device.type, enabled=False):
        radians = math.radians(float(angle_deg))
        c, s = (math.cos(radians), math.sin(radians))
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=x.device).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class KneeSharedConvNeXtNet(nn.Module):

    def __init__(self, model_name: str, gru_hidden=256, gru_layers=1, attn_hidden=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=IN_CHANS, drop_path_rate=0.0)
        self.feature_dim = int(self.backbone.num_features)
        self.seq_dim = gru_hidden * 2
        self.fixed_fp16_backbone = False
        self.grus = nn.ModuleList([nn.GRU(input_size=self.feature_dim, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True) for _ in range(N_SLOTS)])
        self.spatial_attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.seq_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.expert_heads = nn.ModuleList([nn.Linear(self.seq_dim, len(TARGETS)) for _ in range(N_SLOTS)])
        self.fusion_gate_logits = nn.Parameter(torch.zeros(len(TARGETS), N_SLOTS, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))
        shape = (1, IN_CHANS, 1, 1)
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float16 if self.fixed_fp16_backbone else torch.float32)
            c = int(BACKBONE_CHUNK_SIZE)
            if c <= 0 or x.shape[0] <= c:
                feat = self.backbone.forward_features(x)
            else:
                parts = [self.backbone.forward_features(x[i:i + c]) for i in range(0, x.shape[0], c)]
                feat = torch.cat(parts, dim=0)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat) or feat.ndim != 4:
            raise RuntimeError(f"backbone forward_features 预期返回4维空间图，实际类型/shape: {type(feat)} / {getattr(feat, 'shape', None)}")
        if feat.shape[1] == self.feature_dim:
            pass
        elif feat.shape[-1] == self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f'backbone 空间图找不到 feature_dim={self.feature_dim} 的通道轴: {tuple(feat.shape)}')
        return feat

    def _apply_backbone_head(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3 or feat.shape[-1] != self.feature_dim:
            raise RuntimeError(f'backbone head 预期 [N,T,{self.feature_dim}]，得到 {tuple(feat.shape)}')
        n, t, d = feat.shape
        x = feat.reshape(n * t, d, 1, 1)
        try:
            x = self.backbone.forward_head(x, pre_logits=True)
        except TypeError:
            x = self.backbone.head(x, pre_logits=True)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[-1] != d:
            raise RuntimeError(f"backbone head 预期返回 [{n * t},{d}]，实际 {type(x)} / {getattr(x, 'shape', None)}")
        return x.reshape(n, t, d)

    def _encode_slot(self, slot: int, imgs: torch.Tensor, valid_idx: torch.Tensor, rotation_deg: float=0.0) -> torch.Tensor:
        G, C, R = (N_TOKENS, TOKEN_WINDOW, IMG_SIZE)
        x = imgs[:, slot].index_select(0, valid_idx)
        bv = x.shape[0]
        if rotation_deg != 0.0:
            x = rotate_grayscale_batch(x.reshape(bv * SLICES_PER_SLOT, 1, R, R), rotation_deg).reshape(bv, SLICES_PER_SLOT, R, R)
        x = x.unfold(1, C, TOKEN_STRIDE)
        if x.shape[1] != G:
            raise RuntimeError(f'unfold 得到 {x.shape[1]} 个 token，预期 {G} —— SLICES_PER_SLOT/TOKEN_WINDOW/TOKEN_STRIDE 配置不一致')
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(bv * G, C, R, R).float().div(255.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        feat_map = self._run_backbone_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float().flatten(2).transpose(1, 2)
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)
            label_tokens = torch.einsum('npt,npd->ntd', spatial_a, pixels)
            label_tokens = self._apply_backbone_head(label_tokens)
            label_tokens = label_tokens.reshape(bv, G, len(TARGETS), self.feature_dim).permute(0, 2, 1, 3).contiguous()
            h = label_tokens.reshape(bv * len(TARGETS), G, self.feature_dim)
            h, _ = self.grus[slot](h)
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)
            temporal_all = self.attentions[slot](h)
            temporal_a = temporal_all.diagonal(dim1=1, dim2=3).permute(0, 2, 1).softmax(dim=-1)
            label_feat = torch.einsum('btg,btgd->btd', temporal_a, h)
            head = self.expert_heads[slot]
            return torch.einsum('btd,td->bt', label_feat, head.weight) + head.bias

    def forward(self, imgs, mask, *, valid_indices=None, return_expert_logits: bool=False, rotation_deg: float=0.0):
        B = imgs.shape[0]
        expected = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(f'期望 imgs [B,{N_SLOTS},{SLICES_PER_SLOT},{IMG_SIZE},{IMG_SIZE}]，得到 {tuple(imgs.shape)}')
        available = mask > 0.5
        per_slot = []
        for s in range(N_SLOTS):
            valid_idx = valid_indices[s] if valid_indices is not None else torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=torch.float32)
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx, rotation_deg=rotation_deg)
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)
        expert_logits = torch.stack(per_slot, dim=1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -10000.0)
            gate = gate.softmax(dim=-1)
            logits = (gate * expert_logits.float().permute(0, 2, 1)).sum(dim=-1) + self.fusion_bias.float()
        if return_expert_logits:
            return (logits, expert_logits)
        return logits
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug',)

def _check_sequence_layout(ckpt, fold):
    for key, want in (('n_groups', N_GROUPS), ('group_size', GROUP_SIZE), ('sequence_tokens_per_slot', N_TOKENS)):
        got = ckpt.get(key)
        if got is None:
            print(f'WARNING: fold{fold} ckpt 缺少 {key}，跳过该项校验')
            continue
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符' + (f'\n  (sequence_tokens_per_slot=24 说明这是上一版 ConvNeXt graycopy24 的 ckpt，不能用这份推理代码)' if key == 'sequence_tokens_per_slot' and int(got) == SLICES_PER_SLOT else ''))
    sps = ckpt.get('slices_per_slot')
    if sps is not None and int(sps) != N_GROUPS * GROUP_SIZE:
        raise ValueError(f'fold{fold} slices_per_slot={sps} != n_groups*group_size={N_GROUPS * GROUP_SIZE}')
    mode = ckpt.get('input_mode')
    if mode is None:
        raise ValueError(f'fold{fold} ckpt 没有 input_mode 字段，无法确认输入构造。')
    if mode != '3slice_2p5d_overlap_window3_stride2':
        raise ValueError(f'fold{fold} input_mode={mode!r}，与本文件实现的「3 层 2.5D 重叠三元组 (窗口 3 / 步长 2)」不符 —— single_grayscale_slice_repeated_to_3_channels 是上一版 graycopy24。')
    sa = ckpt.get('spatial_attention')
    if sa is None:
        raise ValueError(f'fold{fold} ckpt 没有 spatial_attention 字段，多半是只有时序注意力的旧版产物；本文件多一组 spatial_attentions.* 权重，strict=True 会报 missing keys。')
    if sa != 'per_slot_per_label_softmax_over_final_convnext_spatial_map':
        raise ValueError(f'fold{fold} spatial_attention={sa!r}，与本文件实现的 label-wise 空间注意力不符。')
    ta = ckpt.get('temporal_attention')
    if ta is None:
        raise ValueError(f'fold{fold} ckpt 没有 temporal_attention 字段，多半是(所有标签共用一套注意力) 的旧版产物，前向不同，不能用这份推理代码。')
    if ta != 'per_slot_per_label_softmax_over_6_tokens':
        raise ValueError(f'fold{fold} temporal_attention={ta!r}，与本文件实现的 label-wise 注意力不符。')
    vs = ckpt.get('val_sequence_strategy')
    if vs is None:
        if ckpt.get('val_group_strategy') is not None:
            raise ValueError(f"fold{fold} 只有旧版的 val_group_strategy={ckpt.get('val_group_strategy')!r}，不是本版训练产物。")
        print(f'WARNING: fold{fold} 无 val_sequence_strategy，跳过验证端策略校验')
    elif vs not in VAL_SEQ_STRATEGIES:
        raise ValueError(f'fold{fold} val_sequence_strategy={vs!r} 不在已知列表 {VAL_SEQ_STRATEGIES}；验证端前向可能与本文件不同，不要混用。')

def load_fold_models(devices, run_dir):
    models = []
    ref_name = None
    fusion_stack = []
    for k, (fold, path) in enumerate(resolve_checkpoints(run_dir)):
        dev = devices[k % len(devices)]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in ckpt:
            raise KeyError(f'{path} 中找不到 state_dict')
        arch = ckpt.get('architecture')
        if EXPECTED_ARCH is not None and arch != EXPECTED_ARCH:
            raise ValueError(f'fold{fold} architecture={arch!r}，但本文件实现的是 {EXPECTED_ARCH!r}。\nrun_dir 可能挂错了 run。\nrun_dir = {run_dir}')
        if ckpt.get('independent_encoder_per_slot') is True:
            raise ValueError(f'fold{fold} 的 ckpt 是 5 独立 encoder 版本，不能用这份共享 backbone 的推理代码。')
        if ckpt.get('shared_backbone_across_slots') is not True:
            raise ValueError(f'fold{fold} 的 ckpt 没有 shared_backbone_across_slots=True，不是共享 backbone 版本的训练产物。')

        def need(key, want, cast=int, tol=None):
            got = ckpt.get(key)
            if got is None:
                return
            got = cast(got)
            ok = abs(got - want) <= tol if tol is not None else got == want
            if not ok:
                raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
        need('img_size', IMG_SIZE)
        need('n_slots', N_SLOTS)
        need('slices_per_slot', SLICES_PER_SLOT)
        need('crop_mm', CROP_MM, cast=float, tol=1.0)
        ckpt_targets = ckpt.get('targets')
        if ckpt_targets is not None and list(ckpt_targets) != TARGETS:
            raise ValueError(f'fold{fold} targets 与当前 TARGETS 不一致。\ncheckpoint: {ckpt_targets}\ncurrent   : {TARGETS}')
        ckpt_slot_names = ckpt.get('slot_names')
        expected_slot_names = [s[0] for s in SLOTS]
        if ckpt_slot_names is not None and list(ckpt_slot_names) != expected_slot_names:
            raise ValueError(f'fold{fold} slot_names 与推理端不一致。\ncheckpoint: {list(ckpt_slot_names)}\ncurrent   : {expected_slot_names}')
        if ckpt_slot_names is None and k == 0:
            print('  (ckpt 无 slot_names 字段，跳过 slot 顺序校验 —— SLOTS 必须与预处理端逐字一致，时序专家是逐 slot 的)')
        _check_sequence_layout(ckpt, fold)
        slice_order = ckpt.get('slice_order')
        if slice_order is not None and slice_order != 'instance_number':
            print(f'WARNING: fold{fold} slice_order={slice_order!r}，与推理端预期不一致!')
        model_name = ckpt.get('model_name')
        if model_name is None:
            raise KeyError(f'{path} 里没有 model_name，多半是旧版 checkpoint，与当前训练代码不兼容。')
        if EXPECTED_MODEL_NAME is not None and model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f'fold{fold} backbone={model_name}，但 EXPECTED_MODEL_NAME={EXPECTED_MODEL_NAME}。\n!! 本版与上一版 ConvNeXt graycopy24 的 architecture 字符串相同，这道校验是分辨两者的唯一干净手段 —— 不要为了跑通而放宽它。\nrun_dir = {run_dir}')
        if ref_name is None:
            ref_name = model_name
        elif model_name != ref_name:
            raise ValueError(f'fold{fold} backbone ({model_name}) 与 fold{FOLDS[0]} ({ref_name}) 不同')
        model = KneeSharedConvNeXtNet(model_name, gru_hidden=int(ckpt.get('gru_hidden', 256)), gru_layers=int(ckpt.get('gru_layers', 1)), attn_hidden=int(ckpt.get('attn_hidden', 128)))
        ckpt_fd = ckpt.get('feature_dim')
        if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
            raise ValueError(f'fold{fold} feature_dim={ckpt_fd}，实例化得到 {model.feature_dim}')
        ckpt_sq = ckpt.get('seq_dim')
        if ckpt_sq is not None and int(ckpt_sq) != model.seq_dim:
            raise ValueError(f'fold{fold} seq_dim={ckpt_sq}，实例化得到 {model.seq_dim}')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        ref_fusion = ckpt.get('fusion_weights_no_missing')
        got_fusion = model.fusion_weights().detach()
        if ref_fusion is not None:
            d = float((torch.as_tensor(ref_fusion).float() - got_fusion).abs().max())
            if d > 0.0001:
                raise RuntimeError(f'fold{fold} 融合权重与 ckpt 存的 fusion_weights_no_missing 不一致 (max|Δ|={d:.2e}) —— gate 参数没有正确载入。')
        fusion_stack.append(got_fusion.cpu().numpy())
        models.append((fold, model.to(dev, memory_format=torch.channels_last).eval(), dev))
        print(f"  fold{fold} -> {dev} | {path.name} | epoch {ckpt.get('epoch', 'NA')} | derived {_fmt(ckpt.get('derived_auc'))} | gold {_fmt(ckpt.get('gold_auc'))} | select {_fmt(ckpt.get('selection_score'))} ({ckpt.get('selection', 'NA')}, n_gold_val={ckpt.get('n_gold_val', 'NA')}) | EMA={ckpt.get('ema', 'NA')}(decay={ckpt.get('ema_decay', 'NA')}) | aux_w={ckpt.get('expert_aux_weight', 'NA')}")
        if ckpt.get('aug'):
            print(f"         train aug: {ckpt['aug']}")
        if ckpt.get('label_mix') and k == 0:
            print(f"         label mix: {ckpt['label_mix']}")
    ref_model = models[0][1]
    mean = [round(v, 4) for v in ref_model.mean.flatten().tolist()]
    std = [round(v, 4) for v in ref_model.std.flatten().tolist()]
    print(f'\nnormalize: mean {mean} | std {std}')
    if all((v == 0.0 for v in mean)) or all((v == 1.0 for v in std)):
        raise RuntimeError('mean/std 仍是初始占位值，checkpoint 里没有这两个 buffer！')
    n_param = sum((p.numel() for p in ref_model.parameters()))
    print(f'backbone: {ref_name} x1 (shared over {N_SLOTS} slots) | feature_dim {ref_model.feature_dim} | seq_dim {ref_model.seq_dim} | {n_param / 1000000.0:.1f}M params/fold | {len(models)} folds on {len(devices)} device(s)')
    if ref_model.feature_dim != EXPECTED_FEATURE_DIM:
        print(f'  !! feature_dim={ref_model.feature_dim}，ConvNeXt-Tiny 预期 {EXPECTED_FEATURE_DIM}')
    print('^^ 请把上面的 epoch / derived / gold 与训练日志逐行核对；对不上说明 run_dir 挂错了 run。')
    if PRINT_FUSION_WEIGHTS and fusion_stack:
        w = np.mean(np.stack(fusion_stack), axis=0)
        names = [s[0] for s in SLOTS]
        print('\nlabel-wise 融合权重 (无缺失 slot 时, %d fold 平均):' % len(fusion_stack))
        print('  ' + ' ' * 18 + ''.join((f'{n:>14s}' for n in names)))
        for j, t in enumerate(TARGETS):
            row = ''.join((f'{w[j, i]:14.4f}' for i in range(N_SLOTS)))
            top = names[int(np.argmax(w[j]))]
            print(f'  {t:18s}{row}   <- {top}')
        print('  (缺失 slot 在 softmax 前被屏蔽，实际权重按可用 slot 重新归一化)')
    return models

def resolve_backbone_head(backbone):
    mod = getattr(backbone, 'head', None)
    if mod is None or not isinstance(mod, nn.Module):
        return (None, None, 0)
    return ('head', mod, sum((p.numel() for p in mod.parameters())))

def clear_attention_bias_cache(module):
    n = 0
    for m in module.modules():
        if hasattr(m, 'attention_bias_cache'):
            m.attention_bias_cache = {}
            n += 1
    return n

@torch.inference_mode()
def prepare_fixed_fp16(models, loader):
    if len(loader.dataset) == 0:
        raise RuntimeError('No test studies to predict.')
    sample = next(iter(loader)) if FIXED_FP16_SANITY_CHECK else None
    cpu_indices = make_cpu_slot_indices(sample[1]) if sample is not None else None
    for fold, model, dev in models:
        if dev.type != 'cuda':
            raise RuntimeError('This fixed-FP16 inference script requires CUDA.')
        with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
            ref = None
            if sample is not None:
                print(f'[precision] fold{fold}: FP32 reference -> fixed FP16 check', flush=True)
                im = sample[0].to(dev, non_blocking=True)
                mk = sample[1].to(dev, non_blocking=True)
                ix = [v.to(dev, non_blocking=True) for v in cpu_indices]
                ref = [model(im, mk, valid_indices=ix, rotation_deg=a).float() for a in TTA_ANGLES]
                if not all((torch.isfinite(v).all().item() for v in ref)):
                    raise RuntimeError(f'fold{fold}: FP32 reference contains NaN/Inf.')
            head_name, head_mod, n_head_param = resolve_backbone_head(model.backbone)
            original_head = None
            if head_mod is not None:
                original_head = {k: v.detach().cpu().clone() for k, v in head_mod.state_dict().items()}
            model.backbone.half()
            if head_mod is not None:
                head_mod = getattr(model.backbone, head_name)
                head_mod.float()
                head_mod.load_state_dict(original_head, strict=True)
            model.fixed_fp16_backbone = True
            n_cleared = clear_attention_bias_cache(model.backbone)
            if n_cleared:
                print(f'  cleared {n_cleared} attention_bias_cache entries')
            n_bn, n_sub, n_zero = (0, 0, 0)
            for m in model.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                    n_bn += 1
                    v = m.running_var.float()
                    n_sub += int((v < 6.104e-05).sum())
                    n_zero += int((v == 0).sum())
            if n_bn:
                flag = '   <-- 看下面的对拍' if n_sub or n_zero else ''
                print(f'  BatchNorm: {n_bn} 层 | running_var 落入 fp16 次正规区 (<6.1e-5) 的通道 {n_sub} 个 | 归零 {n_zero} 个{flag}')
            head_prefix = f'backbone.{head_name}.' if head_name else None
            for name, p in model.named_parameters():
                half_feature = name.startswith('backbone.') and (not (head_prefix and name.startswith(head_prefix)))
                expected = torch.float16 if half_feature else torch.float32
                if p.is_floating_point() and p.dtype != expected:
                    raise RuntimeError(f'Unexpected dtype: {name} {p.dtype}, expected {expected}')
            if sample is not None:
                for angle, reference in zip(TTA_ANGLES, ref):
                    actual = model(im, mk, valid_indices=ix, rotation_deg=angle).float()
                    finite = torch.isfinite(actual).all().item()
                    dl = float((actual - reference).abs().max().item())
                    dp = float((actual.sigmoid() - reference.sigmoid()).abs().max().item())
                    print(f'  angle={angle:+g}: max|delta logit|={dl:.6f}, max|delta probability|={dp:.6f}', flush=True)
                    if not finite or dl > FP16_MAX_LOGIT_DIFF or dp > FP16_MAX_PROB_DIFF:
                        raise RuntimeError(f'fold{fold}, angle={angle}: FP16 check failed; logit={dl}, probability={dp}, finite={finite}')
                del im, mk, ix, ref, reference, actual
            head_txt = f'backbone.{head_name} ({n_head_param / 1000.0:.1f}K params) FP32' if head_mod is not None else 'backbone 无 head 子模块 (EffNet 的 forward_head 无参数)'
            print(f'[precision] fold{fold}: feature extractor FP16; {head_txt}; attention/GRU/fusion FP32')

def make_cpu_slot_indices(mask):
    if mask.device.type != 'cpu':
        raise ValueError('Build indices from the original CPU mask before H2D.')
    return [torch.nonzero(mask[:, s] > 0.5, as_tuple=False).flatten().pin_memory() for s in range(N_SLOTS)]

@torch.inference_mode()
def _predict_device_batch(dev, fold_models, imgs, mask, cpu_indices):
    with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
        im = imgs.to(dev, non_blocking=True)
        mk = mask.to(dev, non_blocking=True)
        indices = [v.to(dev, non_blocking=True) for v in cpu_indices]
        result = []
        for fold, model in fold_models:
            logits = None
            for angle in TTA_ANGLES:
                view_logits = model(im, mk, valid_indices=indices, rotation_deg=angle).float()
                logits = view_logits if logits is None else logits + view_logits
            logits = logits / len(TTA_ANGLES)
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f'fold{fold}: prediction contains NaN/Inf.')
            result.append((fold, values))
        return result

@torch.inference_mode()
def predict_test(models, loader):
    devices = sorted({d for _, _, d in models}, key=str)
    if any((d.type != 'cuda' for d in devices)):
        raise RuntimeError('Fixed-FP16 inference requires CUDA devices.')
    groups = {d: [(f, m) for f, m, md in models if md == d] for d in devices}
    fold_order = [f for f, _, _ in models]
    if len(set(fold_order)) != len(fold_order):
        raise RuntimeError('Duplicate fold IDs would misalign predictions.')
    per_fold = {f: [] for f in fold_order}
    start = time.perf_counter()
    n_seen = 0
    with ExitStack() as stack:
        pools = {d: stack.enter_context(ThreadPoolExecutor(max_workers=1)) for d in devices}
        for imgs, mask in tqdm(loader, total=len(loader), desc=f'Inference ({len(models)} folds / {len(devices)} GPUs)' if tuple(TTA_ANGLES) == (0.0,) else f'Rotation TTA{len(TTA_ANGLES)} ({len(models)} folds / {len(devices)} GPUs)', unit='batch', mininterval=0.5):
            cpu_indices = make_cpu_slot_indices(mask)
            futures = [pools[d].submit(_predict_device_batch, d, groups[d], imgs, mask, cpu_indices) for d in devices]
            results, error = ([], None)
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            got = dict(results)
            if len(results) != len(models) or set(got) != set(fold_order):
                raise RuntimeError('Missing/duplicate fold result in GPU dispatch.')
            for fold in fold_order:
                if got[fold].shape != (len(imgs), len(TARGETS)):
                    raise RuntimeError(f'fold{fold}: unexpected output shape {got[fold].shape}')
                per_fold[fold].append(got[fold])
            n_seen += len(imgs)
    seconds = time.perf_counter() - start
    logits = np.stack([np.concatenate(per_fold[f], axis=0) for f in fold_order])
    print(f'[timing] {n_seen} studies / {len(models)} folds: {seconds:.2f}s, {n_seen / max(seconds, 1e-09):.3f} studies/s (includes loader/H2D/D2H)')
    return (fold_order, logits)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _rank_normalize(p):
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for j in range(p.shape[1]):
        s = pd.Series(p[:, j]).rank(method='average').to_numpy()
        out[:, j] = (s - 1) / max(n - 1, 1)
    return out

def combine_folds(logits, mode):
    if mode == 'logit':
        return _sigmoid(logits.mean(axis=0))
    if mode == 'prob':
        return _sigmoid(logits).mean(axis=0)
    if mode == 'rank':
        probs = _sigmoid(logits)
        return np.stack([_rank_normalize(probs[k]) for k in range(probs.shape[0])]).mean(axis=0)
    raise ValueError(f'ENSEMBLE 必须是 logit | prob | rank，得到 {mode!r}')

def report_fold_agreement(logits, fold_order):
    n_folds, n, n_tgt = logits.shape
    if n_folds < 2 or n < 30:
        print(f'\n样本数 {n} 太少 (rank 后取值被量化成少数几档)，fold 一致性检查无意义，跳过。')
        return
    probs = _sigmoid(logits)
    ranks = [_rank_normalize(probs[k]) for k in range(n_folds)]
    print('\nfold 间 Spearman 相关 (per-target，列出最低的一对):')
    for j, t in enumerate(TARGETS):
        pairs = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                r = float(np.corrcoef(ranks[a][:, j], ranks[b][:, j])[0, 1])
                pairs.append((r, fold_order[a], fold_order[b]))
        rs = [p[0] for p in pairs]
        lo, fa, fb = min(pairs)
        flag = '  <-- 偏低' if lo < 0.7 else ''
        print(f'  {t:18s} mean {np.mean(rs):+.4f} | min {lo:+.4f} (fold{fa} vs fold{fb}){flag}')
''',
'eff': r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/minminandy/distillstage1effs384/knee_effv2s_384_oai_soft_v3_mix4')
RUN_DIR_NAME = 'knee_effv2s_384_oai_soft_v3_mix4'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (-12.0, 0.0, 12.0)
EXPECTED_MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'
EXPECTED_FEATURE_DIM = 1280
EXPECTED_ARCH = 'shared_convnext_tiny_graycopy_5expert_labelwise_spatiotemporal_attention_v3'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
IMG_SIZE = 384
CROP_MM = 130.0
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = 2
N_TOKENS = (SLICES_PER_SLOT - TOKEN_WINDOW) // TOKEN_STRIDE + 1
IN_CHANS = GROUP_SIZE
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

class TestStudyDataset(Dataset):

    def __init__(self, cache, mask):
        self.cache = cache
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, i):
        imgs = torch.from_numpy(np.array(self.cache[i]))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i])).float()
        return (imgs, m)

def rotate_grayscale_batch(x, angle_deg):
    if float(angle_deg) == 0.0:
        return x
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation TTA expects square [N,1,H,W] grayscale images.')
    with torch.autocast(device_type=x.device.type, enabled=False):
        radians = math.radians(float(angle_deg))
        c, s = (math.cos(radians), math.sin(radians))
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=x.device).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class KneeSharedConvNeXtNet(nn.Module):

    def __init__(self, model_name: str, gru_hidden=256, gru_layers=1, attn_hidden=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=IN_CHANS, drop_path_rate=0.0)
        self.feature_dim = int(self.backbone.num_features)
        self.seq_dim = gru_hidden * 2
        self.fixed_fp16_backbone = False
        self.grus = nn.ModuleList([nn.GRU(input_size=self.feature_dim, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True) for _ in range(N_SLOTS)])
        self.spatial_attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.seq_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.expert_heads = nn.ModuleList([nn.Linear(self.seq_dim, len(TARGETS)) for _ in range(N_SLOTS)])
        self.fusion_gate_logits = nn.Parameter(torch.zeros(len(TARGETS), N_SLOTS, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))
        shape = (1, IN_CHANS, 1, 1)
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float16 if self.fixed_fp16_backbone else torch.float32)
            c = int(BACKBONE_CHUNK_SIZE)
            if c <= 0 or x.shape[0] <= c:
                feat = self.backbone.forward_features(x)
            else:
                parts = [self.backbone.forward_features(x[i:i + c]) for i in range(0, x.shape[0], c)]
                feat = torch.cat(parts, dim=0)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat) or feat.ndim != 4:
            raise RuntimeError(f"backbone forward_features 预期返回4维空间图，实际类型/shape: {type(feat)} / {getattr(feat, 'shape', None)}")
        if feat.shape[1] == self.feature_dim:
            pass
        elif feat.shape[-1] == self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f'backbone 空间图找不到 feature_dim={self.feature_dim} 的通道轴: {tuple(feat.shape)}')
        return feat

    def _apply_backbone_head(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3 or feat.shape[-1] != self.feature_dim:
            raise RuntimeError(f'backbone head 预期 [N,T,{self.feature_dim}]，得到 {tuple(feat.shape)}')
        n, t, d = feat.shape
        x = feat.reshape(n * t, d, 1, 1)
        try:
            x = self.backbone.forward_head(x, pre_logits=True)
        except TypeError:
            x = self.backbone.head(x, pre_logits=True)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[-1] != d:
            raise RuntimeError(f"backbone head 预期返回 [{n * t},{d}]，实际 {type(x)} / {getattr(x, 'shape', None)}")
        return x.reshape(n, t, d)

    def _encode_slot(self, slot: int, imgs: torch.Tensor, valid_idx: torch.Tensor, rotation_deg: float=0.0) -> torch.Tensor:
        G, C, R = (N_TOKENS, TOKEN_WINDOW, IMG_SIZE)
        x = imgs[:, slot].index_select(0, valid_idx)
        bv = x.shape[0]
        if rotation_deg != 0.0:
            x = rotate_grayscale_batch(x.reshape(bv * SLICES_PER_SLOT, 1, R, R), rotation_deg).reshape(bv, SLICES_PER_SLOT, R, R)
        x = x.unfold(1, C, TOKEN_STRIDE)
        if x.shape[1] != G:
            raise RuntimeError(f'unfold 得到 {x.shape[1]} 个 token，预期 {G} —— SLICES_PER_SLOT/TOKEN_WINDOW/TOKEN_STRIDE 配置不一致')
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(bv * G, C, R, R).float().div(255.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        feat_map = self._run_backbone_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float().flatten(2).transpose(1, 2)
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)
            label_tokens = torch.einsum('npt,npd->ntd', spatial_a, pixels)
            label_tokens = self._apply_backbone_head(label_tokens)
            label_tokens = label_tokens.reshape(bv, G, len(TARGETS), self.feature_dim).permute(0, 2, 1, 3).contiguous()
            h = label_tokens.reshape(bv * len(TARGETS), G, self.feature_dim)
            h, _ = self.grus[slot](h)
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)
            temporal_all = self.attentions[slot](h)
            temporal_a = temporal_all.diagonal(dim1=1, dim2=3).permute(0, 2, 1).softmax(dim=-1)
            label_feat = torch.einsum('btg,btgd->btd', temporal_a, h)
            head = self.expert_heads[slot]
            return torch.einsum('btd,td->bt', label_feat, head.weight) + head.bias

    def forward(self, imgs, mask, *, valid_indices=None, return_expert_logits: bool=False, rotation_deg: float=0.0):
        B = imgs.shape[0]
        expected = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(f'期望 imgs [B,{N_SLOTS},{SLICES_PER_SLOT},{IMG_SIZE},{IMG_SIZE}]，得到 {tuple(imgs.shape)}')
        available = mask > 0.5
        per_slot = []
        for s in range(N_SLOTS):
            valid_idx = valid_indices[s] if valid_indices is not None else torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=torch.float32)
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx, rotation_deg=rotation_deg)
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)
        expert_logits = torch.stack(per_slot, dim=1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -10000.0)
            gate = gate.softmax(dim=-1)
            logits = (gate * expert_logits.float().permute(0, 2, 1)).sum(dim=-1) + self.fusion_bias.float()
        if return_expert_logits:
            return (logits, expert_logits)
        return logits
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug',)

def _check_sequence_layout(ckpt, fold):
    for key, want in (('n_groups', N_GROUPS), ('group_size', GROUP_SIZE), ('sequence_tokens_per_slot', N_TOKENS)):
        got = ckpt.get(key)
        if got is None:
            print(f'WARNING: fold{fold} ckpt 缺少 {key}，跳过该项校验')
            continue
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符' + (f'\n  (sequence_tokens_per_slot=24 说明这是上一版 ConvNeXt graycopy24 的 ckpt，不能用这份推理代码)' if key == 'sequence_tokens_per_slot' and int(got) == SLICES_PER_SLOT else ''))
    sps = ckpt.get('slices_per_slot')
    if sps is not None and int(sps) != N_GROUPS * GROUP_SIZE:
        raise ValueError(f'fold{fold} slices_per_slot={sps} != n_groups*group_size={N_GROUPS * GROUP_SIZE}')
    mode = ckpt.get('input_mode')
    if mode is None:
        raise ValueError(f'fold{fold} ckpt 没有 input_mode 字段，无法确认输入构造。')
    if mode != '3slice_2p5d_overlap_window3_stride2':
        raise ValueError(f'fold{fold} input_mode={mode!r}，与本文件实现的「3 层 2.5D 重叠三元组 (窗口 3 / 步长 2)」不符 —— single_grayscale_slice_repeated_to_3_channels 是上一版 graycopy24。')
    sa = ckpt.get('spatial_attention')
    if sa is None:
        raise ValueError(f'fold{fold} ckpt 没有 spatial_attention 字段，多半是只有时序注意力的旧版产物；本文件多一组 spatial_attentions.* 权重，strict=True 会报 missing keys。')
    if sa != 'per_slot_per_label_softmax_over_final_convnext_spatial_map':
        raise ValueError(f'fold{fold} spatial_attention={sa!r}，与本文件实现的 label-wise 空间注意力不符。')
    ta = ckpt.get('temporal_attention')
    if ta is None:
        raise ValueError(f'fold{fold} ckpt 没有 temporal_attention 字段，多半是(所有标签共用一套注意力) 的旧版产物，前向不同，不能用这份推理代码。')
    if ta != 'per_slot_per_label_softmax_over_6_tokens':
        raise ValueError(f'fold{fold} temporal_attention={ta!r}，与本文件实现的 label-wise 注意力不符。')
    vs = ckpt.get('val_sequence_strategy')
    if vs is None:
        if ckpt.get('val_group_strategy') is not None:
            raise ValueError(f"fold{fold} 只有旧版的 val_group_strategy={ckpt.get('val_group_strategy')!r}，不是本版训练产物。")
        print(f'WARNING: fold{fold} 无 val_sequence_strategy，跳过验证端策略校验')
    elif vs not in VAL_SEQ_STRATEGIES:
        raise ValueError(f'fold{fold} val_sequence_strategy={vs!r} 不在已知列表 {VAL_SEQ_STRATEGIES}；验证端前向可能与本文件不同，不要混用。')

def load_fold_models(devices, run_dir):
    models = []
    ref_name = None
    fusion_stack = []
    for k, (fold, path) in enumerate(resolve_checkpoints(run_dir)):
        dev = devices[k % len(devices)]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in ckpt:
            raise KeyError(f'{path} 中找不到 state_dict')
        arch = ckpt.get('architecture')
        if EXPECTED_ARCH is not None and arch != EXPECTED_ARCH:
            raise ValueError(f'fold{fold} architecture={arch!r}，但本文件实现的是 {EXPECTED_ARCH!r}。\nrun_dir 可能挂错了 run。\nrun_dir = {run_dir}')
        if ckpt.get('independent_encoder_per_slot') is True:
            raise ValueError(f'fold{fold} 的 ckpt 是 5 独立 encoder 版本，不能用这份共享 backbone 的推理代码。')
        if ckpt.get('shared_backbone_across_slots') is not True:
            raise ValueError(f'fold{fold} 的 ckpt 没有 shared_backbone_across_slots=True，不是共享 backbone 版本的训练产物。')

        def need(key, want, cast=int, tol=None):
            got = ckpt.get(key)
            if got is None:
                return
            got = cast(got)
            ok = abs(got - want) <= tol if tol is not None else got == want
            if not ok:
                raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
        need('img_size', IMG_SIZE)
        need('n_slots', N_SLOTS)
        need('slices_per_slot', SLICES_PER_SLOT)
        need('crop_mm', CROP_MM, cast=float, tol=1.0)
        ckpt_targets = ckpt.get('targets')
        if ckpt_targets is not None and list(ckpt_targets) != TARGETS:
            raise ValueError(f'fold{fold} targets 与当前 TARGETS 不一致。\ncheckpoint: {ckpt_targets}\ncurrent   : {TARGETS}')
        ckpt_slot_names = ckpt.get('slot_names')
        expected_slot_names = [s[0] for s in SLOTS]
        if ckpt_slot_names is not None and list(ckpt_slot_names) != expected_slot_names:
            raise ValueError(f'fold{fold} slot_names 与推理端不一致。\ncheckpoint: {list(ckpt_slot_names)}\ncurrent   : {expected_slot_names}')
        if ckpt_slot_names is None and k == 0:
            print('  (ckpt 无 slot_names 字段，跳过 slot 顺序校验 —— SLOTS 必须与预处理端逐字一致，时序专家是逐 slot 的)')
        _check_sequence_layout(ckpt, fold)
        slice_order = ckpt.get('slice_order')
        if slice_order is not None and slice_order != 'instance_number':
            print(f'WARNING: fold{fold} slice_order={slice_order!r}，与推理端预期不一致!')
        model_name = ckpt.get('model_name')
        if model_name is None:
            raise KeyError(f'{path} 里没有 model_name，多半是旧版 checkpoint，与当前训练代码不兼容。')
        if EXPECTED_MODEL_NAME is not None and model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f'fold{fold} backbone={model_name}，但 EXPECTED_MODEL_NAME={EXPECTED_MODEL_NAME}。\n!! 本版与上一版 ConvNeXt graycopy24 的 architecture 字符串相同，这道校验是分辨两者的唯一干净手段 —— 不要为了跑通而放宽它。\nrun_dir = {run_dir}')
        if ref_name is None:
            ref_name = model_name
        elif model_name != ref_name:
            raise ValueError(f'fold{fold} backbone ({model_name}) 与 fold{FOLDS[0]} ({ref_name}) 不同')
        model = KneeSharedConvNeXtNet(model_name, gru_hidden=int(ckpt.get('gru_hidden', 256)), gru_layers=int(ckpt.get('gru_layers', 1)), attn_hidden=int(ckpt.get('attn_hidden', 128)))
        ckpt_fd = ckpt.get('feature_dim')
        if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
            raise ValueError(f'fold{fold} feature_dim={ckpt_fd}，实例化得到 {model.feature_dim}')
        ckpt_sq = ckpt.get('seq_dim')
        if ckpt_sq is not None and int(ckpt_sq) != model.seq_dim:
            raise ValueError(f'fold{fold} seq_dim={ckpt_sq}，实例化得到 {model.seq_dim}')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        ref_fusion = ckpt.get('fusion_weights_no_missing')
        got_fusion = model.fusion_weights().detach()
        if ref_fusion is not None:
            d = float((torch.as_tensor(ref_fusion).float() - got_fusion).abs().max())
            if d > 0.0001:
                raise RuntimeError(f'fold{fold} 融合权重与 ckpt 存的 fusion_weights_no_missing 不一致 (max|Δ|={d:.2e}) —— gate 参数没有正确载入。')
        fusion_stack.append(got_fusion.cpu().numpy())
        models.append((fold, model.to(dev, memory_format=torch.channels_last).eval(), dev))
        print(f"  fold{fold} -> {dev} | {path.name} | epoch {ckpt.get('epoch', 'NA')} | derived {_fmt(ckpt.get('derived_auc'))} | gold {_fmt(ckpt.get('gold_auc'))} | select {_fmt(ckpt.get('selection_score'))} ({ckpt.get('selection', 'NA')}, n_gold_val={ckpt.get('n_gold_val', 'NA')}) | EMA={ckpt.get('ema', 'NA')}(decay={ckpt.get('ema_decay', 'NA')}) | aux_w={ckpt.get('expert_aux_weight', 'NA')}")
        if ckpt.get('aug'):
            print(f"         train aug: {ckpt['aug']}")
    ref_model = models[0][1]
    mean = [round(v, 4) for v in ref_model.mean.flatten().tolist()]
    std = [round(v, 4) for v in ref_model.std.flatten().tolist()]
    print(f'\nnormalize: mean {mean} | std {std}')
    if all((v == 0.0 for v in mean)) or all((v == 1.0 for v in std)):
        raise RuntimeError('mean/std 仍是初始占位值，checkpoint 里没有这两个 buffer！')
    n_param = sum((p.numel() for p in ref_model.parameters()))
    print(f'backbone: {ref_name} x1 (shared over {N_SLOTS} slots) | feature_dim {ref_model.feature_dim} | seq_dim {ref_model.seq_dim} | {n_param / 1000000.0:.1f}M params/fold | {len(models)} folds on {len(devices)} device(s)')
    if ref_model.feature_dim != EXPECTED_FEATURE_DIM:
        print(f'  !! feature_dim={ref_model.feature_dim}，TinyViT-21M 预期 {EXPECTED_FEATURE_DIM}')
    print('^^ 请把上面的 epoch / derived / gold 与训练日志逐行核对；对不上说明 run_dir 挂错了 run。')
    if PRINT_FUSION_WEIGHTS and fusion_stack:
        w = np.mean(np.stack(fusion_stack), axis=0)
        names = [s[0] for s in SLOTS]
        print('\nlabel-wise 融合权重 (无缺失 slot 时, %d fold 平均):' % len(fusion_stack))
        print('  ' + ' ' * 18 + ''.join((f'{n:>14s}' for n in names)))
        for j, t in enumerate(TARGETS):
            row = ''.join((f'{w[j, i]:14.4f}' for i in range(N_SLOTS)))
            top = names[int(np.argmax(w[j]))]
            print(f'  {t:18s}{row}   <- {top}')
        print('  (缺失 slot 在 softmax 前被屏蔽，实际权重按可用 slot 重新归一化)')
    return models

def resolve_backbone_head(backbone):
    mod = getattr(backbone, 'head', None)
    if mod is None or not isinstance(mod, nn.Module):
        return (None, None, 0)
    return ('head', mod, sum((p.numel() for p in mod.parameters())))

def clear_attention_bias_cache(module):
    n = 0
    for m in module.modules():
        if hasattr(m, 'attention_bias_cache'):
            m.attention_bias_cache = {}
            n += 1
    return n

@torch.inference_mode()
def prepare_fixed_fp16(models, loader):
    if len(loader.dataset) == 0:
        raise RuntimeError('No test studies to predict.')
    sample = next(iter(loader)) if FIXED_FP16_SANITY_CHECK else None
    cpu_indices = make_cpu_slot_indices(sample[1]) if sample is not None else None
    for fold, model, dev in models:
        if dev.type != 'cuda':
            raise RuntimeError('This fixed-FP16 inference script requires CUDA.')
        with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
            ref = None
            if sample is not None:
                print(f'[precision] fold{fold}: FP32 reference -> fixed FP16 check', flush=True)
                im = sample[0].to(dev, non_blocking=True)
                mk = sample[1].to(dev, non_blocking=True)
                ix = [v.to(dev, non_blocking=True) for v in cpu_indices]
                ref = [model(im, mk, valid_indices=ix, rotation_deg=a).float() for a in TTA_ANGLES]
                if not all((torch.isfinite(v).all().item() for v in ref)):
                    raise RuntimeError(f'fold{fold}: FP32 reference contains NaN/Inf.')
            head_name, head_mod, n_head_param = resolve_backbone_head(model.backbone)
            original_head = None
            if head_mod is not None:
                original_head = {k: v.detach().cpu().clone() for k, v in head_mod.state_dict().items()}
            model.backbone.half()
            if head_mod is not None:
                head_mod = getattr(model.backbone, head_name)
                head_mod.float()
                head_mod.load_state_dict(original_head, strict=True)
            model.fixed_fp16_backbone = True
            n_cleared = clear_attention_bias_cache(model.backbone)
            if n_cleared:
                print(f'  cleared {n_cleared} attention_bias_cache entries')
            n_bn, n_sub, n_zero = (0, 0, 0)
            for m in model.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                    n_bn += 1
                    v = m.running_var.float()
                    n_sub += int((v < 6.104e-05).sum())
                    n_zero += int((v == 0).sum())
            if n_bn:
                flag = '   <-- 看下面的对拍' if n_sub or n_zero else ''
                print(f'  BatchNorm: {n_bn} 层 | running_var 落入 fp16 次正规区 (<6.1e-5) 的通道 {n_sub} 个 | 归零 {n_zero} 个{flag}')
            head_prefix = f'backbone.{head_name}.' if head_name else None
            for name, p in model.named_parameters():
                half_feature = name.startswith('backbone.') and (not (head_prefix and name.startswith(head_prefix)))
                expected = torch.float16 if half_feature else torch.float32
                if p.is_floating_point() and p.dtype != expected:
                    raise RuntimeError(f'Unexpected dtype: {name} {p.dtype}, expected {expected}')
            if sample is not None:
                for angle, reference in zip(TTA_ANGLES, ref):
                    actual = model(im, mk, valid_indices=ix, rotation_deg=angle).float()
                    finite = torch.isfinite(actual).all().item()
                    dl = float((actual - reference).abs().max().item())
                    dp = float((actual.sigmoid() - reference.sigmoid()).abs().max().item())
                    print(f'  angle={angle:+g}: max|delta logit|={dl:.6f}, max|delta probability|={dp:.6f}', flush=True)
                    if not finite or dl > FP16_MAX_LOGIT_DIFF or dp > FP16_MAX_PROB_DIFF:
                        raise RuntimeError(f'fold{fold}, angle={angle}: FP16 check failed; logit={dl}, probability={dp}, finite={finite}')
                del im, mk, ix, ref, reference, actual
            head_txt = f'backbone.{head_name} ({n_head_param / 1000.0:.1f}K params) FP32' if head_mod is not None else 'backbone 无 head 子模块 (EffNet 的 forward_head 无参数)'
            print(f'[precision] fold{fold}: feature extractor FP16; {head_txt}; attention/GRU/fusion FP32')

def make_cpu_slot_indices(mask):
    if mask.device.type != 'cpu':
        raise ValueError('Build indices from the original CPU mask before H2D.')
    return [torch.nonzero(mask[:, s] > 0.5, as_tuple=False).flatten().pin_memory() for s in range(N_SLOTS)]

@torch.inference_mode()
def _predict_device_batch(dev, fold_models, imgs, mask, cpu_indices):
    with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
        im = imgs.to(dev, non_blocking=True)
        mk = mask.to(dev, non_blocking=True)
        indices = [v.to(dev, non_blocking=True) for v in cpu_indices]
        result = []
        for fold, model in fold_models:
            logits = None
            for angle in TTA_ANGLES:
                view_logits = model(im, mk, valid_indices=indices, rotation_deg=angle).float()
                logits = view_logits if logits is None else logits + view_logits
            logits = logits / len(TTA_ANGLES)
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f'fold{fold}: prediction contains NaN/Inf.')
            result.append((fold, values))
        return result

@torch.inference_mode()
def predict_test(models, loader):
    devices = sorted({d for _, _, d in models}, key=str)
    if any((d.type != 'cuda' for d in devices)):
        raise RuntimeError('Fixed-FP16 inference requires CUDA devices.')
    groups = {d: [(f, m) for f, m, md in models if md == d] for d in devices}
    fold_order = [f for f, _, _ in models]
    if len(set(fold_order)) != len(fold_order):
        raise RuntimeError('Duplicate fold IDs would misalign predictions.')
    per_fold = {f: [] for f in fold_order}
    start = time.perf_counter()
    n_seen = 0
    with ExitStack() as stack:
        pools = {d: stack.enter_context(ThreadPoolExecutor(max_workers=1)) for d in devices}
        for imgs, mask in tqdm(loader, total=len(loader), desc=f'Inference ({len(models)} folds / {len(devices)} GPUs)' if tuple(TTA_ANGLES) == (0.0,) else f'Rotation TTA{len(TTA_ANGLES)} ({len(models)} folds / {len(devices)} GPUs)', unit='batch', mininterval=0.5):
            cpu_indices = make_cpu_slot_indices(mask)
            futures = [pools[d].submit(_predict_device_batch, d, groups[d], imgs, mask, cpu_indices) for d in devices]
            results, error = ([], None)
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            got = dict(results)
            if len(results) != len(models) or set(got) != set(fold_order):
                raise RuntimeError('Missing/duplicate fold result in GPU dispatch.')
            for fold in fold_order:
                if got[fold].shape != (len(imgs), len(TARGETS)):
                    raise RuntimeError(f'fold{fold}: unexpected output shape {got[fold].shape}')
                per_fold[fold].append(got[fold])
            n_seen += len(imgs)
    seconds = time.perf_counter() - start
    logits = np.stack([np.concatenate(per_fold[f], axis=0) for f in fold_order])
    print(f'[timing] {n_seen} studies / {len(models)} folds: {seconds:.2f}s, {n_seen / max(seconds, 1e-09):.3f} studies/s (includes loader/H2D/D2H)')
    return (fold_order, logits)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _rank_normalize(p):
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for j in range(p.shape[1]):
        s = pd.Series(p[:, j]).rank(method='average').to_numpy()
        out[:, j] = (s - 1) / max(n - 1, 1)
    return out

def combine_folds(logits, mode):
    if mode == 'logit':
        return _sigmoid(logits.mean(axis=0))
    if mode == 'prob':
        return _sigmoid(logits).mean(axis=0)
    if mode == 'rank':
        probs = _sigmoid(logits)
        return np.stack([_rank_normalize(probs[k]) for k in range(probs.shape[0])]).mean(axis=0)
    raise ValueError(f'ENSEMBLE 必须是 logit | prob | rank，得到 {mode!r}')

def report_fold_agreement(logits, fold_order):
    n_folds, n, n_tgt = logits.shape
    if n_folds < 2 or n < 30:
        print(f'\n样本数 {n} 太少 (rank 后取值被量化成少数几档)，fold 一致性检查无意义，跳过。')
        return
    probs = _sigmoid(logits)
    ranks = [_rank_normalize(probs[k]) for k in range(n_folds)]
    print('\nfold 间 Spearman 相关 (per-target，列出最低的一对):')
    for j, t in enumerate(TARGETS):
        pairs = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                r = float(np.corrcoef(ranks[a][:, j], ranks[b][:, j])[0, 1])
                pairs.append((r, fold_order[a], fold_order[b]))
        rs = [p[0] for p in pairs]
        lo, fa, fb = min(pairs)
        flag = '  <-- 偏低' if lo < 0.7 else ''
        print(f'  {t:18s} mean {np.mean(rs):+.4f} | min {lo:+.4f} (fold{fa} vs fold{fb}){flag}')
''',
'dino': r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/zhuyifanss/distillstage1dinov2s384/rsna_dinov2s_392_oai_mix_unfreeze9')
RUN_DIR_NAME = 'rsna_dinov2s_392_oai_mix_unfreeze9'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (-12.0, 0.0, 12.0)
EXPECTED_MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'
EXPECTED_FEATURE_DIM = 384
PATCH_SIZE = 14
EXPECTED_ARCH = 'shared_dinov2s_2p5d_5expert_labelwise_spatiotemporal_attention'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CACHE_IMG_SIZE = 384
IMG_SIZE = 392
CROP_MM = 130.0
RESIZE_INTERP = cv2.INTER_LINEAR
RESIZE_DESC = f'cv2_INTER_LINEAR_{CACHE_IMG_SIZE}_to_{IMG_SIZE}'
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = 2
N_TOKENS = (SLICES_PER_SLOT - TOKEN_WINDOW) // TOKEN_STRIDE + 1
IN_CHANS = GROUP_SIZE
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

def resize_cache_study(imgs: np.ndarray) -> np.ndarray:
    if imgs.shape[-2:] == (IMG_SIZE, IMG_SIZE):
        return imgs
    if imgs.shape[-2:] != (CACHE_IMG_SIZE, CACHE_IMG_SIZE):
        raise ValueError(f'预期 cache 切片是 {CACHE_IMG_SIZE}x{CACHE_IMG_SIZE}，得到 {imgs.shape}')
    out = np.empty((*imgs.shape[:-2], IMG_SIZE, IMG_SIZE), dtype=imgs.dtype)
    for slot in range(imgs.shape[0]):
        for depth in range(imgs.shape[1]):
            out[slot, depth] = cv2.resize(imgs[slot, depth], (IMG_SIZE, IMG_SIZE), interpolation=RESIZE_INTERP)
    return out

class TestStudyDataset(Dataset):

    def __init__(self, cache, mask):
        self.cache = cache
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, i):
        imgs_np = resize_cache_study(np.array(self.cache[i]))
        imgs = torch.from_numpy(np.ascontiguousarray(imgs_np))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i])).float()
        return (imgs, m)

def rotate_grayscale_batch(x, angle_deg):
    if float(angle_deg) == 0.0:
        return x
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation TTA expects square [N,1,H,W] grayscale images.')
    with torch.autocast(device_type=x.device.type, enabled=False):
        radians = math.radians(float(angle_deg))
        c, s = (math.cos(radians), math.sin(radians))
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=x.device).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class KneeSharedDINOv2Net(nn.Module):

    def __init__(self, model_name: str, gru_hidden=256, gru_layers=1, attn_hidden=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=IN_CHANS, drop_path_rate=0.0, img_size=IMG_SIZE, global_pool='token')
        if len(getattr(self.backbone, 'blocks', [])) != 12:
            raise RuntimeError(f"预期 DINOv2-S/14 有 12 个 Transformer block，实际 {len(getattr(self.backbone, 'blocks', []))}")
        self.feature_dim = int(self.backbone.num_features)
        self.seq_dim = gru_hidden * 2
        self.fixed_fp16_backbone = False
        self.grus = nn.ModuleList([nn.GRU(input_size=self.feature_dim, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True) for _ in range(N_SLOTS)])
        self.spatial_attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.seq_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.expert_heads = nn.ModuleList([nn.Linear(self.seq_dim, len(TARGETS)) for _ in range(N_SLOTS)])
        self.fusion_gate_logits = nn.Parameter(torch.zeros(len(TARGETS), N_SLOTS, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))
        shape = (1, IN_CHANS, 1, 1)
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float16 if self.fixed_fp16_backbone else torch.float32)

            def run(images):
                tokens = self.backbone.forward_features(images)
                if not torch.is_tensor(tokens) or tokens.ndim != 3:
                    raise RuntimeError(f"预期 DINOv2 返回 [N,1+P,D]，实际 {getattr(tokens, 'shape', None)}")
                prefix = int(self.backbone.num_prefix_tokens)
                patches = tokens[:, prefix:, :]
                expected = images.shape[-2] // PATCH_SIZE * (images.shape[-1] // PATCH_SIZE)
                if patches.shape[1:] != (expected, self.feature_dim):
                    raise RuntimeError(f'patch token 形状异常: {tuple(patches.shape)}，预期 [*,{expected},{self.feature_dim}]')
                return patches
            c = int(BACKBONE_CHUNK_SIZE)
            if c <= 0 or x.shape[0] <= c:
                return run(x)
            return torch.cat([run(x[i:i + c]) for i in range(0, x.shape[0], c)], dim=0)

    def _encode_slot(self, slot: int, imgs: torch.Tensor, valid_idx: torch.Tensor, rotation_deg: float=0.0) -> torch.Tensor:
        G, C, R = (N_TOKENS, TOKEN_WINDOW, IMG_SIZE)
        x = imgs[:, slot].index_select(0, valid_idx)
        bv = x.shape[0]
        if rotation_deg != 0.0:
            x = rotate_grayscale_batch(x.reshape(bv * SLICES_PER_SLOT, 1, R, R), rotation_deg).reshape(bv, SLICES_PER_SLOT, R, R)
        x = x.unfold(1, C, TOKEN_STRIDE)
        if x.shape[1] != G:
            raise RuntimeError(f'unfold 得到 {x.shape[1]} 个 token，预期 {G} —— SLICES_PER_SLOT/TOKEN_WINDOW/TOKEN_STRIDE 配置不一致')
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(bv * G, C, R, R).float().div(255.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        feat_map = self._run_backbone_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float()
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)
            label_tokens = torch.einsum('npt,npd->ntd', spatial_a, pixels)
            label_tokens = label_tokens.reshape(bv, G, len(TARGETS), self.feature_dim).permute(0, 2, 1, 3).contiguous()
            h = label_tokens.reshape(bv * len(TARGETS), G, self.feature_dim)
            h, _ = self.grus[slot](h)
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)
            temporal_all = self.attentions[slot](h)
            temporal_a = temporal_all.diagonal(dim1=1, dim2=3).permute(0, 2, 1).softmax(dim=-1)
            label_feat = torch.einsum('btg,btgd->btd', temporal_a, h)
            head = self.expert_heads[slot]
            return torch.einsum('btd,td->bt', label_feat, head.weight) + head.bias

    def forward(self, imgs, mask, *, valid_indices=None, return_expert_logits: bool=False, rotation_deg: float=0.0):
        B = imgs.shape[0]
        expected = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(f'期望 imgs [B,{N_SLOTS},{SLICES_PER_SLOT},{IMG_SIZE},{IMG_SIZE}]，得到 {tuple(imgs.shape)}')
        available = mask > 0.5
        per_slot = []
        for s in range(N_SLOTS):
            valid_idx = valid_indices[s] if valid_indices is not None else torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=torch.float32)
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx, rotation_deg=rotation_deg)
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)
        expert_logits = torch.stack(per_slot, dim=1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -10000.0)
            gate = gate.softmax(dim=-1)
            logits = (gate * expert_logits.float().permute(0, 2, 1)).sum(dim=-1) + self.fusion_bias.float()
        if return_expert_logits:
            return (logits, expert_logits)
        return logits
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug', '6_fixed_nonoverlap_3slice_2p5d_groups_noaug')

def _check_sequence_layout(ckpt, fold):
    for key, want in (('n_groups', N_GROUPS), ('group_size', GROUP_SIZE), ('sequence_tokens_per_slot', N_TOKENS)):
        got = ckpt.get(key)
        if got is None:
            print(f'WARNING: fold{fold} ckpt 缺少 {key}，跳过该项校验')
            continue
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符' + (f'\n  (sequence_tokens_per_slot=24 说明这是上一版 ConvNeXt graycopy24 的 ckpt，不能用这份推理代码)' if key == 'sequence_tokens_per_slot' and int(got) == SLICES_PER_SLOT else ''))
    sps = ckpt.get('slices_per_slot')
    if sps is not None and int(sps) != N_GROUPS * GROUP_SIZE:
        raise ValueError(f'fold{fold} slices_per_slot={sps} != n_groups*group_size={N_GROUPS * GROUP_SIZE}')
    mode = ckpt.get('input_mode')
    if mode is None:
        raise ValueError(f'fold{fold} ckpt 没有 input_mode 字段，无法确认输入构造。')
    if mode != '3slice_2p5d_overlap_window3_stride2':
        raise ValueError(f'fold{fold} input_mode={mode!r}，与本文件实现的「3 层 2.5D 重叠三元组 (窗口 3 / 步长 2)」不符。\n  3slice_2p5d_neighboring_slices_as_channels 是上一次 DINOv2-392 run 的写法。')
    sa = ckpt.get('spatial_attention')
    if sa is None:
        raise ValueError(f'fold{fold} ckpt 没有 spatial_attention 字段，多半是只有时序注意力的旧版产物；本文件多一组 spatial_attentions.* 权重，strict=True 会报 missing keys。')
    if sa != 'per_slot_per_label_softmax_over_dinov2_patch_tokens':
        raise ValueError(f'fold{fold} spatial_attention={sa!r}，与本文件实现的「在归一化 DINOv2 patch token 上做 label-wise 空间注意力」不符。')
    ta = ckpt.get('temporal_attention')
    if ta is None:
        raise ValueError(f'fold{fold} ckpt 没有 temporal_attention 字段，多半是(所有标签共用一套注意力) 的旧版产物，前向不同，不能用这份推理代码。')
    if ta != 'per_slot_per_label_softmax_over_11_tokens':
        raise ValueError(f'fold{fold} temporal_attention={ta!r}，与本文件实现的 label-wise 注意力不符。')
    vs = ckpt.get('val_sequence_strategy')
    if vs is None:
        if ckpt.get('val_group_strategy') is not None:
            raise ValueError(f"fold{fold} 只有旧版的 val_group_strategy={ckpt.get('val_group_strategy')!r}，不是本版训练产物。")
        print(f'WARNING: fold{fold} 无 val_sequence_strategy，跳过验证端策略校验')
    elif vs not in VAL_SEQ_STRATEGIES:
        raise ValueError(f'fold{fold} val_sequence_strategy={vs!r} 不在已知列表 {VAL_SEQ_STRATEGIES}；验证端前向可能与本文件不同，不要混用。')

def load_fold_models(devices, run_dir):
    models = []
    ref_name = None
    fusion_stack = []
    for k, (fold, path) in enumerate(resolve_checkpoints(run_dir)):
        dev = devices[k % len(devices)]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in ckpt:
            raise KeyError(f'{path} 中找不到 state_dict')
        arch = ckpt.get('architecture')
        if EXPECTED_ARCH is not None and arch != EXPECTED_ARCH:
            raise ValueError(f'fold{fold} architecture={arch!r}，但本文件实现的是 {EXPECTED_ARCH!r}。\nrun_dir 可能挂错了 run。\nrun_dir = {run_dir}')
        if ckpt.get('independent_encoder_per_slot') is True:
            raise ValueError(f'fold{fold} 的 ckpt 是 5 独立 encoder 版本，不能用这份共享 backbone 的推理代码。')
        if ckpt.get('shared_backbone_across_slots') is not True:
            raise ValueError(f'fold{fold} 的 ckpt 没有 shared_backbone_across_slots=True，不是共享 backbone 版本的训练产物。')

        def need(key, want, cast=int, tol=None):
            got = ckpt.get(key)
            if got is None:
                return
            got = cast(got)
            ok = abs(got - want) <= tol if tol is not None else got == want
            if not ok:
                raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
        need('img_size', IMG_SIZE)
        need('cache_img_size', CACHE_IMG_SIZE)
        need('n_slots', N_SLOTS)
        need('slices_per_slot', SLICES_PER_SLOT)
        need('crop_mm', CROP_MM, cast=float, tol=1.0)
        resize = ckpt.get('input_resize')
        if resize is not None and resize != RESIZE_DESC:
            raise ValueError(f'fold{fold} input_resize={resize!r}，推理端是 {RESIZE_DESC!r}。')
        if resize is None and k == 0:
            print(f'  (ckpt 无 input_resize 字段 —— 本版训练端没写，属正常；插值方式按硬编码的 {RESIZE_DESC} 执行)')
        fb = ckpt.get('freeze_blocks')
        if fb is not None and int(fb) != 3:
            raise ValueError(f'fold{fold} freeze_blocks={fb}，与 unfreeze9 (冻结前 3 block) 不符。\n  冻结多少 block 不影响推理前向，但说明挂的不是这次 run。')
        bf = ckpt.get('backbone_finetune')
        if bf is not None and bf != 'last_9_blocks_and_final_norm':
            raise ValueError(f'fold{fold} backbone_finetune={bf!r}，与本次 run 不符。')
        ckpt_targets = ckpt.get('targets')
        if ckpt_targets is not None and list(ckpt_targets) != TARGETS:
            raise ValueError(f'fold{fold} targets 与当前 TARGETS 不一致。\ncheckpoint: {ckpt_targets}\ncurrent   : {TARGETS}')
        ckpt_slot_names = ckpt.get('slot_names')
        expected_slot_names = [s[0] for s in SLOTS]
        if ckpt_slot_names is not None and list(ckpt_slot_names) != expected_slot_names:
            raise ValueError(f'fold{fold} slot_names 与推理端不一致。\ncheckpoint: {list(ckpt_slot_names)}\ncurrent   : {expected_slot_names}')
        if ckpt_slot_names is None and k == 0:
            print('  (ckpt 无 slot_names 字段，跳过 slot 顺序校验 —— SLOTS 必须与预处理端逐字一致，时序专家是逐 slot 的)')
        _check_sequence_layout(ckpt, fold)
        slice_order = ckpt.get('slice_order')
        if slice_order is not None and slice_order != 'instance_number':
            print(f'WARNING: fold{fold} slice_order={slice_order!r}，与推理端预期不一致!')
        model_name = ckpt.get('model_name')
        if model_name is None:
            raise KeyError(f'{path} 里没有 model_name，多半是旧版 checkpoint，与当前训练代码不兼容。')
        if EXPECTED_MODEL_NAME is not None and model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f'fold{fold} backbone={model_name}，但 EXPECTED_MODEL_NAME={EXPECTED_MODEL_NAME}。\n!! 本版与上一版 ConvNeXt graycopy24 的 architecture 字符串相同，这道校验是分辨两者的唯一干净手段 —— 不要为了跑通而放宽它。\nrun_dir = {run_dir}')
        if ref_name is None:
            ref_name = model_name
        elif model_name != ref_name:
            raise ValueError(f'fold{fold} backbone ({model_name}) 与 fold{FOLDS[0]} ({ref_name}) 不同')
        model = KneeSharedDINOv2Net(model_name, gru_hidden=int(ckpt.get('gru_hidden', 256)), gru_layers=int(ckpt.get('gru_layers', 1)), attn_hidden=int(ckpt.get('attn_hidden', 128)))
        ckpt_fd = ckpt.get('feature_dim')
        if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
            raise ValueError(f'fold{fold} feature_dim={ckpt_fd}，实例化得到 {model.feature_dim}')
        ckpt_sq = ckpt.get('seq_dim')
        if ckpt_sq is not None and int(ckpt_sq) != model.seq_dim:
            raise ValueError(f'fold{fold} seq_dim={ckpt_sq}，实例化得到 {model.seq_dim}')
        ckpt_npx = ckpt.get('num_prefix_tokens_excluded')
        got_npx = int(model.backbone.num_prefix_tokens)
        if ckpt_npx is not None and int(ckpt_npx) != got_npx:
            raise ValueError(f'fold{fold} num_prefix_tokens_excluded={ckpt_npx}，本地实例化得到 {got_npx} —— timm 版本或 model_name 不一致。')
        if k == 0:
            print(f'  (DINOv2 prefix tokens 切掉 {got_npx} 个，保留 {N_PATCHES} 个 patch token)')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        ref_fusion = ckpt.get('fusion_weights_no_missing')
        got_fusion = model.fusion_weights().detach()
        if ref_fusion is not None:
            d = float((torch.as_tensor(ref_fusion).float() - got_fusion).abs().max())
            if d > 0.0001:
                raise RuntimeError(f'fold{fold} 融合权重与 ckpt 存的 fusion_weights_no_missing 不一致 (max|Δ|={d:.2e}) —— gate 参数没有正确载入。')
        fusion_stack.append(got_fusion.cpu().numpy())
        models.append((fold, model.to(dev, memory_format=torch.channels_last).eval(), dev))
        print(f"  fold{fold} -> {dev} | {path.name} | epoch {ckpt.get('epoch', 'NA')} | derived {_fmt(ckpt.get('derived_auc'))} | gold {_fmt(ckpt.get('gold_auc'))} | select {_fmt(ckpt.get('selection_score'))} ({ckpt.get('selection', 'NA')}, n_gold_val={ckpt.get('n_gold_val', 'NA')}) | EMA={ckpt.get('ema', 'NA')}(decay={ckpt.get('ema_decay', 'NA')}) | aux_w={ckpt.get('expert_aux_weight', 'NA')}")
        if ckpt.get('aug'):
            print(f"         train aug: {ckpt['aug']}")
    ref_model = models[0][1]
    mean = [round(v, 4) for v in ref_model.mean.flatten().tolist()]
    std = [round(v, 4) for v in ref_model.std.flatten().tolist()]
    print(f'\nnormalize: mean {mean} | std {std}')
    if all((v == 0.0 for v in mean)) or all((v == 1.0 for v in std)):
        raise RuntimeError('mean/std 仍是初始占位值，checkpoint 里没有这两个 buffer！')
    n_param = sum((p.numel() for p in ref_model.parameters()))
    print(f'backbone: {ref_name} x1 (shared over {N_SLOTS} slots) | feature_dim {ref_model.feature_dim} | seq_dim {ref_model.seq_dim} | {n_param / 1000000.0:.1f}M params/fold | {len(models)} folds on {len(devices)} device(s)')
    if ref_model.feature_dim != EXPECTED_FEATURE_DIM:
        print(f'  !! feature_dim={ref_model.feature_dim}，TinyViT-21M 预期 {EXPECTED_FEATURE_DIM}')
    print('^^ 请把上面的 epoch / derived / gold 与训练日志逐行核对；对不上说明 run_dir 挂错了 run。')
    if PRINT_FUSION_WEIGHTS and fusion_stack:
        w = np.mean(np.stack(fusion_stack), axis=0)
        names = [s[0] for s in SLOTS]
        print('\nlabel-wise 融合权重 (无缺失 slot 时, %d fold 平均):' % len(fusion_stack))
        print('  ' + ' ' * 18 + ''.join((f'{n:>14s}' for n in names)))
        for j, t in enumerate(TARGETS):
            row = ''.join((f'{w[j, i]:14.4f}' for i in range(N_SLOTS)))
            top = names[int(np.argmax(w[j]))]
            print(f'  {t:18s}{row}   <- {top}')
        print('  (缺失 slot 在 softmax 前被屏蔽，实际权重按可用 slot 重新归一化)')
    return models

def resolve_backbone_head(backbone):
    mod = getattr(backbone, 'head', None)
    if mod is None or not isinstance(mod, nn.Module):
        return (None, None, 0)
    return ('head', mod, sum((p.numel() for p in mod.parameters())))

def clear_attention_bias_cache(module):
    n = 0
    for m in module.modules():
        if hasattr(m, 'attention_bias_cache'):
            m.attention_bias_cache = {}
            n += 1
    return n

@torch.inference_mode()
def prepare_fixed_fp16(models, loader):
    if len(loader.dataset) == 0:
        raise RuntimeError('No test studies to predict.')
    sample = next(iter(loader)) if FIXED_FP16_SANITY_CHECK else None
    cpu_indices = make_cpu_slot_indices(sample[1]) if sample is not None else None
    for fold, model, dev in models:
        if dev.type != 'cuda':
            raise RuntimeError('This fixed-FP16 inference script requires CUDA.')
        with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
            ref = None
            if sample is not None:
                print(f'[precision] fold{fold}: FP32 reference -> fixed FP16 check', flush=True)
                im = sample[0].to(dev, non_blocking=True)
                mk = sample[1].to(dev, non_blocking=True)
                ix = [v.to(dev, non_blocking=True) for v in cpu_indices]
                ref = [model(im, mk, valid_indices=ix, rotation_deg=a).float() for a in TTA_ANGLES]
                if not all((torch.isfinite(v).all().item() for v in ref)):
                    raise RuntimeError(f'fold{fold}: FP32 reference contains NaN/Inf.')
            head_name, head_mod, n_head_param = resolve_backbone_head(model.backbone)
            original_head = None
            if head_mod is not None:
                original_head = {k: v.detach().cpu().clone() for k, v in head_mod.state_dict().items()}
            model.backbone.half()
            if head_mod is not None:
                head_mod = getattr(model.backbone, head_name)
                head_mod.float()
                head_mod.load_state_dict(original_head, strict=True)
            model.fixed_fp16_backbone = True
            n_cleared = clear_attention_bias_cache(model.backbone)
            if n_cleared:
                print(f'  cleared {n_cleared} attention_bias_cache entries')
            n_bn, n_sub, n_zero = (0, 0, 0)
            for m in model.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                    n_bn += 1
                    v = m.running_var.float()
                    n_sub += int((v < 6.104e-05).sum())
                    n_zero += int((v == 0).sum())
            if n_bn:
                flag = '   <-- 看下面的对拍' if n_sub or n_zero else ''
                print(f'  BatchNorm: {n_bn} 层 | running_var 落入 fp16 次正规区 (<6.1e-5) 的通道 {n_sub} 个 | 归零 {n_zero} 个{flag}')
            head_prefix = f'backbone.{head_name}.' if head_name else None
            for name, p in model.named_parameters():
                half_feature = name.startswith('backbone.') and (not (head_prefix and name.startswith(head_prefix)))
                expected = torch.float16 if half_feature else torch.float32
                if p.is_floating_point() and p.dtype != expected:
                    raise RuntimeError(f'Unexpected dtype: {name} {p.dtype}, expected {expected}')
            if sample is not None:
                for angle, reference in zip(TTA_ANGLES, ref):
                    actual = model(im, mk, valid_indices=ix, rotation_deg=angle).float()
                    finite = torch.isfinite(actual).all().item()
                    dl = float((actual - reference).abs().max().item())
                    dp = float((actual.sigmoid() - reference.sigmoid()).abs().max().item())
                    print(f'  angle={angle:+g}: max|delta logit|={dl:.6f}, max|delta probability|={dp:.6f}', flush=True)
                    if not finite or dl > FP16_MAX_LOGIT_DIFF or dp > FP16_MAX_PROB_DIFF:
                        raise RuntimeError(f'fold{fold}, angle={angle}: FP16 check failed; logit={dl}, probability={dp}, finite={finite}')
                del im, mk, ix, ref, reference, actual
            head_txt = f'backbone.{head_name} ({n_head_param / 1000.0:.1f}K params) FP32' if head_mod is not None else 'backbone 无 head 子模块 (EffNet 的 forward_head 无参数)'
            print(f'[precision] fold{fold}: feature extractor FP16; {head_txt}; attention/GRU/fusion FP32')

def make_cpu_slot_indices(mask):
    if mask.device.type != 'cpu':
        raise ValueError('Build indices from the original CPU mask before H2D.')
    return [torch.nonzero(mask[:, s] > 0.5, as_tuple=False).flatten().pin_memory() for s in range(N_SLOTS)]

@torch.inference_mode()
def _predict_device_batch(dev, fold_models, imgs, mask, cpu_indices):
    with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
        im = imgs.to(dev, non_blocking=True)
        mk = mask.to(dev, non_blocking=True)
        indices = [v.to(dev, non_blocking=True) for v in cpu_indices]
        result = []
        for fold, model in fold_models:
            logits = None
            for angle in TTA_ANGLES:
                view_logits = model(im, mk, valid_indices=indices, rotation_deg=angle).float()
                logits = view_logits if logits is None else logits + view_logits
            logits = logits / len(TTA_ANGLES)
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f'fold{fold}: prediction contains NaN/Inf.')
            result.append((fold, values))
        return result

@torch.inference_mode()
def predict_test(models, loader):
    devices = sorted({d for _, _, d in models}, key=str)
    if any((d.type != 'cuda' for d in devices)):
        raise RuntimeError('Fixed-FP16 inference requires CUDA devices.')
    groups = {d: [(f, m) for f, m, md in models if md == d] for d in devices}
    fold_order = [f for f, _, _ in models]
    if len(set(fold_order)) != len(fold_order):
        raise RuntimeError('Duplicate fold IDs would misalign predictions.')
    per_fold = {f: [] for f in fold_order}
    start = time.perf_counter()
    n_seen = 0
    with ExitStack() as stack:
        pools = {d: stack.enter_context(ThreadPoolExecutor(max_workers=1)) for d in devices}
        for imgs, mask in tqdm(loader, total=len(loader), desc=f'Inference ({len(models)} folds / {len(devices)} GPUs)' if tuple(TTA_ANGLES) == (0.0,) else f'Rotation TTA{len(TTA_ANGLES)} ({len(models)} folds / {len(devices)} GPUs)', unit='batch', mininterval=0.5):
            cpu_indices = make_cpu_slot_indices(mask)
            futures = [pools[d].submit(_predict_device_batch, d, groups[d], imgs, mask, cpu_indices) for d in devices]
            results, error = ([], None)
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            got = dict(results)
            if len(results) != len(models) or set(got) != set(fold_order):
                raise RuntimeError('Missing/duplicate fold result in GPU dispatch.')
            for fold in fold_order:
                if got[fold].shape != (len(imgs), len(TARGETS)):
                    raise RuntimeError(f'fold{fold}: unexpected output shape {got[fold].shape}')
                per_fold[fold].append(got[fold])
            n_seen += len(imgs)
    seconds = time.perf_counter() - start
    logits = np.stack([np.concatenate(per_fold[f], axis=0) for f in fold_order])
    print(f'[timing] {n_seen} studies / {len(models)} folds: {seconds:.2f}s, {n_seen / max(seconds, 1e-09):.3f} studies/s (includes loader/H2D/D2H)')
    return (fold_order, logits)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _rank_normalize(p):
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for j in range(p.shape[1]):
        s = pd.Series(p[:, j]).rank(method='average').to_numpy()
        out[:, j] = (s - 1) / max(n - 1, 1)
    return out

def combine_folds(logits, mode):
    if mode == 'logit':
        return _sigmoid(logits.mean(axis=0))
    if mode == 'prob':
        return _sigmoid(logits).mean(axis=0)
    if mode == 'rank':
        probs = _sigmoid(logits)
        return np.stack([_rank_normalize(probs[k]) for k in range(probs.shape[0])]).mean(axis=0)
    raise ValueError(f'ENSEMBLE 必须是 logit | prob | rank，得到 {mode!r}')

def report_fold_agreement(logits, fold_order):
    n_folds, n, n_tgt = logits.shape
    if n_folds < 2 or n < 30:
        print(f'\n样本数 {n} 太少 (rank 后取值被量化成少数几档)，fold 一致性检查无意义，跳过。')
        return
    probs = _sigmoid(logits)
    ranks = [_rank_normalize(probs[k]) for k in range(n_folds)]
    print('\nfold 间 Spearman 相关 (per-target，列出最低的一对):')
    for j, t in enumerate(TARGETS):
        pairs = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                r = float(np.corrcoef(ranks[a][:, j], ranks[b][:, j])[0, 1])
                pairs.append((r, fold_order[a], fold_order[b]))
        rs = [p[0] for p in pairs]
        lo, fa, fb = min(pairs)
        flag = '  <-- 偏低' if lo < 0.7 else ''
        print(f'  {t:18s} mean {np.mean(rs):+.4f} | min {lo:+.4f} (fold{fa} vs fold{fb}){flag}')
''',
'dino3': r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/lihu6666/dinov3-convnext-tiny-lihu-of-fork-of-rsna-cd4e8a/rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr')
RUN_DIR_NAME = 'rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (0.0,)
EXPECTED_MODEL_NAME = 'convnext_tiny.dinov3_lvd1689m'
EXPECTED_FEATURE_DIM = 768
EXPECTED_ARCH = 'shared_dinov3_convnext_tiny_2p5d11_oai58_42_labelwise_spatiotemporal_attention_v3'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
IMG_SIZE = 384
CROP_MM = 130.0
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = 2
N_TOKENS = (SLICES_PER_SLOT - TOKEN_WINDOW) // TOKEN_STRIDE + 1
IN_CHANS = GROUP_SIZE
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

class TestStudyDataset(Dataset):

    def __init__(self, cache, mask):
        self.cache = cache
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, i):
        imgs = torch.from_numpy(np.array(self.cache[i]))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i])).float()
        return (imgs, m)

def rotate_grayscale_batch(x, angle_deg):
    if float(angle_deg) == 0.0:
        return x
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation TTA expects square [N,1,H,W] grayscale images.')
    with torch.autocast(device_type=x.device.type, enabled=False):
        radians = math.radians(float(angle_deg))
        c, s = (math.cos(radians), math.sin(radians))
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=x.device).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class KneeSharedConvNeXtNet(nn.Module):

    def __init__(self, model_name: str, gru_hidden=256, gru_layers=1, attn_hidden=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=IN_CHANS, drop_path_rate=0.0)
        self.feature_dim = int(self.backbone.num_features)
        self.seq_dim = gru_hidden * 2
        self.fixed_fp16_backbone = False
        self.grus = nn.ModuleList([nn.GRU(input_size=self.feature_dim, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True) for _ in range(N_SLOTS)])
        self.spatial_attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.seq_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.expert_heads = nn.ModuleList([nn.Linear(self.seq_dim, len(TARGETS)) for _ in range(N_SLOTS)])
        self.fusion_gate_logits = nn.Parameter(torch.zeros(len(TARGETS), N_SLOTS, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))
        shape = (1, IN_CHANS, 1, 1)
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float16 if self.fixed_fp16_backbone else torch.float32)
            c = int(BACKBONE_CHUNK_SIZE)
            if c <= 0 or x.shape[0] <= c:
                feat = self.backbone.forward_features(x)
            else:
                parts = [self.backbone.forward_features(x[i:i + c]) for i in range(0, x.shape[0], c)]
                feat = torch.cat(parts, dim=0)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat) or feat.ndim != 4:
            raise RuntimeError(f"backbone forward_features 预期返回4维空间图，实际类型/shape: {type(feat)} / {getattr(feat, 'shape', None)}")
        if feat.shape[1] == self.feature_dim:
            pass
        elif feat.shape[-1] == self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f'backbone 空间图找不到 feature_dim={self.feature_dim} 的通道轴: {tuple(feat.shape)}')
        return feat

    def _apply_backbone_head(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3 or feat.shape[-1] != self.feature_dim:
            raise RuntimeError(f'backbone head 预期 [N,T,{self.feature_dim}]，得到 {tuple(feat.shape)}')
        n, t, d = feat.shape
        x = feat.reshape(n * t, d, 1, 1)
        try:
            x = self.backbone.forward_head(x, pre_logits=True)
        except TypeError:
            x = self.backbone.head(x, pre_logits=True)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[-1] != d:
            raise RuntimeError(f"backbone head 预期返回 [{n * t},{d}]，实际 {type(x)} / {getattr(x, 'shape', None)}")
        return x.reshape(n, t, d)

    def _encode_slot(self, slot: int, imgs: torch.Tensor, valid_idx: torch.Tensor, rotation_deg: float=0.0) -> torch.Tensor:
        G, C, R = (N_TOKENS, TOKEN_WINDOW, IMG_SIZE)
        x = imgs[:, slot].index_select(0, valid_idx)
        bv = x.shape[0]
        if rotation_deg != 0.0:
            x = rotate_grayscale_batch(x.reshape(bv * SLICES_PER_SLOT, 1, R, R), rotation_deg).reshape(bv, SLICES_PER_SLOT, R, R)
        x = x.unfold(1, C, TOKEN_STRIDE)
        if x.shape[1] != G:
            raise RuntimeError(f'unfold 得到 {x.shape[1]} 个 token，预期 {G} —— SLICES_PER_SLOT/TOKEN_WINDOW/TOKEN_STRIDE 配置不一致')
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(bv * G, C, R, R).float().div(255.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        feat_map = self._run_backbone_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float().flatten(2).transpose(1, 2)
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)
            label_tokens = torch.einsum('npt,npd->ntd', spatial_a, pixels)
            label_tokens = self._apply_backbone_head(label_tokens)
            label_tokens = label_tokens.reshape(bv, G, len(TARGETS), self.feature_dim).permute(0, 2, 1, 3).contiguous()
            h = label_tokens.reshape(bv * len(TARGETS), G, self.feature_dim)
            h, _ = self.grus[slot](h)
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)
            temporal_all = self.attentions[slot](h)
            temporal_a = temporal_all.diagonal(dim1=1, dim2=3).permute(0, 2, 1).softmax(dim=-1)
            label_feat = torch.einsum('btg,btgd->btd', temporal_a, h)
            head = self.expert_heads[slot]
            return torch.einsum('btd,td->bt', label_feat, head.weight) + head.bias

    def forward(self, imgs, mask, *, valid_indices=None, return_expert_logits: bool=False, rotation_deg: float=0.0):
        B = imgs.shape[0]
        expected = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(f'期望 imgs [B,{N_SLOTS},{SLICES_PER_SLOT},{IMG_SIZE},{IMG_SIZE}]，得到 {tuple(imgs.shape)}')
        available = mask > 0.5
        per_slot = []
        for s in range(N_SLOTS):
            valid_idx = valid_indices[s] if valid_indices is not None else torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=torch.float32)
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx, rotation_deg=rotation_deg)
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)
        expert_logits = torch.stack(per_slot, dim=1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -10000.0)
            gate = gate.softmax(dim=-1)
            logits = (gate * expert_logits.float().permute(0, 2, 1)).sum(dim=-1) + self.fusion_bias.float()
        if return_expert_logits:
            return (logits, expert_logits)
        return logits
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug',)

def _check_sequence_layout(ckpt, fold):
    for key, want in (('n_groups', N_GROUPS), ('group_size', GROUP_SIZE), ('sequence_tokens_per_slot', N_TOKENS)):
        got = ckpt.get(key)
        if got is None:
            print(f'WARNING: fold{fold} ckpt 缺少 {key}，跳过该项校验')
            continue
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符' + (f'\n  (sequence_tokens_per_slot=24 说明这是上一版 ConvNeXt graycopy24 的 ckpt，不能用这份推理代码)' if key == 'sequence_tokens_per_slot' and int(got) == SLICES_PER_SLOT else ''))
    sps = ckpt.get('slices_per_slot')
    if sps is not None and int(sps) != N_GROUPS * GROUP_SIZE:
        raise ValueError(f'fold{fold} slices_per_slot={sps} != n_groups*group_size={N_GROUPS * GROUP_SIZE}')
    mode = ckpt.get('input_mode')
    if mode is None:
        raise ValueError(f'fold{fold} ckpt 没有 input_mode 字段，无法确认输入构造。')
    if mode != '3slice_2p5d_overlap_window3_stride2':
        raise ValueError(f'fold{fold} input_mode={mode!r}，与本文件实现的「3 层 2.5D 重叠三元组 (窗口 3 / 步长 2)」不符 —— single_grayscale_slice_repeated_to_3_channels 是上一版 graycopy24。')
    sa = ckpt.get('spatial_attention')
    if sa is None:
        raise ValueError(f'fold{fold} ckpt 没有 spatial_attention 字段，多半是只有时序注意力的旧版产物；本文件多一组 spatial_attentions.* 权重，strict=True 会报 missing keys。')
    if sa != 'per_slot_per_label_softmax_over_final_convnext_spatial_map':
        raise ValueError(f'fold{fold} spatial_attention={sa!r}，与本文件实现的 label-wise 空间注意力不符。')
    ta = ckpt.get('temporal_attention')
    if ta is None:
        raise ValueError(f'fold{fold} ckpt 没有 temporal_attention 字段，多半是(所有标签共用一套注意力) 的旧版产物，前向不同，不能用这份推理代码。')
    if ta != 'per_slot_per_label_softmax_over_11_tokens':
        raise ValueError(f'fold{fold} temporal_attention={ta!r}，与本文件实现的 label-wise 注意力不符。')
    vs = ckpt.get('val_sequence_strategy')
    if vs is None:
        if ckpt.get('val_group_strategy') is not None:
            raise ValueError(f"fold{fold} 只有旧版的 val_group_strategy={ckpt.get('val_group_strategy')!r}，不是本版训练产物。")
        print(f'WARNING: fold{fold} 无 val_sequence_strategy，跳过验证端策略校验')
    elif vs not in VAL_SEQ_STRATEGIES:
        raise ValueError(f'fold{fold} val_sequence_strategy={vs!r} 不在已知列表 {VAL_SEQ_STRATEGIES}；验证端前向可能与本文件不同，不要混用。')

def load_fold_models(devices, run_dir):
    models = []
    ref_name = None
    fusion_stack = []
    for k, (fold, path) in enumerate(resolve_checkpoints(run_dir)):
        dev = devices[k % len(devices)]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in ckpt:
            raise KeyError(f'{path} 中找不到 state_dict')
        arch = ckpt.get('architecture')
        if EXPECTED_ARCH is not None and arch != EXPECTED_ARCH:
            raise ValueError(f'fold{fold} architecture={arch!r}，但本文件实现的是 {EXPECTED_ARCH!r}。\nrun_dir 可能挂错了 run。\nrun_dir = {run_dir}')
        if ckpt.get('independent_encoder_per_slot') is True:
            raise ValueError(f'fold{fold} 的 ckpt 是 5 独立 encoder 版本，不能用这份共享 backbone 的推理代码。')
        if ckpt.get('shared_backbone_across_slots') is not True:
            raise ValueError(f'fold{fold} 的 ckpt 没有 shared_backbone_across_slots=True，不是共享 backbone 版本的训练产物。')

        def need(key, want, cast=int, tol=None):
            got = ckpt.get(key)
            if got is None:
                return
            got = cast(got)
            ok = abs(got - want) <= tol if tol is not None else got == want
            if not ok:
                raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
        need('img_size', IMG_SIZE)
        need('n_slots', N_SLOTS)
        need('slices_per_slot', SLICES_PER_SLOT)
        need('crop_mm', CROP_MM, cast=float, tol=1.0)
        ckpt_targets = ckpt.get('targets')
        if ckpt_targets is not None and list(ckpt_targets) != TARGETS:
            raise ValueError(f'fold{fold} targets 与当前 TARGETS 不一致。\ncheckpoint: {ckpt_targets}\ncurrent   : {TARGETS}')
        ckpt_slot_names = ckpt.get('slot_names')
        expected_slot_names = [s[0] for s in SLOTS]
        if ckpt_slot_names is not None and list(ckpt_slot_names) != expected_slot_names:
            raise ValueError(f'fold{fold} slot_names 与推理端不一致。\ncheckpoint: {list(ckpt_slot_names)}\ncurrent   : {expected_slot_names}')
        if ckpt_slot_names is None and k == 0:
            print('  (ckpt 无 slot_names 字段，跳过 slot 顺序校验 —— SLOTS 必须与预处理端逐字一致，时序专家是逐 slot 的)')
        _check_sequence_layout(ckpt, fold)
        slice_order = ckpt.get('slice_order')
        if slice_order is not None and slice_order != 'physical_position_with_instance_fallback':
            print(f'WARNING: fold{fold} slice_order={slice_order!r}，与推理端预期不一致!')
        model_name = ckpt.get('model_name')
        if model_name is None:
            raise KeyError(f'{path} 里没有 model_name，多半是旧版 checkpoint，与当前训练代码不兼容。')
        if EXPECTED_MODEL_NAME is not None and model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f'fold{fold} backbone={model_name}，但 EXPECTED_MODEL_NAME={EXPECTED_MODEL_NAME}。\n!! 本版与上一版 ConvNeXt graycopy24 的 architecture 字符串相同，这道校验是分辨两者的唯一干净手段 —— 不要为了跑通而放宽它。\nrun_dir = {run_dir}')
        if ref_name is None:
            ref_name = model_name
        elif model_name != ref_name:
            raise ValueError(f'fold{fold} backbone ({model_name}) 与 fold{FOLDS[0]} ({ref_name}) 不同')
        model = KneeSharedConvNeXtNet(model_name, gru_hidden=int(ckpt.get('gru_hidden', 256)), gru_layers=int(ckpt.get('gru_layers', 1)), attn_hidden=int(ckpt.get('attn_hidden', 128)))
        ckpt_fd = ckpt.get('feature_dim')
        if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
            raise ValueError(f'fold{fold} feature_dim={ckpt_fd}，实例化得到 {model.feature_dim}')
        ckpt_sq = ckpt.get('seq_dim')
        if ckpt_sq is not None and int(ckpt_sq) != model.seq_dim:
            raise ValueError(f'fold{fold} seq_dim={ckpt_sq}，实例化得到 {model.seq_dim}')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        ref_fusion = ckpt.get('fusion_weights_no_missing')
        got_fusion = model.fusion_weights().detach()
        if ref_fusion is not None:
            d = float((torch.as_tensor(ref_fusion).float() - got_fusion).abs().max())
            if d > 0.0001:
                raise RuntimeError(f'fold{fold} 融合权重与 ckpt 存的 fusion_weights_no_missing 不一致 (max|Δ|={d:.2e}) —— gate 参数没有正确载入。')
        fusion_stack.append(got_fusion.cpu().numpy())
        models.append((fold, model.to(dev, memory_format=torch.channels_last).eval(), dev))
        print(f"  fold{fold} -> {dev} | {path.name} | epoch {ckpt.get('epoch', 'NA')} | derived {_fmt(ckpt.get('derived_auc'))} | gold {_fmt(ckpt.get('gold_auc'))} | select {_fmt(ckpt.get('selection_score'))} ({ckpt.get('selection', 'NA')}, n_gold_val={ckpt.get('n_gold_val', 'NA')}) | EMA={ckpt.get('ema', 'NA')}(decay={ckpt.get('ema_decay', 'NA')}) | aux_w={ckpt.get('expert_aux_weight', 'NA')}")
        if ckpt.get('aug'):
            print(f"         train aug: {ckpt['aug']}")
    ref_model = models[0][1]
    mean = [round(v, 4) for v in ref_model.mean.flatten().tolist()]
    std = [round(v, 4) for v in ref_model.std.flatten().tolist()]
    print(f'\nnormalize: mean {mean} | std {std}')
    if all((v == 0.0 for v in mean)) or all((v == 1.0 for v in std)):
        raise RuntimeError('mean/std 仍是初始占位值，checkpoint 里没有这两个 buffer！')
    n_param = sum((p.numel() for p in ref_model.parameters()))
    print(f'backbone: {ref_name} x1 (shared over {N_SLOTS} slots) | feature_dim {ref_model.feature_dim} | seq_dim {ref_model.seq_dim} | {n_param / 1000000.0:.1f}M params/fold | {len(models)} folds on {len(devices)} device(s)')
    if ref_model.feature_dim != EXPECTED_FEATURE_DIM:
        print(f'  !! feature_dim={ref_model.feature_dim}，DINOv3 ConvNeXt-Tiny 预期 {EXPECTED_FEATURE_DIM}')
    print('^^ 请把上面的 epoch / derived / gold 与训练日志逐行核对；对不上说明 run_dir 挂错了 run。')
    if PRINT_FUSION_WEIGHTS and fusion_stack:
        w = np.mean(np.stack(fusion_stack), axis=0)
        names = [s[0] for s in SLOTS]
        print('\nlabel-wise 融合权重 (无缺失 slot 时, %d fold 平均):' % len(fusion_stack))
        print('  ' + ' ' * 18 + ''.join((f'{n:>14s}' for n in names)))
        for j, t in enumerate(TARGETS):
            row = ''.join((f'{w[j, i]:14.4f}' for i in range(N_SLOTS)))
            top = names[int(np.argmax(w[j]))]
            print(f'  {t:18s}{row}   <- {top}')
        print('  (缺失 slot 在 softmax 前被屏蔽，实际权重按可用 slot 重新归一化)')
    return models

def resolve_backbone_head(backbone):
    mod = getattr(backbone, 'head', None)
    if mod is None or not isinstance(mod, nn.Module):
        return (None, None, 0)
    return ('head', mod, sum((p.numel() for p in mod.parameters())))

def clear_attention_bias_cache(module):
    n = 0
    for m in module.modules():
        if hasattr(m, 'attention_bias_cache'):
            m.attention_bias_cache = {}
            n += 1
    return n

@torch.inference_mode()
def prepare_fixed_fp16(models, loader):
    if len(loader.dataset) == 0:
        raise RuntimeError('No test studies to predict.')
    sample = next(iter(loader)) if FIXED_FP16_SANITY_CHECK else None
    cpu_indices = make_cpu_slot_indices(sample[1]) if sample is not None else None
    for fold, model, dev in models:
        if dev.type != 'cuda':
            raise RuntimeError('This fixed-FP16 inference script requires CUDA.')
        with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
            ref = None
            if sample is not None:
                print(f'[precision] fold{fold}: FP32 reference -> fixed FP16 check', flush=True)
                im = sample[0].to(dev, non_blocking=True)
                mk = sample[1].to(dev, non_blocking=True)
                ix = [v.to(dev, non_blocking=True) for v in cpu_indices]
                ref = [model(im, mk, valid_indices=ix, rotation_deg=a).float() for a in TTA_ANGLES]
                if not all((torch.isfinite(v).all().item() for v in ref)):
                    raise RuntimeError(f'fold{fold}: FP32 reference contains NaN/Inf.')
            head_name, head_mod, n_head_param = resolve_backbone_head(model.backbone)
            original_head = None
            if head_mod is not None:
                original_head = {k: v.detach().cpu().clone() for k, v in head_mod.state_dict().items()}
            model.backbone.half()
            if head_mod is not None:
                head_mod = getattr(model.backbone, head_name)
                head_mod.float()
                head_mod.load_state_dict(original_head, strict=True)
            model.fixed_fp16_backbone = True
            n_cleared = clear_attention_bias_cache(model.backbone)
            if n_cleared:
                print(f'  cleared {n_cleared} attention_bias_cache entries')
            n_bn, n_sub, n_zero = (0, 0, 0)
            for m in model.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                    n_bn += 1
                    v = m.running_var.float()
                    n_sub += int((v < 6.104e-05).sum())
                    n_zero += int((v == 0).sum())
            if n_bn:
                flag = '   <-- 看下面的对拍' if n_sub or n_zero else ''
                print(f'  BatchNorm: {n_bn} 层 | running_var 落入 fp16 次正规区 (<6.1e-5) 的通道 {n_sub} 个 | 归零 {n_zero} 个{flag}')
            head_prefix = f'backbone.{head_name}.' if head_name else None
            for name, p in model.named_parameters():
                half_feature = name.startswith('backbone.') and (not (head_prefix and name.startswith(head_prefix)))
                expected = torch.float16 if half_feature else torch.float32
                if p.is_floating_point() and p.dtype != expected:
                    raise RuntimeError(f'Unexpected dtype: {name} {p.dtype}, expected {expected}')
            if sample is not None:
                for angle, reference in zip(TTA_ANGLES, ref):
                    actual = model(im, mk, valid_indices=ix, rotation_deg=angle).float()
                    finite = torch.isfinite(actual).all().item()
                    dl = float((actual - reference).abs().max().item())
                    dp = float((actual.sigmoid() - reference.sigmoid()).abs().max().item())
                    print(f'  angle={angle:+g}: max|delta logit|={dl:.6f}, max|delta probability|={dp:.6f}', flush=True)
                    if not finite or dl > FP16_MAX_LOGIT_DIFF or dp > FP16_MAX_PROB_DIFF:
                        raise RuntimeError(f'fold{fold}, angle={angle}: FP16 check failed; logit={dl}, probability={dp}, finite={finite}')
                del im, mk, ix, ref, reference, actual
            head_txt = f'backbone.{head_name} ({n_head_param / 1000.0:.1f}K params) FP32' if head_mod is not None else 'backbone 无 head 子模块 (EffNet 的 forward_head 无参数)'
            print(f'[precision] fold{fold}: feature extractor FP16; {head_txt}; attention/GRU/fusion FP32')

def make_cpu_slot_indices(mask):
    if mask.device.type != 'cpu':
        raise ValueError('Build indices from the original CPU mask before H2D.')
    return [torch.nonzero(mask[:, s] > 0.5, as_tuple=False).flatten().pin_memory() for s in range(N_SLOTS)]

@torch.inference_mode()
def _predict_device_batch(dev, fold_models, imgs, mask, cpu_indices):
    with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
        im = imgs.to(dev, non_blocking=True)
        mk = mask.to(dev, non_blocking=True)
        indices = [v.to(dev, non_blocking=True) for v in cpu_indices]
        result = []
        for fold, model in fold_models:
            logits = None
            for angle in TTA_ANGLES:
                view_logits = model(im, mk, valid_indices=indices, rotation_deg=angle).float()
                logits = view_logits if logits is None else logits + view_logits
            logits = logits / len(TTA_ANGLES)
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f'fold{fold}: prediction contains NaN/Inf.')
            result.append((fold, values))
        return result

@torch.inference_mode()
def predict_test(models, loader):
    devices = sorted({d for _, _, d in models}, key=str)
    if any((d.type != 'cuda' for d in devices)):
        raise RuntimeError('Fixed-FP16 inference requires CUDA devices.')
    groups = {d: [(f, m) for f, m, md in models if md == d] for d in devices}
    fold_order = [f for f, _, _ in models]
    if len(set(fold_order)) != len(fold_order):
        raise RuntimeError('Duplicate fold IDs would misalign predictions.')
    per_fold = {f: [] for f in fold_order}
    start = time.perf_counter()
    n_seen = 0
    with ExitStack() as stack:
        pools = {d: stack.enter_context(ThreadPoolExecutor(max_workers=1)) for d in devices}
        for imgs, mask in tqdm(loader, total=len(loader), desc=f'Inference ({len(models)} folds / {len(devices)} GPUs)' if tuple(TTA_ANGLES) == (0.0,) else f'Rotation TTA{len(TTA_ANGLES)} ({len(models)} folds / {len(devices)} GPUs)', unit='batch', mininterval=0.5):
            cpu_indices = make_cpu_slot_indices(mask)
            futures = [pools[d].submit(_predict_device_batch, d, groups[d], imgs, mask, cpu_indices) for d in devices]
            results, error = ([], None)
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            got = dict(results)
            if len(results) != len(models) or set(got) != set(fold_order):
                raise RuntimeError('Missing/duplicate fold result in GPU dispatch.')
            for fold in fold_order:
                if got[fold].shape != (len(imgs), len(TARGETS)):
                    raise RuntimeError(f'fold{fold}: unexpected output shape {got[fold].shape}')
                per_fold[fold].append(got[fold])
            n_seen += len(imgs)
    seconds = time.perf_counter() - start
    logits = np.stack([np.concatenate(per_fold[f], axis=0) for f in fold_order])
    print(f'[timing] {n_seen} studies / {len(models)} folds: {seconds:.2f}s, {n_seen / max(seconds, 1e-09):.3f} studies/s (includes loader/H2D/D2H)')
    return (fold_order, logits)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _rank_normalize(p):
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for j in range(p.shape[1]):
        s = pd.Series(p[:, j]).rank(method='average').to_numpy()
        out[:, j] = (s - 1) / max(n - 1, 1)
    return out

def combine_folds(logits, mode):
    if mode == 'logit':
        return _sigmoid(logits.mean(axis=0))
    if mode == 'prob':
        return _sigmoid(logits).mean(axis=0)
    if mode == 'rank':
        probs = _sigmoid(logits)
        return np.stack([_rank_normalize(probs[k]) for k in range(probs.shape[0])]).mean(axis=0)
    raise ValueError(f'ENSEMBLE 必须是 logit | prob | rank，得到 {mode!r}')

def report_fold_agreement(logits, fold_order):
    n_folds, n, n_tgt = logits.shape
    if n_folds < 2 or n < 30:
        print(f'\n样本数 {n} 太少 (rank 后取值被量化成少数几档)，fold 一致性检查无意义，跳过。')
        return
    probs = _sigmoid(logits)
    ranks = [_rank_normalize(probs[k]) for k in range(n_folds)]
    print('\nfold 间 Spearman 相关 (per-target，列出最低的一对):')
    for j, t in enumerate(TARGETS):
        pairs = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                r = float(np.corrcoef(ranks[a][:, j], ranks[b][:, j])[0, 1])
                pairs.append((r, fold_order[a], fold_order[b]))
        rs = [p[0] for p in pairs]
        lo, fa, fb = min(pairs)
        flag = '  <-- 偏低' if lo < 0.7 else ''
        print(f'  {t:18s} mean {np.mean(rs):+.4f} | min {lo:+.4f} (fold{fa} vs fold{fb}){flag}')
''',
'rad': r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/zhuyifanss/distillstage1radimageresnet224/rsna_radimagenet50_cache384_input224_oai_graycopy24')
RUN_DIR_NAME = 'rsna_radimagenet50_cache384_input224_oai_graycopy24'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (-12.0, 0.0, 12.0)
EXPECTED_MODEL_NAME = 'radimagenet_resnet50'
EXPECTED_FEATURE_DIM = 2048
EXPECTED_ARCH = 'shared_convnext_tiny_graycopy_5expert_labelwise_spatiotemporal_attention_v3'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
IMG_SIZE = 384
CROP_MM = 130.0
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = None
N_TOKENS = SLICES_PER_SLOT
IN_CHANS = 3
MODEL_IMG_SIZE = 224
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

class TestStudyDataset(Dataset):

    def __init__(self, cache, mask):
        self.cache = cache
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, i):
        imgs = torch.from_numpy(np.array(self.cache[i]))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i])).float()
        return (imgs, m)

def rotate_grayscale_batch(x, angle_deg):
    if float(angle_deg) == 0.0:
        return x
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation TTA expects square [N,1,H,W] grayscale images.')
    with torch.autocast(device_type=x.device.type, enabled=False):
        radians = math.radians(float(angle_deg))
        c, s = (math.cos(radians), math.sin(radians))
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=x.device).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class RadImageNetResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        self.encoder = resnet50(weights=None)
        self.encoder.fc = nn.Identity()
        self.num_features = 2048

    def forward_features(self, x):
        m = self.encoder
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        return m.layer4(x)

class KneeSharedConvNeXtNet(nn.Module):

    def __init__(self, model_name: str, gru_hidden=256, gru_layers=1, attn_hidden=128):
        super().__init__()
        if model_name != EXPECTED_MODEL_NAME:
            raise RuntimeError(f'本文件只实现了 {EXPECTED_MODEL_NAME}，得到 {model_name!r}')
        self.backbone = RadImageNetResNet50()
        self.feature_dim = int(self.backbone.num_features)
        self.seq_dim = gru_hidden * 2
        self.fixed_fp16_backbone = False
        self.grus = nn.ModuleList([nn.GRU(input_size=self.feature_dim, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True) for _ in range(N_SLOTS)])
        self.spatial_attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.seq_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.expert_heads = nn.ModuleList([nn.Linear(self.seq_dim, len(TARGETS)) for _ in range(N_SLOTS)])
        self.fusion_gate_logits = nn.Parameter(torch.zeros(len(TARGETS), N_SLOTS, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))
        shape = (1, IN_CHANS, 1, 1)
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float16 if self.fixed_fp16_backbone else torch.float32)
            c = int(BACKBONE_CHUNK_SIZE)
            if c <= 0 or x.shape[0] <= c:
                feat = self.backbone.forward_features(x)
            else:
                parts = [self.backbone.forward_features(x[i:i + c]) for i in range(0, x.shape[0], c)]
                feat = torch.cat(parts, dim=0)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat) or feat.ndim != 4:
            raise RuntimeError(f"backbone forward_features 预期返回4维空间图，实际类型/shape: {type(feat)} / {getattr(feat, 'shape', None)}")
        if feat.shape[1] == self.feature_dim:
            pass
        elif feat.shape[-1] == self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f'backbone 空间图找不到 feature_dim={self.feature_dim} 的通道轴: {tuple(feat.shape)}')
        return feat

    def _apply_backbone_head(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3 or feat.shape[-1] != self.feature_dim:
            raise RuntimeError(f'backbone head 预期 [N,T,{self.feature_dim}]，得到 {tuple(feat.shape)}')
        return feat

    def _encode_slot(self, slot: int, imgs: torch.Tensor, valid_idx: torch.Tensor, rotation_deg: float=0.0) -> torch.Tensor:
        G, R = (N_TOKENS, IMG_SIZE)
        x = imgs[:, slot].index_select(0, valid_idx)
        bv = x.shape[0]
        x = x.reshape(bv * G, 1, R, R)
        if rotation_deg != 0.0:
            x = rotate_grayscale_batch(x, rotation_deg)
        x = x.repeat(1, IN_CHANS, 1, 1).float().div(255.0)
        x = F.interpolate(x, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE), mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        feat_map = self._run_backbone_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float().flatten(2).transpose(1, 2)
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)
            label_tokens = torch.einsum('npt,npd->ntd', spatial_a, pixels)
            label_tokens = self._apply_backbone_head(label_tokens)
            label_tokens = label_tokens.reshape(bv, G, len(TARGETS), self.feature_dim).permute(0, 2, 1, 3).contiguous()
            h = label_tokens.reshape(bv * len(TARGETS), G, self.feature_dim)
            h, _ = self.grus[slot](h)
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)
            temporal_all = self.attentions[slot](h)
            temporal_a = temporal_all.diagonal(dim1=1, dim2=3).permute(0, 2, 1).softmax(dim=-1)
            label_feat = torch.einsum('btg,btgd->btd', temporal_a, h)
            head = self.expert_heads[slot]
            return torch.einsum('btd,td->bt', label_feat, head.weight) + head.bias

    def forward(self, imgs, mask, *, valid_indices=None, return_expert_logits: bool=False, rotation_deg: float=0.0):
        B = imgs.shape[0]
        expected = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(f'期望 imgs [B,{N_SLOTS},{SLICES_PER_SLOT},{IMG_SIZE},{IMG_SIZE}]，得到 {tuple(imgs.shape)}')
        available = mask > 0.5
        per_slot = []
        for s in range(N_SLOTS):
            valid_idx = valid_indices[s] if valid_indices is not None else torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=torch.float32)
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx, rotation_deg=rotation_deg)
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)
        expert_logits = torch.stack(per_slot, dim=1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -10000.0)
            gate = gate.softmax(dim=-1)
            logits = (gate * expert_logits.float().permute(0, 2, 1)).sum(dim=-1) + self.fusion_bias.float()
        if return_expert_logits:
            return (logits, expert_logits)
        return logits
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug',)

def _check_sequence_layout(ckpt, fold):
    for key, want in (('n_groups', N_GROUPS), ('group_size', GROUP_SIZE), ('sequence_tokens_per_slot', N_TOKENS)):
        got = ckpt.get(key)
        if got is None:
            print(f'WARNING: fold{fold} ckpt 缺少 {key}，跳过该项校验')
            continue
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符' + (f'\n  (sequence_tokens_per_slot=11 说明这是 2.5D 重叠三元组那几版的 ckpt，不能用这份灰度复制的推理代码)' if key == 'sequence_tokens_per_slot' and int(got) == 11 else ''))
    sps = ckpt.get('slices_per_slot')
    if sps is not None and int(sps) != N_GROUPS * GROUP_SIZE:
        raise ValueError(f'fold{fold} slices_per_slot={sps} != n_groups*group_size={N_GROUPS * GROUP_SIZE}')
    mode = ckpt.get('input_mode')
    if mode is None:
        raise ValueError(f'fold{fold} ckpt 没有 input_mode 字段，无法确认输入构造。')
    if mode != 'single_grayscale_slice_repeated_to_3_channels':
        raise ValueError(f'fold{fold} input_mode={mode!r}，与本文件实现的「单切片灰度复制成 3 通道」不符 —— 3slice_2p5d_overlap_window3_stride2 是 2.5D 三元组那几版。')
    for key, want in (('cache_img_size', IMG_SIZE), ('model_img_size', MODEL_IMG_SIZE)):
        got = ckpt.get(key)
        if got is None:
            raise ValueError(f'fold{fold} ckpt 缺少 {key} —— 不是 cache384/input224 这一版的产物。')
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
    rs = ckpt.get('input_resize')
    if rs != 'bilinear_align_corners_false_antialias_true':
        raise ValueError(f'fold{fold} input_resize={rs!r}，与本文件的 384->224 抗锯齿双线性不符。')
    nm = ckpt.get('input_normalization')
    if nm != 'official_pytorch_demo_minus1_plus1':
        raise ValueError(f'fold{fold} input_normalization={nm!r}，与本文件预期不符 (mean=std=0.5，从 ckpt buffer 载入)。')
    sa = ckpt.get('spatial_attention')
    if sa is None:
        raise ValueError(f'fold{fold} ckpt 没有 spatial_attention 字段，多半是只有时序注意力的旧版产物；本文件多一组 spatial_attentions.* 权重，strict=True 会报 missing keys。')
    if sa != 'per_slot_per_label_softmax_over_final_convnext_spatial_map':
        raise ValueError(f'fold{fold} spatial_attention={sa!r}，与本文件实现的 label-wise 空间注意力不符。')
    ta = ckpt.get('temporal_attention')
    if ta is None:
        raise ValueError(f'fold{fold} ckpt 没有 temporal_attention 字段，多半是(所有标签共用一套注意力) 的旧版产物，前向不同，不能用这份推理代码。')
    if ta != 'per_slot_per_label_softmax_over_6_tokens':
        raise ValueError(f'fold{fold} temporal_attention={ta!r}，与本文件实现的 label-wise 注意力不符。')
    vs = ckpt.get('val_sequence_strategy')
    if vs is None:
        if ckpt.get('val_group_strategy') is not None:
            raise ValueError(f"fold{fold} 只有旧版的 val_group_strategy={ckpt.get('val_group_strategy')!r}，不是本版训练产物。")
        print(f'WARNING: fold{fold} 无 val_sequence_strategy，跳过验证端策略校验')
    elif vs not in VAL_SEQ_STRATEGIES:
        raise ValueError(f'fold{fold} val_sequence_strategy={vs!r} 不在已知列表 {VAL_SEQ_STRATEGIES}；验证端前向可能与本文件不同，不要混用。')

def load_fold_models(devices, run_dir):
    models = []
    ref_name = None
    fusion_stack = []
    for k, (fold, path) in enumerate(resolve_checkpoints(run_dir)):
        dev = devices[k % len(devices)]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in ckpt:
            raise KeyError(f'{path} 中找不到 state_dict')
        arch = ckpt.get('architecture')
        if EXPECTED_ARCH is not None and arch != EXPECTED_ARCH:
            raise ValueError(f'fold{fold} architecture={arch!r}，但本文件实现的是 {EXPECTED_ARCH!r}。\nrun_dir 可能挂错了 run。\nrun_dir = {run_dir}')
        if ckpt.get('independent_encoder_per_slot') is True:
            raise ValueError(f'fold{fold} 的 ckpt 是 5 独立 encoder 版本，不能用这份共享 backbone 的推理代码。')
        if ckpt.get('shared_backbone_across_slots') is not True:
            raise ValueError(f'fold{fold} 的 ckpt 没有 shared_backbone_across_slots=True，不是共享 backbone 版本的训练产物。')

        def need(key, want, cast=int, tol=None):
            got = ckpt.get(key)
            if got is None:
                return
            got = cast(got)
            ok = abs(got - want) <= tol if tol is not None else got == want
            if not ok:
                raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
        need('img_size', IMG_SIZE)
        need('n_slots', N_SLOTS)
        need('slices_per_slot', SLICES_PER_SLOT)
        need('crop_mm', CROP_MM, cast=float, tol=1.0)
        ckpt_targets = ckpt.get('targets')
        if ckpt_targets is not None and list(ckpt_targets) != TARGETS:
            raise ValueError(f'fold{fold} targets 与当前 TARGETS 不一致。\ncheckpoint: {ckpt_targets}\ncurrent   : {TARGETS}')
        ckpt_slot_names = ckpt.get('slot_names')
        expected_slot_names = [s[0] for s in SLOTS]
        if ckpt_slot_names is not None and list(ckpt_slot_names) != expected_slot_names:
            raise ValueError(f'fold{fold} slot_names 与推理端不一致。\ncheckpoint: {list(ckpt_slot_names)}\ncurrent   : {expected_slot_names}')
        if ckpt_slot_names is None and k == 0:
            print('  (ckpt 无 slot_names 字段，跳过 slot 顺序校验 —— SLOTS 必须与预处理端逐字一致，时序专家是逐 slot 的)')
        _check_sequence_layout(ckpt, fold)
        slice_order = ckpt.get('slice_order')
        if slice_order is not None and slice_order != 'instance_number':
            print(f'WARNING: fold{fold} slice_order={slice_order!r}，与推理端预期不一致!')
        model_name = ckpt.get('model_name')
        if model_name is None:
            raise KeyError(f'{path} 里没有 model_name，多半是旧版 checkpoint，与当前训练代码不兼容。')
        if EXPECTED_MODEL_NAME is not None and model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f'fold{fold} backbone={model_name}，但 EXPECTED_MODEL_NAME={EXPECTED_MODEL_NAME}。\n!! 本版与上一版 ConvNeXt graycopy24 的 architecture 字符串相同，这道校验是分辨两者的唯一干净手段 —— 不要为了跑通而放宽它。\nrun_dir = {run_dir}')
        if ref_name is None:
            ref_name = model_name
        elif model_name != ref_name:
            raise ValueError(f'fold{fold} backbone ({model_name}) 与 fold{FOLDS[0]} ({ref_name}) 不同')
        model = KneeSharedConvNeXtNet(model_name, gru_hidden=int(ckpt.get('gru_hidden', 256)), gru_layers=int(ckpt.get('gru_layers', 1)), attn_hidden=int(ckpt.get('attn_hidden', 128)))
        ckpt_fd = ckpt.get('feature_dim')
        if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
            raise ValueError(f'fold{fold} feature_dim={ckpt_fd}，实例化得到 {model.feature_dim}')
        ckpt_sq = ckpt.get('seq_dim')
        if ckpt_sq is not None and int(ckpt_sq) != model.seq_dim:
            raise ValueError(f'fold{fold} seq_dim={ckpt_sq}，实例化得到 {model.seq_dim}')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        ref_fusion = ckpt.get('fusion_weights_no_missing')
        got_fusion = model.fusion_weights().detach()
        if ref_fusion is not None:
            d = float((torch.as_tensor(ref_fusion).float() - got_fusion).abs().max())
            if d > 0.0001:
                raise RuntimeError(f'fold{fold} 融合权重与 ckpt 存的 fusion_weights_no_missing 不一致 (max|Δ|={d:.2e}) —— gate 参数没有正确载入。')
        fusion_stack.append(got_fusion.cpu().numpy())
        models.append((fold, model.to(dev, memory_format=torch.channels_last).eval(), dev))
        print(f"  fold{fold} -> {dev} | {path.name} | epoch {ckpt.get('epoch', 'NA')} | derived {_fmt(ckpt.get('derived_auc'))} | gold {_fmt(ckpt.get('gold_auc'))} | select {_fmt(ckpt.get('selection_score'))} ({ckpt.get('selection', 'NA')}, n_gold_val={ckpt.get('n_gold_val', 'NA')}) | EMA={ckpt.get('ema', 'NA')}(decay={ckpt.get('ema_decay', 'NA')}) | aux_w={ckpt.get('expert_aux_weight', 'NA')}")
        if ckpt.get('aug'):
            print(f"         train aug: {ckpt['aug']}")
        if ckpt.get('label_mix') and k == 0:
            print(f"         label mix: {ckpt['label_mix']}")
    ref_model = models[0][1]
    mean = [round(v, 4) for v in ref_model.mean.flatten().tolist()]
    std = [round(v, 4) for v in ref_model.std.flatten().tolist()]
    print(f'\nnormalize: mean {mean} | std {std}')
    if all((v == 0.0 for v in mean)) or all((v == 1.0 for v in std)):
        raise RuntimeError('mean/std 仍是初始占位值，checkpoint 里没有这两个 buffer！')
    n_param = sum((p.numel() for p in ref_model.parameters()))
    print(f'backbone: {ref_name} x1 (shared over {N_SLOTS} slots) | feature_dim {ref_model.feature_dim} | seq_dim {ref_model.seq_dim} | {n_param / 1000000.0:.1f}M params/fold | {len(models)} folds on {len(devices)} device(s)')
    if ref_model.feature_dim != EXPECTED_FEATURE_DIM:
        print(f'  !! feature_dim={ref_model.feature_dim}，RadImageNet ResNet50 预期 {EXPECTED_FEATURE_DIM}')
    print('^^ 请把上面的 epoch / derived / gold 与训练日志逐行核对；对不上说明 run_dir 挂错了 run。')
    if PRINT_FUSION_WEIGHTS and fusion_stack:
        w = np.mean(np.stack(fusion_stack), axis=0)
        names = [s[0] for s in SLOTS]
        print('\nlabel-wise 融合权重 (无缺失 slot 时, %d fold 平均):' % len(fusion_stack))
        print('  ' + ' ' * 18 + ''.join((f'{n:>14s}' for n in names)))
        for j, t in enumerate(TARGETS):
            row = ''.join((f'{w[j, i]:14.4f}' for i in range(N_SLOTS)))
            top = names[int(np.argmax(w[j]))]
            print(f'  {t:18s}{row}   <- {top}')
        print('  (缺失 slot 在 softmax 前被屏蔽，实际权重按可用 slot 重新归一化)')
    return models

def resolve_backbone_head(backbone):
    mod = getattr(backbone, 'head', None)
    if mod is None or not isinstance(mod, nn.Module):
        return (None, None, 0)
    return ('head', mod, sum((p.numel() for p in mod.parameters())))

def clear_attention_bias_cache(module):
    n = 0
    for m in module.modules():
        if hasattr(m, 'attention_bias_cache'):
            m.attention_bias_cache = {}
            n += 1
    return n

@torch.inference_mode()
def prepare_fixed_fp16(models, loader):
    if len(loader.dataset) == 0:
        raise RuntimeError('No test studies to predict.')
    sample = next(iter(loader)) if FIXED_FP16_SANITY_CHECK else None
    cpu_indices = make_cpu_slot_indices(sample[1]) if sample is not None else None
    for fold, model, dev in models:
        if dev.type != 'cuda':
            raise RuntimeError('This fixed-FP16 inference script requires CUDA.')
        with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
            ref = None
            if sample is not None:
                print(f'[precision] fold{fold}: FP32 reference -> fixed FP16 check', flush=True)
                im = sample[0].to(dev, non_blocking=True)
                mk = sample[1].to(dev, non_blocking=True)
                ix = [v.to(dev, non_blocking=True) for v in cpu_indices]
                ref = [model(im, mk, valid_indices=ix, rotation_deg=a).float() for a in TTA_ANGLES]
                if not all((torch.isfinite(v).all().item() for v in ref)):
                    raise RuntimeError(f'fold{fold}: FP32 reference contains NaN/Inf.')
            head_name, head_mod, n_head_param = resolve_backbone_head(model.backbone)
            original_head = None
            if head_mod is not None:
                original_head = {k: v.detach().cpu().clone() for k, v in head_mod.state_dict().items()}
            model.backbone.half()
            if head_mod is not None:
                head_mod = getattr(model.backbone, head_name)
                head_mod.float()
                head_mod.load_state_dict(original_head, strict=True)
            model.fixed_fp16_backbone = True
            n_cleared = clear_attention_bias_cache(model.backbone)
            if n_cleared:
                print(f'  cleared {n_cleared} attention_bias_cache entries')
            n_bn, n_sub, n_zero = (0, 0, 0)
            for m in model.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                    n_bn += 1
                    v = m.running_var.float()
                    n_sub += int((v < 6.104e-05).sum())
                    n_zero += int((v == 0).sum())
            if n_bn:
                flag = '   <-- 看下面的对拍' if n_sub or n_zero else ''
                print(f'  BatchNorm: {n_bn} 层 | running_var 落入 fp16 次正规区 (<6.1e-5) 的通道 {n_sub} 个 | 归零 {n_zero} 个{flag}')
            head_prefix = f'backbone.{head_name}.' if head_name else None
            for name, p in model.named_parameters():
                half_feature = name.startswith('backbone.') and (not (head_prefix and name.startswith(head_prefix)))
                expected = torch.float16 if half_feature else torch.float32
                if p.is_floating_point() and p.dtype != expected:
                    raise RuntimeError(f'Unexpected dtype: {name} {p.dtype}, expected {expected}')
            if sample is not None:
                for angle, reference in zip(TTA_ANGLES, ref):
                    actual = model(im, mk, valid_indices=ix, rotation_deg=angle).float()
                    finite = torch.isfinite(actual).all().item()
                    dl = float((actual - reference).abs().max().item())
                    dp = float((actual.sigmoid() - reference.sigmoid()).abs().max().item())
                    print(f'  angle={angle:+g}: max|delta logit|={dl:.6f}, max|delta probability|={dp:.6f}', flush=True)
                    if not finite or dl > FP16_MAX_LOGIT_DIFF or dp > FP16_MAX_PROB_DIFF:
                        raise RuntimeError(f'fold{fold}, angle={angle}: FP16 check failed; logit={dl}, probability={dp}, finite={finite}')
                del im, mk, ix, ref, reference, actual
            head_txt = f'backbone.{head_name} ({n_head_param / 1000.0:.1f}K params) FP32' if head_mod is not None else 'backbone 无 head 子模块 (EffNet 的 forward_head 无参数)'
            print(f'[precision] fold{fold}: feature extractor FP16; {head_txt}; attention/GRU/fusion FP32')

def make_cpu_slot_indices(mask):
    if mask.device.type != 'cpu':
        raise ValueError('Build indices from the original CPU mask before H2D.')
    return [torch.nonzero(mask[:, s] > 0.5, as_tuple=False).flatten().pin_memory() for s in range(N_SLOTS)]

@torch.inference_mode()
def _predict_device_batch(dev, fold_models, imgs, mask, cpu_indices):
    with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
        im = imgs.to(dev, non_blocking=True)
        mk = mask.to(dev, non_blocking=True)
        indices = [v.to(dev, non_blocking=True) for v in cpu_indices]
        result = []
        for fold, model in fold_models:
            logits = None
            for angle in TTA_ANGLES:
                view_logits = model(im, mk, valid_indices=indices, rotation_deg=angle).float()
                logits = view_logits if logits is None else logits + view_logits
            logits = logits / len(TTA_ANGLES)
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f'fold{fold}: prediction contains NaN/Inf.')
            result.append((fold, values))
        return result

@torch.inference_mode()
def predict_test(models, loader):
    devices = sorted({d for _, _, d in models}, key=str)
    if any((d.type != 'cuda' for d in devices)):
        raise RuntimeError('Fixed-FP16 inference requires CUDA devices.')
    groups = {d: [(f, m) for f, m, md in models if md == d] for d in devices}
    fold_order = [f for f, _, _ in models]
    if len(set(fold_order)) != len(fold_order):
        raise RuntimeError('Duplicate fold IDs would misalign predictions.')
    per_fold = {f: [] for f in fold_order}
    start = time.perf_counter()
    n_seen = 0
    with ExitStack() as stack:
        pools = {d: stack.enter_context(ThreadPoolExecutor(max_workers=1)) for d in devices}
        for imgs, mask in tqdm(loader, total=len(loader), desc=f'Inference ({len(models)} folds / {len(devices)} GPUs)' if tuple(TTA_ANGLES) == (0.0,) else f'Rotation TTA{len(TTA_ANGLES)} ({len(models)} folds / {len(devices)} GPUs)', unit='batch', mininterval=0.5):
            cpu_indices = make_cpu_slot_indices(mask)
            futures = [pools[d].submit(_predict_device_batch, d, groups[d], imgs, mask, cpu_indices) for d in devices]
            results, error = ([], None)
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            got = dict(results)
            if len(results) != len(models) or set(got) != set(fold_order):
                raise RuntimeError('Missing/duplicate fold result in GPU dispatch.')
            for fold in fold_order:
                if got[fold].shape != (len(imgs), len(TARGETS)):
                    raise RuntimeError(f'fold{fold}: unexpected output shape {got[fold].shape}')
                per_fold[fold].append(got[fold])
            n_seen += len(imgs)
    seconds = time.perf_counter() - start
    logits = np.stack([np.concatenate(per_fold[f], axis=0) for f in fold_order])
    print(f'[timing] {n_seen} studies / {len(models)} folds: {seconds:.2f}s, {n_seen / max(seconds, 1e-09):.3f} studies/s (includes loader/H2D/D2H)')
    return (fold_order, logits)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _rank_normalize(p):
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for j in range(p.shape[1]):
        s = pd.Series(p[:, j]).rank(method='average').to_numpy()
        out[:, j] = (s - 1) / max(n - 1, 1)
    return out

def combine_folds(logits, mode):
    if mode == 'logit':
        return _sigmoid(logits.mean(axis=0))
    if mode == 'prob':
        return _sigmoid(logits).mean(axis=0)
    if mode == 'rank':
        probs = _sigmoid(logits)
        return np.stack([_rank_normalize(probs[k]) for k in range(probs.shape[0])]).mean(axis=0)
    raise ValueError(f'ENSEMBLE 必须是 logit | prob | rank，得到 {mode!r}')

def report_fold_agreement(logits, fold_order):
    n_folds, n, n_tgt = logits.shape
    if n_folds < 2 or n < 30:
        print(f'\n样本数 {n} 太少 (rank 后取值被量化成少数几档)，fold 一致性检查无意义，跳过。')
        return
    probs = _sigmoid(logits)
    ranks = [_rank_normalize(probs[k]) for k in range(n_folds)]
    print('\nfold 间 Spearman 相关 (per-target，列出最低的一对):')
    for j, t in enumerate(TARGETS):
        pairs = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                r = float(np.corrcoef(ranks[a][:, j], ranks[b][:, j])[0, 1])
                pairs.append((r, fold_order[a], fold_order[b]))
        rs = [p[0] for p in pairs]
        lo, fa, fb = min(pairs)
        flag = '  <-- 偏低' if lo < 0.7 else ''
        print(f'  {t:18s} mean {np.mean(rs):+.4f} | min {lo:+.4f} (fold{fa} vs fold{fb}){flag}')
''',
'tiny': r'''
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import time
import math
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
DATA_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
TEST_DICOM_DIR = DATA_ROOT / 'test_series'
TEST_CSV = DATA_ROOT / 'test.csv'
TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
MODEL_INPUT_DIR = Path('/kaggle/input/datasets/minminandy/distillstage1tinyvit384/rsna_tinyvit21m384_oai8649_mix58_42_seven')
RUN_DIR_NAME = 'rsna_tinyvit21m384_oai8649_mix58_42_seven'
RUN_DIR = MODEL_INPUT_DIR
WORK_DIR = Path('/kaggle/working')
TEST_CACHE_DIR = WORK_DIR / 'knee_v5c80_384px_24sl_crop130_5slot'
TEST_CACHE_PATH = TEST_CACHE_DIR / 'test_cache.npy'
TEST_MASK_PATH = TEST_CACHE_DIR / 'test_mask.npy'
TEST_IDS_PATH = TEST_CACHE_DIR / 'test_study_ids.npy'
SUBMISSION_PATH = WORK_DIR / 'submission.csv'
FOLDS = (0, 1, 2, 3)
CKPT_NAME = 'fold0_final.pt'
ENSEMBLE = 'logit'
SAVE_PER_FOLD_PRED = True
USE_MULTI_GPU = True
TTA_ANGLES = (-12.0, 0.0, 12.0)
EXPECTED_MODEL_NAME = 'tiny_vit_21m_384.dist_in22k_ft_in1k'
EXPECTED_FEATURE_DIM = 576
EXPECTED_ARCH = 'shared_tinyvit21m_2p5d_11tokens_5expert_labelwise_spatiotemporal_attention_v3'
BACKBONE_CHUNK_SIZE = 0
PRINT_FUSION_WEIGHTS = True
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
IMG_SIZE = 384
CROP_MM = 130.0
N_SLOTS = 5
SLICES_PER_SLOT = 24
GROUP_SIZE = 3
N_GROUPS = 8
TOKEN_WINDOW = GROUP_SIZE
TOKEN_STRIDE = 2
N_TOKENS = (SLICES_PER_SLOT - TOKEN_WINDOW) // TOKEN_STRIDE + 1
IN_CHANS = GROUP_SIZE
CENTRAL_FRACTION = 0.8
SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
TRAIN_SLOT_FILL = {'SAG_FLUID_FS': 0.946, 'COR_FLUID_FS': 0.963, 'AX_FLUID_FS': 1.0, 'COR_T1': 0.766, 'SAG_T1': 0.97}
TRAIN_SAG_FLIP_RATE = 0.494
TRAIN_LAT_COVERAGE = 0.998
TRAIN_BRIGHTNESS = (57.3, 89.2, 81.9)
TRAIN_SAMPLING_MEAN_UNIQUE = 23.41
COLUMN_ALIASES = {'study': ['StudyInstanceUID', 'Study Instance UID'], 'series': ['SeriesInstanceUID', 'Series Instance UID'], 'plane': ['Plane', 'AnatomicalPlane', 'Anatomical Plane', 'Anatomical_Plane'], 'fluid': ['Fluid_Sensitive', 'Fluid Sensitive', 'FluidSensitive'], 'fatsat': ['Fat_Suppression', 'Fat Suppression', 'FatSuppression']}
LAT_MIN_OFFSET_MM = 5.0
EVAL_BATCH_STUDIES = 2
NUM_WORKERS = 0
HEADER_WORKERS = 12
PIXEL_WORKERS = 8
CPU_THREADS = 1
FIXED_FP16_SANITY_CHECK = True
FP16_MAX_LOGIT_DIFF = 0.05
FP16_MAX_PROB_DIFF = 0.01
VERIFY_MODE = False
VERIFY_N_STUDIES = 3
TRAIN_CACHE_DIRS = (Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0'), Path('/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2'), Path('/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3'))
TRAIN_DICOM_DIR = DATA_ROOT / 'train_series'
TRAIN_SERIES_CSV = DATA_ROOT / 'train_series.csv'
TRAIN_CSV_CANDIDATES = (DATA_ROOT / 'train.csv', Path('/kaggle/input/datasets/minminandy/rsna-file/train.csv'))
torch.backends.cudnn.benchmark = True
_ORDER_TAGS = [(32, 19), (32, 50), (32, 55)]

class TestStudyDataset(Dataset):

    def __init__(self, cache, mask):
        self.cache = cache
        self.mask = mask

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, i):
        imgs = torch.from_numpy(np.array(self.cache[i]))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[i])).float()
        return (imgs, m)

def rotate_grayscale_batch(x, angle_deg):
    if float(angle_deg) == 0.0:
        return x
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Rotation TTA expects square [N,1,H,W] grayscale images.')
    with torch.autocast(device_type=x.device.type, enabled=False):
        radians = math.radians(float(angle_deg))
        c, s = (math.cos(radians), math.sin(radians))
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=x.device).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class KneeSharedConvNeXtNet(nn.Module):

    def __init__(self, model_name: str, gru_hidden=256, gru_layers=1, attn_hidden=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=IN_CHANS, drop_path_rate=0.0)
        self.feature_dim = int(self.backbone.num_features)
        self.seq_dim = gru_hidden * 2
        self.fixed_fp16_backbone = False
        self.grus = nn.ModuleList([nn.GRU(input_size=self.feature_dim, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True) for _ in range(N_SLOTS)])
        self.spatial_attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.attentions = nn.ModuleList([nn.Sequential(nn.Linear(self.seq_dim, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, len(TARGETS))) for _ in range(N_SLOTS)])
        self.expert_heads = nn.ModuleList([nn.Linear(self.seq_dim, len(TARGETS)) for _ in range(N_SLOTS)])
        self.fusion_gate_logits = nn.Parameter(torch.zeros(len(TARGETS), N_SLOTS, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))
        shape = (1, IN_CHANS, 1, 1)
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('std', torch.ones(shape))

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float16 if self.fixed_fp16_backbone else torch.float32)
            c = int(BACKBONE_CHUNK_SIZE)
            if c <= 0 or x.shape[0] <= c:
                feat = self.backbone.forward_features(x)
            else:
                parts = [self.backbone.forward_features(x[i:i + c]) for i in range(0, x.shape[0], c)]
                feat = torch.cat(parts, dim=0)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat) or feat.ndim != 4:
            raise RuntimeError(f"backbone forward_features 预期返回4维空间图，实际类型/shape: {type(feat)} / {getattr(feat, 'shape', None)}")
        if feat.shape[1] == self.feature_dim:
            pass
        elif feat.shape[-1] == self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f'backbone 空间图找不到 feature_dim={self.feature_dim} 的通道轴: {tuple(feat.shape)}')
        return feat

    def _apply_backbone_head(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3 or feat.shape[-1] != self.feature_dim:
            raise RuntimeError(f'backbone head 预期 [N,T,{self.feature_dim}]，得到 {tuple(feat.shape)}')
        n, t, d = feat.shape
        x = feat.reshape(n * t, d, 1, 1)
        try:
            x = self.backbone.forward_head(x, pre_logits=True)
        except TypeError:
            x = self.backbone.head(x, pre_logits=True)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[-1] != d:
            raise RuntimeError(f"backbone head 预期返回 [{n * t},{d}]，实际 {type(x)} / {getattr(x, 'shape', None)}")
        return x.reshape(n, t, d)

    def _encode_slot(self, slot: int, imgs: torch.Tensor, valid_idx: torch.Tensor, rotation_deg: float=0.0) -> torch.Tensor:
        G, C, R = (N_TOKENS, TOKEN_WINDOW, IMG_SIZE)
        x = imgs[:, slot].index_select(0, valid_idx)
        bv = x.shape[0]
        if rotation_deg != 0.0:
            x = rotate_grayscale_batch(x.reshape(bv * SLICES_PER_SLOT, 1, R, R), rotation_deg).reshape(bv, SLICES_PER_SLOT, R, R)
        x = x.unfold(1, C, TOKEN_STRIDE)
        if x.shape[1] != G:
            raise RuntimeError(f'unfold 得到 {x.shape[1]} 个 token，预期 {G} —— SLICES_PER_SLOT/TOKEN_WINDOW/TOKEN_STRIDE 配置不一致')
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(bv * G, C, R, R).float().div(255.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)
        feat_map = self._run_backbone_features(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float().flatten(2).transpose(1, 2)
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)
            label_tokens = torch.einsum('npt,npd->ntd', spatial_a, pixels)
            label_tokens = self._apply_backbone_head(label_tokens)
            label_tokens = label_tokens.reshape(bv, G, len(TARGETS), self.feature_dim).permute(0, 2, 1, 3).contiguous()
            h = label_tokens.reshape(bv * len(TARGETS), G, self.feature_dim)
            h, _ = self.grus[slot](h)
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)
            temporal_all = self.attentions[slot](h)
            temporal_a = temporal_all.diagonal(dim1=1, dim2=3).permute(0, 2, 1).softmax(dim=-1)
            label_feat = torch.einsum('btg,btgd->btd', temporal_a, h)
            head = self.expert_heads[slot]
            return torch.einsum('btd,td->bt', label_feat, head.weight) + head.bias

    def forward(self, imgs, mask, *, valid_indices=None, return_expert_logits: bool=False, rotation_deg: float=0.0):
        B = imgs.shape[0]
        expected = (N_SLOTS, SLICES_PER_SLOT, IMG_SIZE, IMG_SIZE)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(f'期望 imgs [B,{N_SLOTS},{SLICES_PER_SLOT},{IMG_SIZE},{IMG_SIZE}]，得到 {tuple(imgs.shape)}')
        available = mask > 0.5
        per_slot = []
        for s in range(N_SLOTS):
            valid_idx = valid_indices[s] if valid_indices is not None else torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=torch.float32)
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx, rotation_deg=rotation_deg)
                full = torch.zeros(B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)
        expert_logits = torch.stack(per_slot, dim=1)
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -10000.0)
            gate = gate.softmax(dim=-1)
            logits = (gate * expert_logits.float().permute(0, 2, 1)).sum(dim=-1) + self.fusion_bias.float()
        if return_expert_logits:
            return (logits, expert_logits)
        return logits
VAL_SEQ_STRATEGIES = ('11_fixed_overlap_3slice_2p5d_tokens_noaug',)

def _check_sequence_layout(ckpt, fold):
    for key, want in (('n_groups', N_GROUPS), ('group_size', GROUP_SIZE), ('sequence_tokens_per_slot', N_TOKENS)):
        got = ckpt.get(key)
        if got is None:
            print(f'WARNING: fold{fold} ckpt 缺少 {key}，跳过该项校验')
            continue
        if int(got) != want:
            raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符' + (f'\n  (sequence_tokens_per_slot=24 说明这是上一版 ConvNeXt graycopy24 的 ckpt，不能用这份推理代码)' if key == 'sequence_tokens_per_slot' and int(got) == SLICES_PER_SLOT else ''))
    sps = ckpt.get('slices_per_slot')
    if sps is not None and int(sps) != N_GROUPS * GROUP_SIZE:
        raise ValueError(f'fold{fold} slices_per_slot={sps} != n_groups*group_size={N_GROUPS * GROUP_SIZE}')
    mode = ckpt.get('input_mode')
    if mode is None:
        raise ValueError(f'fold{fold} ckpt 没有 input_mode 字段，无法确认输入构造。')
    if mode != '3slice_2p5d_overlap_window3_stride2':
        raise ValueError(f'fold{fold} input_mode={mode!r}，与本文件实现的「3 层 2.5D 重叠三元组 (窗口 3 / 步长 2)」不符 —— single_grayscale_slice_repeated_to_3_channels 是上一版 graycopy24。')
    sa = ckpt.get('spatial_attention')
    if sa is None:
        raise ValueError(f'fold{fold} ckpt 没有 spatial_attention 字段，多半是只有时序注意力的旧版产物；本文件多一组 spatial_attentions.* 权重，strict=True 会报 missing keys。')
    if sa != 'per_slot_per_label_softmax_over_final_convnext_spatial_map':
        raise ValueError(f'fold{fold} spatial_attention={sa!r}，与本文件实现的 label-wise 空间注意力不符。')
    ta = ckpt.get('temporal_attention')
    if ta is None:
        raise ValueError(f'fold{fold} ckpt 没有 temporal_attention 字段，多半是(所有标签共用一套注意力) 的旧版产物，前向不同，不能用这份推理代码。')
    if ta != 'per_slot_per_label_softmax_over_6_tokens':
        raise ValueError(f'fold{fold} temporal_attention={ta!r}，与本文件实现的 label-wise 注意力不符。')
    vs = ckpt.get('val_sequence_strategy')
    if vs is None:
        if ckpt.get('val_group_strategy') is not None:
            raise ValueError(f"fold{fold} 只有旧版的 val_group_strategy={ckpt.get('val_group_strategy')!r}，不是本版训练产物。")
        print(f'WARNING: fold{fold} 无 val_sequence_strategy，跳过验证端策略校验')
    elif vs not in VAL_SEQ_STRATEGIES:
        raise ValueError(f'fold{fold} val_sequence_strategy={vs!r} 不在已知列表 {VAL_SEQ_STRATEGIES}；验证端前向可能与本文件不同，不要混用。')

def load_fold_models(devices, run_dir):
    models = []
    ref_name = None
    fusion_stack = []
    for k, (fold, path) in enumerate(resolve_checkpoints(run_dir)):
        dev = devices[k % len(devices)]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' not in ckpt:
            raise KeyError(f'{path} 中找不到 state_dict')
        arch = ckpt.get('architecture')
        if EXPECTED_ARCH is not None and arch != EXPECTED_ARCH:
            raise ValueError(f'fold{fold} architecture={arch!r}，但本文件实现的是 {EXPECTED_ARCH!r}。\nrun_dir 可能挂错了 run。\nrun_dir = {run_dir}')
        if ckpt.get('independent_encoder_per_slot') is True:
            raise ValueError(f'fold{fold} 的 ckpt 是 5 独立 encoder 版本，不能用这份共享 backbone 的推理代码。')
        if ckpt.get('shared_backbone_across_slots') is not True:
            raise ValueError(f'fold{fold} 的 ckpt 没有 shared_backbone_across_slots=True，不是共享 backbone 版本的训练产物。')

        def need(key, want, cast=int, tol=None):
            got = ckpt.get(key)
            if got is None:
                return
            got = cast(got)
            ok = abs(got - want) <= tol if tol is not None else got == want
            if not ok:
                raise ValueError(f'fold{fold} {key}={got}，与推理端 {want} 不符')
        need('img_size', IMG_SIZE)
        need('n_slots', N_SLOTS)
        need('slices_per_slot', SLICES_PER_SLOT)
        need('crop_mm', CROP_MM, cast=float, tol=1.0)
        ckpt_targets = ckpt.get('targets')
        if ckpt_targets is not None and list(ckpt_targets) != TARGETS:
            raise ValueError(f'fold{fold} targets 与当前 TARGETS 不一致。\ncheckpoint: {ckpt_targets}\ncurrent   : {TARGETS}')
        ckpt_slot_names = ckpt.get('slot_names')
        expected_slot_names = [s[0] for s in SLOTS]
        if ckpt_slot_names is not None and list(ckpt_slot_names) != expected_slot_names:
            raise ValueError(f'fold{fold} slot_names 与推理端不一致。\ncheckpoint: {list(ckpt_slot_names)}\ncurrent   : {expected_slot_names}')
        if ckpt_slot_names is None and k == 0:
            print('  (ckpt 无 slot_names 字段，跳过 slot 顺序校验 —— SLOTS 必须与预处理端逐字一致，时序专家是逐 slot 的)')
        _check_sequence_layout(ckpt, fold)
        slice_order = ckpt.get('slice_order')
        if slice_order is not None and slice_order != 'instance_number':
            print(f'WARNING: fold{fold} slice_order={slice_order!r}，与推理端预期不一致!')
        model_name = ckpt.get('model_name')
        if model_name is None:
            raise KeyError(f'{path} 里没有 model_name，多半是旧版 checkpoint，与当前训练代码不兼容。')
        if EXPECTED_MODEL_NAME is not None and model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f'fold{fold} backbone={model_name}，但 EXPECTED_MODEL_NAME={EXPECTED_MODEL_NAME}。\n!! 本版与上一版 ConvNeXt graycopy24 的 architecture 字符串相同，这道校验是分辨两者的唯一干净手段 —— 不要为了跑通而放宽它。\nrun_dir = {run_dir}')
        if ref_name is None:
            ref_name = model_name
        elif model_name != ref_name:
            raise ValueError(f'fold{fold} backbone ({model_name}) 与 fold{FOLDS[0]} ({ref_name}) 不同')
        model = KneeSharedConvNeXtNet(model_name, gru_hidden=int(ckpt.get('gru_hidden', 256)), gru_layers=int(ckpt.get('gru_layers', 1)), attn_hidden=int(ckpt.get('attn_hidden', 128)))
        ckpt_fd = ckpt.get('feature_dim')
        if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
            raise ValueError(f'fold{fold} feature_dim={ckpt_fd}，实例化得到 {model.feature_dim}')
        ckpt_sq = ckpt.get('seq_dim')
        if ckpt_sq is not None and int(ckpt_sq) != model.seq_dim:
            raise ValueError(f'fold{fold} seq_dim={ckpt_sq}，实例化得到 {model.seq_dim}')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        ref_fusion = ckpt.get('fusion_weights_no_missing')
        got_fusion = model.fusion_weights().detach()
        if ref_fusion is not None:
            d = float((torch.as_tensor(ref_fusion).float() - got_fusion).abs().max())
            if d > 0.0001:
                raise RuntimeError(f'fold{fold} 融合权重与 ckpt 存的 fusion_weights_no_missing 不一致 (max|Δ|={d:.2e}) —— gate 参数没有正确载入。')
        fusion_stack.append(got_fusion.cpu().numpy())
        models.append((fold, model.to(dev, memory_format=torch.channels_last).eval(), dev))
        print(f"  fold{fold} -> {dev} | {path.name} | epoch {ckpt.get('epoch', 'NA')} | derived {_fmt(ckpt.get('derived_auc'))} | gold {_fmt(ckpt.get('gold_auc'))} | select {_fmt(ckpt.get('selection_score'))} ({ckpt.get('selection', 'NA')}, n_gold_val={ckpt.get('n_gold_val', 'NA')}) | EMA={ckpt.get('ema', 'NA')}(decay={ckpt.get('ema_decay', 'NA')}) | aux_w={ckpt.get('expert_aux_weight', 'NA')}")
        if ckpt.get('aug'):
            print(f"         train aug: {ckpt['aug']}")
        if ckpt.get('label_mix') and k == 0:
            print(f"         label mix: {ckpt['label_mix']}")
    ref_model = models[0][1]
    mean = [round(v, 4) for v in ref_model.mean.flatten().tolist()]
    std = [round(v, 4) for v in ref_model.std.flatten().tolist()]
    print(f'\nnormalize: mean {mean} | std {std}')
    if all((v == 0.0 for v in mean)) or all((v == 1.0 for v in std)):
        raise RuntimeError('mean/std 仍是初始占位值，checkpoint 里没有这两个 buffer！')
    n_param = sum((p.numel() for p in ref_model.parameters()))
    print(f'backbone: {ref_name} x1 (shared over {N_SLOTS} slots) | feature_dim {ref_model.feature_dim} | seq_dim {ref_model.seq_dim} | {n_param / 1000000.0:.1f}M params/fold | {len(models)} folds on {len(devices)} device(s)')
    if ref_model.feature_dim != EXPECTED_FEATURE_DIM:
        print(f'  !! feature_dim={ref_model.feature_dim}，TinyViT-21M 预期 {EXPECTED_FEATURE_DIM}')
    print('^^ 请把上面的 epoch / derived / gold 与训练日志逐行核对；对不上说明 run_dir 挂错了 run。')
    if PRINT_FUSION_WEIGHTS and fusion_stack:
        w = np.mean(np.stack(fusion_stack), axis=0)
        names = [s[0] for s in SLOTS]
        print('\nlabel-wise 融合权重 (无缺失 slot 时, %d fold 平均):' % len(fusion_stack))
        print('  ' + ' ' * 18 + ''.join((f'{n:>14s}' for n in names)))
        for j, t in enumerate(TARGETS):
            row = ''.join((f'{w[j, i]:14.4f}' for i in range(N_SLOTS)))
            top = names[int(np.argmax(w[j]))]
            print(f'  {t:18s}{row}   <- {top}')
        print('  (缺失 slot 在 softmax 前被屏蔽，实际权重按可用 slot 重新归一化)')
    return models

def resolve_backbone_head(backbone):
    mod = getattr(backbone, 'head', None)
    if mod is None or not isinstance(mod, nn.Module):
        return (None, None, 0)
    return ('head', mod, sum((p.numel() for p in mod.parameters())))

def clear_attention_bias_cache(module):
    n = 0
    for m in module.modules():
        if hasattr(m, 'attention_bias_cache'):
            m.attention_bias_cache = {}
            n += 1
    return n

@torch.inference_mode()
def prepare_fixed_fp16(models, loader):
    if len(loader.dataset) == 0:
        raise RuntimeError('No test studies to predict.')
    sample = next(iter(loader)) if FIXED_FP16_SANITY_CHECK else None
    cpu_indices = make_cpu_slot_indices(sample[1]) if sample is not None else None
    for fold, model, dev in models:
        if dev.type != 'cuda':
            raise RuntimeError('This fixed-FP16 inference script requires CUDA.')
        with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
            ref = None
            if sample is not None:
                print(f'[precision] fold{fold}: FP32 reference -> fixed FP16 check', flush=True)
                im = sample[0].to(dev, non_blocking=True)
                mk = sample[1].to(dev, non_blocking=True)
                ix = [v.to(dev, non_blocking=True) for v in cpu_indices]
                ref = [model(im, mk, valid_indices=ix, rotation_deg=a).float() for a in TTA_ANGLES]
                if not all((torch.isfinite(v).all().item() for v in ref)):
                    raise RuntimeError(f'fold{fold}: FP32 reference contains NaN/Inf.')
            head_name, head_mod, n_head_param = resolve_backbone_head(model.backbone)
            original_head = None
            if head_mod is not None:
                original_head = {k: v.detach().cpu().clone() for k, v in head_mod.state_dict().items()}
            model.backbone.half()
            if head_mod is not None:
                head_mod = getattr(model.backbone, head_name)
                head_mod.float()
                head_mod.load_state_dict(original_head, strict=True)
            model.fixed_fp16_backbone = True
            n_cleared = clear_attention_bias_cache(model.backbone)
            if n_cleared:
                print(f'  cleared {n_cleared} attention_bias_cache entries')
            n_bn, n_sub, n_zero = (0, 0, 0)
            for m in model.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                    n_bn += 1
                    v = m.running_var.float()
                    n_sub += int((v < 6.104e-05).sum())
                    n_zero += int((v == 0).sum())
            if n_bn:
                flag = '   <-- 看下面的对拍' if n_sub or n_zero else ''
                print(f'  BatchNorm: {n_bn} 层 | running_var 落入 fp16 次正规区 (<6.1e-5) 的通道 {n_sub} 个 | 归零 {n_zero} 个{flag}')
            head_prefix = f'backbone.{head_name}.' if head_name else None
            for name, p in model.named_parameters():
                half_feature = name.startswith('backbone.') and (not (head_prefix and name.startswith(head_prefix)))
                expected = torch.float16 if half_feature else torch.float32
                if p.is_floating_point() and p.dtype != expected:
                    raise RuntimeError(f'Unexpected dtype: {name} {p.dtype}, expected {expected}')
            if sample is not None:
                for angle, reference in zip(TTA_ANGLES, ref):
                    actual = model(im, mk, valid_indices=ix, rotation_deg=angle).float()
                    finite = torch.isfinite(actual).all().item()
                    dl = float((actual - reference).abs().max().item())
                    dp = float((actual.sigmoid() - reference.sigmoid()).abs().max().item())
                    print(f'  angle={angle:+g}: max|delta logit|={dl:.6f}, max|delta probability|={dp:.6f}', flush=True)
                    if not finite or dl > FP16_MAX_LOGIT_DIFF or dp > FP16_MAX_PROB_DIFF:
                        raise RuntimeError(f'fold{fold}, angle={angle}: FP16 check failed; logit={dl}, probability={dp}, finite={finite}')
                del im, mk, ix, ref, reference, actual
            head_txt = f'backbone.{head_name} ({n_head_param / 1000.0:.1f}K params) FP32' if head_mod is not None else 'backbone 无 head 子模块 (EffNet 的 forward_head 无参数)'
            print(f'[precision] fold{fold}: feature extractor FP16; {head_txt}; attention/GRU/fusion FP32')

def make_cpu_slot_indices(mask):
    if mask.device.type != 'cpu':
        raise ValueError('Build indices from the original CPU mask before H2D.')
    return [torch.nonzero(mask[:, s] > 0.5, as_tuple=False).flatten().pin_memory() for s in range(N_SLOTS)]

@torch.inference_mode()
def _predict_device_batch(dev, fold_models, imgs, mask, cpu_indices):
    with torch.cuda.device(dev), torch.autocast('cuda', enabled=False):
        im = imgs.to(dev, non_blocking=True)
        mk = mask.to(dev, non_blocking=True)
        indices = [v.to(dev, non_blocking=True) for v in cpu_indices]
        result = []
        for fold, model in fold_models:
            logits = None
            for angle in TTA_ANGLES:
                view_logits = model(im, mk, valid_indices=indices, rotation_deg=angle).float()
                logits = view_logits if logits is None else logits + view_logits
            logits = logits / len(TTA_ANGLES)
            values = logits.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f'fold{fold}: prediction contains NaN/Inf.')
            result.append((fold, values))
        return result

@torch.inference_mode()
def predict_test(models, loader):
    devices = sorted({d for _, _, d in models}, key=str)
    if any((d.type != 'cuda' for d in devices)):
        raise RuntimeError('Fixed-FP16 inference requires CUDA devices.')
    groups = {d: [(f, m) for f, m, md in models if md == d] for d in devices}
    fold_order = [f for f, _, _ in models]
    if len(set(fold_order)) != len(fold_order):
        raise RuntimeError('Duplicate fold IDs would misalign predictions.')
    per_fold = {f: [] for f in fold_order}
    start = time.perf_counter()
    n_seen = 0
    with ExitStack() as stack:
        pools = {d: stack.enter_context(ThreadPoolExecutor(max_workers=1)) for d in devices}
        for imgs, mask in tqdm(loader, total=len(loader), desc=f'Inference ({len(models)} folds / {len(devices)} GPUs)' if tuple(TTA_ANGLES) == (0.0,) else f'Rotation TTA{len(TTA_ANGLES)} ({len(models)} folds / {len(devices)} GPUs)', unit='batch', mininterval=0.5):
            cpu_indices = make_cpu_slot_indices(mask)
            futures = [pools[d].submit(_predict_device_batch, d, groups[d], imgs, mask, cpu_indices) for d in devices]
            results, error = ([], None)
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            got = dict(results)
            if len(results) != len(models) or set(got) != set(fold_order):
                raise RuntimeError('Missing/duplicate fold result in GPU dispatch.')
            for fold in fold_order:
                if got[fold].shape != (len(imgs), len(TARGETS)):
                    raise RuntimeError(f'fold{fold}: unexpected output shape {got[fold].shape}')
                per_fold[fold].append(got[fold])
            n_seen += len(imgs)
    seconds = time.perf_counter() - start
    logits = np.stack([np.concatenate(per_fold[f], axis=0) for f in fold_order])
    print(f'[timing] {n_seen} studies / {len(models)} folds: {seconds:.2f}s, {n_seen / max(seconds, 1e-09):.3f} studies/s (includes loader/H2D/D2H)')
    return (fold_order, logits)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _rank_normalize(p):
    out = np.empty_like(p, dtype=np.float64)
    n = p.shape[0]
    for j in range(p.shape[1]):
        s = pd.Series(p[:, j]).rank(method='average').to_numpy()
        out[:, j] = (s - 1) / max(n - 1, 1)
    return out

def combine_folds(logits, mode):
    if mode == 'logit':
        return _sigmoid(logits.mean(axis=0))
    if mode == 'prob':
        return _sigmoid(logits).mean(axis=0)
    if mode == 'rank':
        probs = _sigmoid(logits)
        return np.stack([_rank_normalize(probs[k]) for k in range(probs.shape[0])]).mean(axis=0)
    raise ValueError(f'ENSEMBLE 必须是 logit | prob | rank，得到 {mode!r}')

def report_fold_agreement(logits, fold_order):
    n_folds, n, n_tgt = logits.shape
    if n_folds < 2 or n < 30:
        print(f'\n样本数 {n} 太少 (rank 后取值被量化成少数几档)，fold 一致性检查无意义，跳过。')
        return
    probs = _sigmoid(logits)
    ranks = [_rank_normalize(probs[k]) for k in range(n_folds)]
    print('\nfold 间 Spearman 相关 (per-target，列出最低的一对):')
    for j, t in enumerate(TARGETS):
        pairs = []
        for a in range(n_folds):
            for b in range(a + 1, n_folds):
                r = float(np.corrcoef(ranks[a][:, j], ranks[b][:, j])[0, 1])
                pairs.append((r, fold_order[a], fold_order[b]))
        rs = [p[0] for p in pairs]
        lo, fa, fb = min(pairs)
        flag = '  <-- 偏低' if lo < 0.7 else ''
        print(f'  {t:18s} mean {np.mean(rs):+.4f} | min {lo:+.4f} (fold{fa} vs fold{fb}){flag}')
''',
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def _write_csv(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _new_module(name, source, inherited=None):
    module = types.ModuleType(name)
    if inherited is not None:
        module.__dict__.update({k: v for k, v in inherited.__dict__.items()
                                if not k.startswith('__')})
    module.__file__ = '<embedded:' + name + '>'
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def _resolve_run_root(spec):
    # 只接受这个 Dataset 的已核对目录；不在其他 Dataset 中全局搜索权重。
    modern = Path(spec['run_root'])
    legacy = Path('/kaggle/input') / spec['dataset_slug'] / spec['run_name']
    candidates = [modern, legacy]
    for root in candidates:
        paths = [root / f'fold{f}' / 'fold0_final.pt' for f in range(4)]
        if all(p.is_file() for p in paths):
            return root
    expected = '\n'.join(str(modern / f'fold{f}' / 'fold0_final.pt') for f in range(4))
    raise FileNotFoundError(
        f"{spec['key']}: 找不到完整四折权重。请挂载 {spec['dataset_url']}\n"
        f"需要的路径：\n{expected}\n旧式挂载路径也会检查：{legacy}")


def _checkpoint_file_identity(root):
    result = []
    for f in range(4):
        path = root / f'fold{f}' / 'fold0_final.pt'
        stat = path.stat()
        result.append({'fold': f, 'path': str(path), 'size': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns})
    return result


def _input_identity():
    required = ['test_series.csv', 'sample_submission.csv']
    for name in required:
        if not (DATA_ROOT / name).is_file():
            raise FileNotFoundError(f'缺少比赛文件：{DATA_ROOT / name}')
    if not (DATA_ROOT / 'test_series').is_dir():
        raise FileNotFoundError(f'缺少 test DICOM：{DATA_ROOT / "test_series"}')
    hashes = {name: _digest((DATA_ROOT / name).read_bytes())
              for name in required + ['test.csv'] if (DATA_ROOT / name).is_file()}
    return {'data_root': str(DATA_ROOT.resolve()), 'csv_sha256': hashes,
            'preprocessor_sha256': _digest(_COMMON_SOURCE.encode('utf-8')),
            'cache_layout': [5, 24, 384, 384], 'crop_mm': 130.0,
            'central_fraction': 0.8}


def _configure_common(common, identity):
    common.DATA_ROOT = DATA_ROOT
    common.TEST_DICOM_DIR = DATA_ROOT / 'test_series'
    common.TEST_CSV = DATA_ROOT / 'test.csv'
    common.TEST_SERIES_CSV = DATA_ROOT / 'test_series.csv'
    common.SAMPLE_SUBMISSION_CSV = DATA_ROOT / 'sample_submission.csv'
    common.WORK_DIR = OUTPUT_DIR
    common.SUBMISSION_PATH = OUTPUT_DIR / 'submission.csv'
    common.USE_MULTI_GPU = USE_MULTI_GPU
    common.CPU_THREADS = CPU_THREADS
    common.HEADER_WORKERS = HEADER_WORKERS
    common.PIXEL_WORKERS = PIXEL_WORKERS
    key = _digest(_canonical(identity).encode())[:20]
    common.TEST_CACHE_DIR = CACHE_ROOT / key
    common.TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    common.TEST_CACHE_PATH = common.TEST_CACHE_DIR / 'test_cache.npy'
    common.TEST_MASK_PATH = common.TEST_CACHE_DIR / 'test_mask.npy'
    common.TEST_IDS_PATH = common.TEST_CACHE_DIR / 'test_study_ids.npy'
    common.torch.set_num_threads(CPU_THREADS)
    common.torch.backends.cudnn.benchmark = True
    if common.TARGETS != TARGETS:
        raise ValueError('公共预处理 TARGETS 顺序不同')


def _load_shared_cache(common, identity):
    marker = common.TEST_CACHE_DIR / 'complete.json'
    paths = [common.TEST_CACHE_PATH, common.TEST_MASK_PATH, common.TEST_IDS_PATH]
    complete = False
    if marker.is_file() and all(p.is_file() for p in paths):
        complete = json.loads(marker.read_text(encoding='utf-8')) == identity
    if complete:
        cache, mask, ids = common.load_or_build_test_cache()
    else:
        # 只重建本脚本专属目录中的不完整缓存，不碰已有训练/OAI缓存。
        cache, mask, ids = common.build_test_cache()
    ids = np.asarray(ids).astype(str)
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError('缓存病例 ID 为空或重复')
    if cache.dtype != np.uint8 or cache.shape != (len(ids), 5, 24, 384, 384):
        raise ValueError(f'公共 cache shape/dtype 不正确：{cache.shape}/{cache.dtype}')
    if mask.shape != (len(ids), 5) or not np.isfinite(mask).all():
        raise ValueError('公共 mask 不正确')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('公共 mask 应为 0/1')
    expected = pd.read_csv(DATA_ROOT / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
    if 'StudyInstanceUID' not in expected or set(expected.StudyInstanceUID) != set(ids):
        raise ValueError('公共缓存与 sample_submission 的病例 ID 不一致')
    _write_json(marker, identity)
    common.check_mask(mask, ids)
    return cache, mask, ids


def _configure_backend(backend, common, spec, root):
    backend.MODEL_INPUT_DIR = root
    backend.RUN_DIR = root
    backend.RUN_DIR_NAME = root.name
    backend.FOLDS = (0, 1, 2, 3)
    backend.CKPT_NAME = 'fold0_final.pt'
    backend.ENSEMBLE = 'logit'
    backend.WORK_DIR = PREDICTION_DIR / spec['key']
    backend.WORK_DIR.mkdir(parents=True, exist_ok=True)
    backend.BACKBONE_CHUNK_SIZE = BACKBONE_CHUNK_SIZE
    backend.EVAL_BATCH_STUDIES = BATCH_STUDIES
    backend.NUM_WORKERS = 0
    backend.USE_MULTI_GPU = USE_MULTI_GPU
    backend.FIXED_FP16_SANITY_CHECK = FP16_FIRST_BATCH_CHECK
    angles = MODEL_TTA_ANGLES[spec['key']] if TTA_OVERRIDE is None else TTA_OVERRIDE
    backend.TTA_ANGLES = tuple(float(a) for a in angles)
    if not backend.TTA_ANGLES or not np.isfinite(backend.TTA_ANGLES).all():
        raise ValueError('TTA 角度为空或不合法')
    if len(backend.TTA_ANGLES) > 2 or 0.0 not in backend.TTA_ANGLES:
        raise ValueError('精简版每套最多两视图，且必须保留原图0度')
    if backend.TARGETS != TARGETS or backend.SLOTS != common.SLOTS:
        raise ValueError(f"{spec['key']}: 类别或槽位顺序不一致")
    if backend.EXPECTED_MODEL_NAME != spec['model_name']:
        raise ValueError(f"{spec['key']}: 后端 backbone 不一致")
    if backend.EXPECTED_FEATURE_DIM != spec['feature_dim']:
        raise ValueError(f"{spec['key']}: 后端 feature_dim 不一致")
    if hasattr(backend, 'cv2'):
        backend.cv2.setNumThreads(0)


def _read_predictions(path, ids):
    frame = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    if list(frame.columns) != ['StudyInstanceUID'] + TARGETS:
        raise ValueError(f'{path}: 预测列与顺序不一致')
    if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != set(ids):
        raise ValueError(f'{path}: 病例 ID 重复、缺失或多余')
    values = frame.set_index('StudyInstanceUID').loc[list(ids), TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f'{path}: 预测包含非有限值或超出0～1')
    return values


def _prediction_frame(ids, values):
    values = np.asarray(values)
    if values.shape != (len(ids), len(TARGETS)):
        raise ValueError(f'预测 shape 不正确：{values.shape}')
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('预测概率不合法')
    out = pd.DataFrame(values, columns=TARGETS)
    out.insert(0, 'StudyInstanceUID', np.asarray(ids).astype(str))
    return out


def _model_signature(spec, root, input_identity, ids, angles):
    key = spec['key']
    return {'input': input_identity, 'spec': spec,
            'checkpoint_files': _checkpoint_file_identity(root),
            'backend_sha256': _digest(_BACKEND_SOURCES[key].encode('utf-8')),
            'ordered_study_ids_sha256': _digest(_canonical(list(ids)).encode()),
            'targets': TARGETS, 'angles': list(angles), 'fold_ensemble': 'logit',
            'fixed_fp16_feature_extractor': True, 'fp32_attention_gru_head': True,
            'batch_studies': BATCH_STUDIES, 'backbone_chunk_size': BACKBONE_CHUNK_SIZE,
            'precision_check': FP16_FIRST_BATCH_CHECK}


def _run_one_model(common, spec, root, cache, mask, ids, input_identity, devices):
    key = spec['key']
    backend = _new_module('_rsna_six_' + key, _BACKEND_SOURCES[key], common)
    _configure_backend(backend, common, spec, root)
    directory = backend.WORK_DIR
    prediction_path = directory / 'test_probabilities.csv'
    receipt_path = directory / 'complete.json'
    signature = _model_signature(spec, root, input_identity, ids, backend.TTA_ANGLES)
    if RESUME and receipt_path.is_file() and prediction_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('signature') == signature and
                receipt.get('probability_csv_sha256') == _digest(prediction_path.read_bytes())):
            probabilities = _read_predictions(prediction_path, ids)
            print(f'[{key}] 复用已完成的预测：{prediction_path}', flush=True)
            return probabilities, {**receipt['summary'], 'resumed': True}
    print('\n' + '=' * 80, flush=True)
    print(f"[{key}] {spec['model_name']} | {spec['tokens_per_slot']} tokens | "
          f"input={spec['model_input_size']} | TTA={backend.TTA_ANGLES}", flush=True)
    print('weight root:', root, flush=True)
    loader = common.DataLoader(backend.TestStudyDataset(cache, mask),
                               batch_size=BATCH_STUDIES, shuffle=False,
                               num_workers=0, pin_memory=True, drop_last=False)
    models = None
    start = time.perf_counter()
    try:
        models = backend.load_fold_models(devices, root)
        for fold, model, dev in models:
            mean = model.mean.detach().float().cpu().numpy().ravel()
            std = model.std.detach().float().cpu().numpy().ravel()
            if not (np.allclose(mean, spec['mean'], atol=1e-6, rtol=0) and
                    np.allclose(std, spec['std'], atol=1e-6, rtol=0)):
                raise ValueError(f'{key}/fold{fold}: checkpoint 归一化与核对清单不同：{mean}/{std}')
        # 后端保留 strict state_dict、输入布局/尺寸/类别/feature_dim 和 gate 校验。
        # 首 batch 保留原 FP32 vs FP16 对拍，失败就停止，不放宽阈值。
        backend.prepare_fixed_fp16(models, loader)
        fold_order, logits = backend.predict_test(models, loader)
        if fold_order != [0, 1, 2, 3] or logits.shape != (4, len(ids), 12):
            raise ValueError(f'{key}: fold顺序或logits shape不正确')
        if not np.isfinite(logits).all():
            raise ValueError(f'{key}: logits包含NaN/Inf')
        probabilities = backend.combine_folds(logits, 'logit').astype(np.float64)
        _write_csv(prediction_path, _prediction_frame(ids, probabilities))
        if SAVE_FOLD_LOGITS:
            np.save(directory / 'test_fold_logits.npy', logits)
        summary = {'key': key, 'seconds': time.perf_counter() - start,
                   'tta_angles': list(backend.TTA_ANGLES), 'fold_order': fold_order,
                   'input_size': spec['model_input_size'], 'tokens': spec['tokens_per_slot'],
                   'model_name': spec['model_name'], 'root': str(root),
                   'probability_csv': str(prediction_path), 'resumed': False}
        _write_json(receipt_path, {'signature': signature,
                                 'probability_csv_sha256': _digest(prediction_path.read_bytes()),
                                 'summary': summary})
        print(f'[{key}] 完成，耗时 {summary["seconds"] / 60:.2f} 分钟', flush=True)
        return probabilities, summary
    finally:
        # 循环变量也持有模型；显式清理后下一套只占自身四折显存。
        if models is not None:
            models.clear()
        models = None
        model = None
        loader = None
        gc.collect()
        for dev in devices:
            with common.torch.cuda.device(dev):
                common.torch.cuda.synchronize(dev)
                common.torch.cuda.empty_cache()


def _make_final_submission(ids, predictions):
    if set(predictions) != {s['key'] for s in MODEL_SPECS} or len(predictions) != 6:
        raise ValueError('必须拿到六套完整概率后才能融合；不会跳过失败模型')
    # 每套恰好 1/6，不把24个fold或TTA视图混在一起重新加权。
    values = np.stack([predictions[s['key']] for s in MODEL_SPECS]).mean(axis=0)
    frame = _prediction_frame(ids, values)
    _write_csv(OUTPUT_DIR / 'ensemble_A_probabilities.csv', frame)
    sample = pd.read_csv(DATA_ROOT / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
    if sample.StudyInstanceUID.duplicated().any():
        raise ValueError('sample_submission含重复病例')
    if set(sample.StudyInstanceUID) != set(ids) or any(t not in sample for t in TARGETS):
        raise ValueError('sample_submission病例或类别不同')
    ordered = frame.set_index('StudyInstanceUID').loc[sample.StudyInstanceUID, TARGETS]
    out = sample.copy()
    out[TARGETS] = ordered.to_numpy()
    if not np.isfinite(out[TARGETS].to_numpy()).all():
        raise ValueError('最终提交包含非有限值')
    _write_csv(OUTPUT_DIR / 'submission.csv', out)
    return out


def main():
    started = time.perf_counter()
    if BATCH_STUDIES < 1 or CPU_THREADS < 1:
        raise ValueError('batch/CPU线程数必须为正数')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    roots = {s['key']: _resolve_run_root(s) for s in MODEL_SPECS}
    input_identity = _input_identity()
    common = _new_module('_rsna_six_shared', _COMMON_SOURCE)
    _configure_common(common, input_identity)
    devices = common.get_devices()
    if any(d.type != 'cuda' for d in devices):
        raise RuntimeError('请选择Kaggle GPU。推荐T4 x2；本代码不在CPU执行固定FP16前向。')
    print('RSNA 原六模型等权融合 A；输出 submission.csv', flush=True)
    print('GPU:', [str(d) for d in devices], 'timm:', common.timm.__version__, flush=True)
    print('各模型内部：TTA/fold logit mean -> sigmoid；六模型之间：概率1/6平均', flush=True)
    for spec in MODEL_SPECS:
        print(spec['key'], '->', roots[spec['key']], flush=True)
    cache, mask, ids = _load_shared_cache(common, input_identity)
    predictions, reports = {}, []
    for spec in MODEL_SPECS:
        values, report = _run_one_model(common, spec, roots[spec['key']],
                                         cache, mask, ids, input_identity, devices)
        predictions[spec['key']] = values
        reports.append(report)
    submission = _make_final_submission(ids, predictions)
    report = {'ensemble': 'A_original_six_equal_probability',
              'weights': {s['key']: 1 / 6 for s in MODEL_SPECS},
              'targets': TARGETS, 'test_studies': len(ids), 'models': reports,
              'input_identity': input_identity,
              'elapsed_seconds': time.perf_counter() - started,
              'submission': str(OUTPUT_DIR / 'submission.csv')}
    _write_json(OUTPUT_DIR / 'six_model_inference_report.json', report)
    print('\n六套全部完成。submission.csv:', submission.shape, flush=True)
    print(submission.head(), flush=True)
    print(f'本次总耗时 {report["elapsed_seconds"] / 60:.2f} 分钟', flush=True)
    print('本代码只生成文件，不自动提交。', flush=True)


if __name__ == '__main__':
    main()
