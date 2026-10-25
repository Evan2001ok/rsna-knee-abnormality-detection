# DINOv3-distilled ConvNeXt-Tiny | latest RSNA/OAI training recipe.
# Source: eff-of-fork-of-rsna-train-cd4e8a 6(1).ipynb, executed code cell 2.
# 384 px, 24 cached slices, 11 overlapping 2.5D triplets, four GroupKFold folds.
# Latest VLM/LLM labels, Gold override/weight 3, baseline+follow-up OAI pool,
# participant-balanced RSNA/OAI 58/42 epoch sampling, augmentations, ranking loss,
# EMA, optimizer and checkpoint selection remain from the source cell.
# Previously requested DINOv3 duration: 16 epochs per fold.
# Offline pretrained weight:
# /kaggle/input/notebooks/lihu6666/dinvo3convnext-timm/dinov3_convnext_tiny_pretrained/model.safetensors

"""RSNA Knee training: DINOv3 ConvNeXt-Tiny with the latest OAI 58/42 recipe."""
from __future__ import annotations

import gc
import hashlib
import logging
import os
import random
import re
import signal
import time
import unicodedata
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

# ---- 全局警告静默 ----
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("multiprocessing").setLevel(logging.ERROR)

# DataLoader 的 _shutdown_workers 在 join 超时后会放弃等待，退出的 worker 滞留成
# 僵尸并占住 fd；堆积到一定量级 fork/fd 分配开始阻塞，表现为训练中途卡死。
# 交给内核自动回收。注意: 设置后 subprocess.wait() 拿不到退出码，本脚本不用 subprocess。
if hasattr(signal, "SIGCHLD"):
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

# multiprocessing 在 DataLoader worker 退出时会从三个路径产生无害报错:
#   1. QueueFeederThread._feed -> OSError/ValueError  (threading 直接打 stderr)
#   2. _ConnectionBase.__del__ -> OSError              (sys.unraisablehook)
#   3. Pool.__del__ -> OSError                         (sys.unraisablehook)
# warnings.filterwarnings 和 logging 都压不住，需要分别处理。

# (A) patch _feed: 吞掉 QueueFeederThread 的 OSError/ValueError
#     幂等性: notebook cell 重跑时 _orig_feed 会指向 _quiet_feed 自己导致无限递归，
#     所以把原函数挂到类属性上，只在首次时保存。
import multiprocessing.queues as _mpq

if not hasattr(_mpq.Queue, '_knee_orig_feed'):
    _mpq.Queue._knee_orig_feed = _mpq.Queue._feed

    @staticmethod
    def _quiet_feed(*args, **kwargs):
        try:
            _mpq.Queue._knee_orig_feed(*args, **kwargs)
        except (OSError, ValueError, BrokenPipeError):
            pass

    _mpq.Queue._feed = _quiet_feed

# (B) 静默所有 "Exception ignored in" 报错（__del__ 清理阶段的无害异常）
import sys as _sys
_sys.unraisablehook = lambda _: None

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Sampler
import json
from tqdm.auto import tqdm

# cv2 在 DataLoader worker 里默认会开多线程，和 num_workers=12 打架，
# 表现为 CPU 争抢、吞吐反而下降。
cv2.setNumThreads(0)

# 输入形状固定，让 cuDNN 实测挑最快的卷积算法
torch.backends.cudnn.benchmark = True

log = logging.getLogger("knee_train")


# ------------------------------------------------------------
# tqdm 静默处理
# ------------------------------------------------------------

if not hasattr(tqdm, "_knee_orig_close"):
    tqdm._knee_orig_close = tqdm.close


def _quiet_close(self) -> None:
    try:
        tqdm._knee_orig_close(self)
    except AttributeError:
        pass


tqdm.close = _quiet_close


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)


def pbar(iterable=None, **kwargs):
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(iterable, **kwargs)


# ============================================================
# 1. SCHEMA
# ============================================================

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]


# ============================================================
# 2. CONFIG
# ============================================================

SELECTORS = {
    "gold": lambda d, g: g if np.isfinite(g) else d,
    "derived": lambda d, g: d,
    "worse_of_two": lambda d, g: min(d, g) if np.isfinite(g) else d,
}


@dataclass(frozen=True)
class Cfg:
    # ---- 数据路径 ----
    cache_dirs: tuple[Path, ...] = (
        Path("/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part1/knee_v5_384px_24sl_central80_crop130_physicalsort_part0"),
        Path("/kaggle/input/notebooks/zhuyifanss/dataprepare-of-rsnav5part2/knee_v5_384px_24sl_central80_crop130_physicalsort_part1"),
        Path("/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part3/knee_v5_384px_24sl_central80_crop130_physicalsort_part2"),
        Path("/kaggle/input/notebooks/minminandy/dataprepare-of-rsnav5part4/knee_v5_384px_24sl_central80_crop130_physicalsort_part3"),
    )
    # VLM image + report labels; official Gold overrides corresponding rows.
    scored_path: Path = Path(
        "/kaggle/input/datasets/minminandy/vlm-score/labels_vlm_imgrep.csv"
    )
    # ★ 旧 LLM 报告标签：V1 (Claude) 与 V2 (Gemma)。
    v1_path: Path = Path(
        "/kaggle/input/datasets/minminandy/llm-scored/train_scored.csv"
    )
    v2_path: Path = Path(
        "/kaggle/input/datasets/minminandy/llm-scorev2/labels_gold_override.csv"
    )
    # ★ 标签混合: label = vlm_weight * VLM + (1 - vlm_weight) * (v1_weight * V1 + (1 - v1_weight) * V2)
    vlm_weight: float = 0.7
    v1_weight: float = 0.7
    # DINOv3 ConvNeXt-Tiny 本地权重
    timm_weight_path: Path = Path(
        "/kaggle/input/notebooks/lihu6666/dinvo3convnext-timm/dinov3_convnext_tiny_pretrained/model.safetensors"
    )
    allow_pretrained_download: bool = False
    work_dir: Path = Path(
        "/kaggle/working/rsna_dinov3_convnext_tiny_384_oai8649_mix58_42_seven_16ep_splitlr"
    )

    # ---- image (必须与 cache 一致) ----
    img_size: int = 384
    crop_mm: float = 130.0
    n_slots: int = 5
    slices_per_slot: int = 24
    group_size: int = 3
    n_groups: int = 8
    input_channels: int = 3

    # ---- backbone ----
    model_name: str = "convnext_tiny.dinov3_lvd1689m"  # DINOv3 ConvNeXt-Tiny: feature_dim=768
    # 2.5D 11-token input; enable gradient checkpointing only if memory requires it.
    convnext_grad_checkpointing: bool = False
    # Forward the shared ConvNeXt in image-batch chunks if needed. 0 disables chunking.
    backbone_chunk_size: int = 72

    # ---- 切片聚合 ----
    gru_hidden: int = 256      # 双向 -> 输出 2*gru_hidden
    gru_layers: int = 1
    attn_hidden: int = 128     # MLP Attention 的中间维

    # 每个序列专家都直接预测 12 类。主输出是按标签加权后的融合 logits；
    # 这个小辅助损失防止训练早期 gate 只选择一个专家，其余专家收不到有效梯度。
    expert_aux_weight: float = 0.20

    # ---- 几何增强 (train-only, study 级共享同一矩阵) ----
    aug_rotate_p: float = 0.5
    aug_rotate_limit: float = 25.0            # 度，对称 [-limit, +limit]
    aug_perspective_p: float = 0.5
    aug_perspective_scale: tuple = (0.05, 0.1)

    # 序列顺序翻转 (RSNA 2024 2nd place)
    aug_seq_flip_p: float = 0.5

    # 2.5D triplet depth jitter: each 3-slice group can shift by -1/0/+1 inside the 24-slice cache.
    aug_depth_jitter: bool = True

    # ---- ★ 本次新增: 强度增强 (train-only, 逐 slot = 逐序列) ----
    # 总开关。False 时 __getitem__ 完全不进这条分支，结果与旧脚本逐比特相同。
    aug_intensity: bool = True

    # (1) 亮度 / 对比度 —— 六个 RSNA 冠军仓库里 4 个用它，其中腰椎 2nd
    #     (RandomBrightnessContrast ±0.4/±0.25) 和颈椎 5th (±0.25/±0.25)
    #     都是 p=1.0 常开。这里取 0.5，因为你的预处理端已经做过 1%-99%
    #     分位归一化，全局增益差异已被压掉一部分；想复刻他们就把 p 调到 1.0。
    aug_bc_p: float = 0.5
    aug_brightness_limit: float = 0.25        # 加性平移，单位是 255 的比例
    aug_contrast_limit: float = 0.25          # 绕均值的乘性缩放

    # (2) gamma —— 动脉瘤 1st 用了两个 GammaTransform(0.6-2.0) 各 p=0.2。
    #     分位归一化是线性拉伸，改不了对比度曲线的形状，所以这一项不可替代。
    aug_gamma_p: float = 0.3
    aug_gamma_range: tuple = (0.7, 1.4)

    # (3) 亮度梯度场 (= 动脉瘤 1st 的 BrightnessGradientAdditiveTransform)。
    #     MRI 特有的线圈不均匀 / 偏置场：空间上缓变的乘性场，
    #     **全局标量除不掉**，是分位归一化唯一碰不到的那一项。
    aug_bias_p: float = 0.2
    aug_bias_strength: float = 0.25           # 低分辨率控制点的高斯 sigma
    aug_bias_grid: int = 4                    # 控制点网格边长，越小越平缓

    # (4) 高斯噪声 —— 1.5T 61% / 3.0T 39% 的 SNR 差异。
    #     腰椎 2nd 用 GaussNoise(var_limit=(3,9)) -> sigma 1.7~3.0 (0-255 尺度)，
    #     比 albumentations 默认 (10,50) -> 3.2~7.1 还弱。所有方案都下手很轻。
    aug_noise_p: float = 0.3
    aug_noise_sigma: tuple = (2.0, 3.0)

    # (5) 模糊 —— 腰椎 2nd 的 OneOf[MotionBlur/MedianBlur/GaussianBlur] 里的一支。
    aug_blur_p: float = 0.1
    aug_blur_sigma: tuple = (0.3, 1.0)

    # (6) 低分辨率模拟 (= 动脉瘤 1st 的 SimulateLowResolutionTransform)。
    #     你的原生采集矩阵是 960/896/800/640 四档 (px 0.073~1.172mm)，
    #     全部 resize 到 224 之后有效分辨率差好几倍 ——
    #     这是数据里唯一一条完全没被处理的异质轴。
    aug_lowres_p: float = 0.15
    aug_lowres_scale: tuple = (0.5, 1.0)

    # (7) 单像素 dropout —— 腰椎 2nd 的 CoarseDropout(50-250 个 1x1 洞)，
    #     本质是椒盐噪声。遮挡类不算强度增强，默认关。
    aug_dropout_p: float = 0.0
    aug_dropout_holes: tuple = (50, 250)

    # ---- ★ 本次新增: 权重 EMA ----
    use_ema: bool = True
    # decay 的时间尺度是 optimizer step (= 有效 batch)，不是 micro-batch。
    # 本配置下每 epoch 约 200 步，0.995 -> 平均窗口 ~200 步 ≈ 1 epoch。
    # 若把 epochs 调大或 accum 调小，相应把 decay 往 0.999 调。
    ema_decay: float = 0.995
    # 前几十步 head 还是随机初始化，用 min(decay, (1+n)/(warmup+n)) 让 EMA 快速跟上
    ema_warmup_steps: int = 10
    # True: 每个 epoch 额外验证一次 raw 权重，用于对照 (验证耗时翻倍)
    ema_eval_raw: bool = False

    # ---- optim ----
    batch_studies: int = 4
    accum_steps: int = 4
    eval_batch_studies: int = 16
    # DINOv3 pretrained backbone uses the lower LR from the earlier successful run.
    backbone_lr: float = 2e-5
    lr: float = 2e-4
    weight_decay: float = 1e-2
    epochs: int = 16
    use_amp: bool = True

    # ---- labels / selection ----
    gold_weight: float = 3.0
    n_gold_expected: int = 58
    selection: str = "worse_of_two"
    min_gold_val: int = 6

    # ---- split ----
    n_folds: int = 4
    fold_to_run: int = 0

    # ---- runtime ----
    num_workers: int = 12
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.selection not in SELECTORS:
            raise ValueError(
                f"selection 必须是 {sorted(SELECTORS)} 之一，得到 {self.selection!r}"
            )
        if self.slices_per_slot < 1:
            raise ValueError(f"slices_per_slot 必须 >= 1，得到 {self.slices_per_slot}")
        if self.group_size != 3:
            raise ValueError(f"depth jitter 当前要求 group_size=3，得到 {self.group_size}")
        if self.slices_per_slot != self.n_groups * self.group_size:
            raise ValueError(
                "slices_per_slot 必须 = n_groups * group_size，得到 "
                f"{self.slices_per_slot} != {self.n_groups} * {self.group_size}"
            )
        if self.input_channels != 3:
            raise ValueError(
                "2.5D 输入要求 input_channels=3，"
                f"得到 input_channels={self.input_channels}"
            )
        if self.lr <= 0:
            raise ValueError("lr 必须 > 0")
        if self.n_slots != len(SLOTS):
            raise ValueError(f"n_slots={self.n_slots} 与 SLOTS 的 {len(SLOTS)} 项不符")
        if self.accum_steps < 1:
            raise ValueError(f"accum_steps 必须 >= 1，得到 {self.accum_steps}")
        if self.expert_aux_weight < 0:
            raise ValueError(
                f"expert_aux_weight 必须 >= 0，得到 {self.expert_aux_weight}"
            )
        # ★ 标签混合权重校验
        for name in ("vlm_weight", "v1_weight"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} 必须在 [0,1]，得到 {v}")
        for name in ("aug_rotate_p", "aug_perspective_p", "aug_seq_flip_p"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} 必须在 [0,1]，得到 {v}")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay 必须在 (0,1)，得到 {self.ema_decay}")
        if self.ema_warmup_steps < 0:
            raise ValueError(f"ema_warmup_steps 必须 >= 0，得到 {self.ema_warmup_steps}")
        # ---- ★ 强度增强的参数校验 ----
        for name in ("aug_bc_p", "aug_gamma_p", "aug_bias_p", "aug_noise_p",
                     "aug_blur_p", "aug_lowres_p", "aug_dropout_p"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} 必须在 [0,1]，得到 {v}")
        for name in ("aug_brightness_limit", "aug_contrast_limit",
                     "aug_bias_strength"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 必须 >= 0，得到 {getattr(self, name)}")
        if not 0 < self.aug_gamma_range[0] <= self.aug_gamma_range[1]:
            raise ValueError(f"aug_gamma_range 非法: {self.aug_gamma_range}")
        if not 0 < self.aug_lowres_scale[0] <= self.aug_lowres_scale[1] <= 1.0:
            raise ValueError(f"aug_lowres_scale 必须在 (0,1]: {self.aug_lowres_scale}")
        if self.aug_bias_grid < 2:
            raise ValueError(f"aug_bias_grid 必须 >= 2，得到 {self.aug_bias_grid}")
        if self.aug_noise_sigma[0] < 0 or self.aug_noise_sigma[1] < self.aug_noise_sigma[0]:
            raise ValueError(f"aug_noise_sigma 非法: {self.aug_noise_sigma}")
        if not (0 <= self.aug_dropout_holes[0] <= self.aug_dropout_holes[1]):
            raise ValueError(f"aug_dropout_holes 非法: {self.aug_dropout_holes}")

    @property
    def effective_batch(self) -> int:
        return self.batch_studies * self.accum_steps

    @property
    def label_desc(self) -> str:
        w, a = self.vlm_weight, self.v1_weight
        return (f"{w:g}*VLM + {1 - w:g}*({a:g}*V1 + {1 - a:g}*V2)"
                f" = {w:g}*VLM + {(1 - w) * a:.4g}*V1 + {(1 - w) * (1 - a):.4g}*V2")

    @property
    def aug_desc(self) -> str:
        parts = []
        if self.aug_depth_jitter:
            parts.append("depth_jitter±1(0.25/0.50/0.25)")
        if self.aug_rotate_p > 0:
            parts.append(f"rotate±{self.aug_rotate_limit:.0f}°(p={self.aug_rotate_p})")
        if self.aug_perspective_p > 0:
            parts.append(
                f"perspective{self.aug_perspective_scale}(p={self.aug_perspective_p})"
            )
        if self.aug_intensity and self.intensity_desc != "off":
            parts.append(f"intensity[{self.intensity_desc}]")
        if self.aug_seq_flip_p > 0:
            parts.append(f"seq_flip(p={self.aug_seq_flip_p})")
        return " + ".join(parts) if parts else "none"

    @property
    def intensity_desc(self) -> str:
        """只列出真正开着的那几项，便于日志逐行核对。"""
        if not self.aug_intensity:
            return "off"
        p = []
        if self.aug_lowres_p > 0:
            p.append(f"lowres{self.aug_lowres_scale}(p={self.aug_lowres_p})")
        if self.aug_blur_p > 0:
            p.append(f"blur{self.aug_blur_sigma}(p={self.aug_blur_p})")
        if self.aug_bias_p > 0:
            p.append(f"bias_field(sig={self.aug_bias_strength},"
                     f"grid={self.aug_bias_grid},p={self.aug_bias_p})")
        if self.aug_noise_p > 0:
            p.append(f"noise{self.aug_noise_sigma}(p={self.aug_noise_p})")
        if self.aug_gamma_p > 0:
            p.append(f"gamma{self.aug_gamma_range}(p={self.aug_gamma_p})")
        if self.aug_bc_p > 0:
            p.append(f"bc(b=±{self.aug_brightness_limit},"
                     f"c=±{self.aug_contrast_limit},p={self.aug_bc_p})")
        if self.aug_dropout_p > 0:
            p.append(f"px_dropout{self.aug_dropout_holes}(p={self.aug_dropout_p})")
        return " ".join(p) if p else "off"

    @property
    def ema_desc(self) -> str:
        if not self.use_ema:
            return "off"
        return f"decay={self.ema_decay} warmup={self.ema_warmup_steps}步"

    # ---- 派生路径 ----
    @property
    def train_csv(self) -> Path:
        return Path("/kaggle/input/datasets/minminandy/rsna-file/train.csv")

    @property
    def out_dir(self) -> Path:
        return self.work_dir / f"fold{self.fold_to_run}"


# ============================================================
# 3. UTILS
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def report_hash(text) -> str:
    if not isinstance(text, str):
        text = ""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shutdown_loader(dl) -> None:
    """显式回收 DataLoader 的 worker，不依赖 GC 时机。"""
    if dl is None:
        return
    it = getattr(dl, "_iterator", None)
    if it is not None:
        try:
            it._shutdown_workers()
        except Exception:
            pass
        dl._iterator = None


# ============================================================
# 4. DATA CONTAINERS
# ============================================================

@dataclass
class FoldData:
    cache: np.ndarray       # mmap [N_study, 5, 24, H, W] uint8
    mask: np.ndarray        # [N_study, 5] float32
    cache_rows: np.ndarray  # [n] int64
    y: np.ndarray           # [n, 12] float32
    weight: np.ndarray      # [n, 12] float32
    is_gold: np.ndarray     # [n] bool
    groups: np.ndarray      # [n] str
    study_ids: np.ndarray   # [n] str


@dataclass
class ValMetrics:
    derived_auc: float
    gold_auc: float
    selection: float
    n_gold_val: int
    derived_per_label: dict = field(default_factory=dict)
    gold_per_label: dict = field(default_factory=dict)


# ============================================================
# 5. CACHE 加载
# ============================================================

class ShardedCache:
    """Read one global study row without concatenating image volumes.

    Only file paths and small row mappings are serialized to spawned workers.
    """
    def __init__(self, paths, row_map, shape):
        self.paths = tuple(str(p) for p in paths)
        self.row_map = np.asarray(row_map, dtype=np.int64)
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.uint8)
        self._arrays = None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        if not isinstance(row, (int, np.integer)):
            raise TypeError("ShardedCache supports one integer study row at a time")
        if self._arrays is None:
            self._arrays = [np.load(p, mmap_mode="r") for p in self.paths]
        part, local_row = self.row_map[row]
        return self._arrays[int(part)][int(local_row)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = None
        return state


def load_cache(cfg):
    import hashlib
    import json
    want = (cfg.n_slots, cfg.slices_per_slot, cfg.img_size, cfg.img_size)
    paths, masks, ids_parts, mappings, manifests = [], [], [], [], []
    for part_index, directory in enumerate(cfg.cache_dirs):
        directory = Path(directory)
        for name in ('train_cache.npy', 'train_mask.npy', 'study_ids.npy', 'manifest.json'):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        manifest = json.loads((directory / 'manifest.json').read_text())
        arr = np.load(directory / 'train_cache.npy', mmap_mode='r')
        ids = np.load(directory / 'study_ids.npy', allow_pickle=True).astype(str)
        mask = np.load(directory / 'train_mask.npy')
        if ids.ndim != 1 or arr.shape != (len(ids), *want) or arr.dtype != np.uint8:
            raise ValueError(f'Invalid cache shape/dtype: {directory}: {arr.shape}')
        if mask.shape != (len(ids), cfg.n_slots) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Invalid mask: {directory}')
        digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        if (manifest['part_ids_sha256'] != digest or manifest['shape'] != list(arr.shape)
                or manifest['part_studies'] != len(ids) or manifest['dtype'] != 'uint8'
                or manifest['central_fraction'] != 0.8 or manifest['crop_mm'] != cfg.crop_mm
                or manifest['slots'] != [x[0] for x in SLOTS]):
            raise ValueError(f'Manifest mismatch: {directory}')
        paths.append(directory / 'train_cache.npy')
        ids_parts.append(ids); masks.append(mask); manifests.append(manifest)
        mappings.extend((part_index, r) for r in range(len(ids)))
        log.info('cache part%d: %d studies, %s', manifest['part_id'], len(ids), arr.shape)
    ids = np.concatenate(ids_parts)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate study IDs across cache parts')
    first = manifests[0]
    keys = ('format_version', 'n_parts', 'total_studies', 'all_ids_sha256',
            'central_fraction', 'crop_mm', 'lat_min_offset_mm', 'slots')
    if any(any(m[k] != first[k] for k in keys) for m in manifests):
        raise ValueError('Cache parts were built with different configurations or study lists')
    if (len(manifests) != first['n_parts']
            or sorted(m['part_id'] for m in manifests) != list(range(first['n_parts']))):
        raise ValueError('Missing or repeated cache part')
    order = np.argsort(ids, kind='stable')
    ids = ids[order]
    if len(ids) != first['total_studies'] or hashlib.sha256('\n'.join(ids).encode()).hexdigest() != first['all_ids_sha256']:
        raise ValueError('Combined study list does not match manifest')
    mask = np.concatenate(masks, axis=0)[order]
    cache = ShardedCache(paths, np.asarray(mappings)[order], (len(ids), *want))
    log.info('Loaded %d studies, shape %s; sharded mmap, no image concatenation', len(ids), cache.shape)
    return cache, mask, ids


# ============================================================
# 6. 标签加载
# ============================================================

def read_scored_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    missing = [c for c in ["StudyInstanceUID", *TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少 columns: {missing}")
    uid = df["StudyInstanceUID"]
    if uid.isna().any() or uid.str.strip().eq("").any() or uid.duplicated().any():
        raise ValueError(f"{path}: StudyInstanceUID 缺失或重复")
    scores = df.set_index("StudyInstanceUID")[TARGETS].apply(pd.to_numeric, errors="raise")
    values = scores.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{path}: 标签必须是 [0,1] 内的有限数值")
    return scores


def load_vlm_labels(cfg: Cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """★ 标签 = vlm_weight*VLM + (1-vlm_weight)*(v1_weight*V1 + (1-v1_weight)*V2)；
    官方 Gold 最后覆盖，训练和 OOF 共用此入口。"""
    vlm = read_scored_labels(cfg.scored_path)
    v1 = read_scored_labels(cfg.v1_path)
    v2 = read_scored_labels(cfg.v2_path)
    for name, df in (("V1", v1), ("V2", v2)):
        if (not df.index.difference(vlm.index).empty
                or not vlm.index.difference(df.index).empty):
            raise ValueError(f"{name} 与 VLM 标签的 UID 集合不一致")
    v1 = v1.reindex(vlm.index)[TARGETS]
    v2 = v2.reindex(vlm.index)[TARGETS]
    llm_mix = cfg.v1_weight * v1 + (1.0 - cfg.v1_weight) * v2
    blended = (cfg.vlm_weight * vlm[TARGETS] + (1.0 - cfg.vlm_weight) * llm_mix).clip(0.0, 1.0)

    train_df = pd.read_csv(cfg.train_csv, dtype={"StudyInstanceUID": str})
    uid = train_df["StudyInstanceUID"]
    if uid.isna().any() or uid.str.strip().eq("").any() or uid.duplicated().any():
        raise ValueError("官方 train.csv 的 StudyInstanceUID 缺失或重复")
    official = train_df.set_index("StudyInstanceUID")[TARGETS].apply(pd.to_numeric, errors="raise")
    gold = official.loc[official.notna().all(axis=1)]
    if len(gold) != cfg.n_gold_expected:
        raise ValueError(f"预期 {cfg.n_gold_expected} 个 Gold，实际 {len(gold)}")
    if not np.isin(gold.to_numpy(), [0.0, 1.0]).all():
        raise ValueError("官方 Gold 必须全部为 0/1")
    if not gold.index.difference(blended.index).empty:
        raise ValueError("混合标签缺少官方 Gold UID")
    blended.loc[gold.index, TARGETS] = gold[TARGETS]

    ng = blended.index.difference(gold.index)
    log.info("labels: %s | studies: %d | official Gold override: %d",
             cfg.label_desc, len(blended), len(gold))
    log.info("  VLM %s | V1 %s | V2 %s", cfg.scored_path, cfg.v1_path, cfg.v2_path)
    log.info("  non-gold 阳性率(>=0.5)  VLM %.3f | V1 %.3f | V2 %.3f | mixed %.3f",
             float((vlm.loc[ng, TARGETS] >= 0.5).to_numpy().mean()),
             float((v1.loc[ng] >= 0.5).to_numpy().mean()),
             float((v2.loc[ng] >= 0.5).to_numpy().mean()),
             float((blended.loc[ng, TARGETS] >= 0.5).to_numpy().mean()))
    return blended, gold, train_df


def load_labels(cfg: Cfg, study_ids: np.ndarray) -> FoldData:
    llm, gold, train_df = load_vlm_labels(cfg)
    # 保存本次实际使用的标签，便于复现；不四舍五入成两位小数。
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    exported = llm.copy()
    exported["label_source"] = np.where(exported.index.isin(gold.index), "gold", "vlm07_llm03")
    exported.reset_index().to_csv(cfg.work_dir / "labels_used_vlm07_llm03.csv", index=False)

    # ---- 对齐到 cache 行序 ----
    idx = pd.Index(study_ids.astype(str))
    y_df = gold.reindex(idx)[TARGETS].combine_first(llm.reindex(idx)[TARGETS])

    y_all = y_df.to_numpy(dtype=np.float32)
    is_gold_all = np.asarray(idx.isin(gold.index))

    keep = np.isfinite(y_all).all(axis=1)
    if not keep.any():
        raise RuntimeError("没有可用于训练的 study。")

    cache_rows = np.flatnonzero(keep).astype(np.int64)
    y = y_all[keep]
    is_gold = is_gold_all[keep]
    per_study_w = np.where(is_gold, cfg.gold_weight, 1.0).astype(np.float32)
    weight = np.repeat(per_study_w[:, None], len(TARGETS), axis=1)
    ids = idx[keep].to_numpy()

    # ---- GroupKFold 分组键 ----
    if "Report" in train_df.columns:
        report_map = (
            train_df.drop_duplicates("StudyInstanceUID")
            .set_index("StudyInstanceUID")["Report"].fillna("").to_dict()
        )
        groups = np.array([report_hash(report_map.get(sid, sid)) for sid in ids])
    else:
        groups = ids.astype(str)

    n_gold = int(is_gold.sum())
    log.info(
        "supervised: %d | gold: %d (w=%.1f) | mixed VLM/LLM: %d (w=1.0)",
        len(ids), n_gold, cfg.gold_weight, len(ids) - n_gold,
    )

    return FoldData(
        cache=None,  # 外部赋值
        mask=None,
        cache_rows=cache_rows,
        y=y,
        weight=weight,
        is_gold=is_gold,
        groups=groups,
        study_ids=ids,
    )


# ============================================================
# 7. 几何增强
# ============================================================

_DST_CORNERS_CACHE: dict[int, np.ndarray] = {}


def _dst_corners(size: int) -> np.ndarray:
    c = _DST_CORNERS_CACHE.get(size)
    if c is None:
        c = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        _DST_CORNERS_CACHE[size] = c
    return c


def sample_geo_matrix(cfg: Cfg, size: int) -> np.ndarray | None:
    """为一个 study 采样一个 3x3 变换矩阵；两个增强都没触发时返回 None。

    M = P @ R  —— 先旋转再透视，合成成单个矩阵只做一次重采样。
    两次 warp 会二次插值模糊，对 ACL 那种 5-6mm (224px/160mm 下只有 7-8
    像素宽) 的结构影响不小。

    随机数一律用 python 的 `random`：torch DataLoader 每个 worker 只 seed
    python random / torch，不 seed numpy。用 np.random 会让 12 个 worker
    生成完全相同的增强序列。
    """
    M = None

    # ---- Rotate: 绕图像中心 ----
    if cfg.aug_rotate_p > 0 and random.random() < cfg.aug_rotate_p:
        angle = random.uniform(-cfg.aug_rotate_limit, cfg.aug_rotate_limit)
        center = ((size - 1) * 0.5, (size - 1) * 0.5)
        R = cv2.getRotationMatrix2D(center, angle, 1.0)          # [2,3]
        M = np.vstack([R, [0.0, 0.0, 1.0]])                      # [3,3]

    # ---- Perspective: 四角独立抖动 (复刻 albumentations.Perspective) ----
    if cfg.aug_perspective_p > 0 and random.random() < cfg.aug_perspective_p:
        s = random.uniform(*cfg.aug_perspective_scale)
        # |N(0, s)| mod 0.32 —— 与 albumentations 一致，上限 0.32 防止退化
        jit = [[abs(random.gauss(0.0, s)) % 0.32 for _ in range(2)] for _ in range(4)]
        pts = np.asarray(jit, dtype=np.float32)
        pts[1, 0] = 1.0 - pts[1, 0]        # 右上
        pts[2] = 1.0 - pts[2]              # 右下
        pts[3, 1] = 1.0 - pts[3, 1]        # 左下
        src = pts * (size - 1)
        P = cv2.getPerspectiveTransform(src, _dst_corners(size))  # [3,3]
        M = P if M is None else (P @ M)

    return M


def apply_geo(imgs: np.ndarray, M: np.ndarray) -> np.ndarray:
    """把同一个 3x3 矩阵应用到 [S, N, H, W] 的全部 S*N 张切片。

    整个 study 共用 M 是关键：逐层随机变换会破坏层间连续性，
    BiGRU 沿 group 维扫描时相邻层变成两张对不齐的图，学不到东西。

    背景填充 0 —— MRI 背景本来就是黑的。
    插值用 INTER_LINEAR；uint8 输入输出，无需额外 clip。
    """
    S, N, H, W = imgs.shape
    out = np.empty_like(imgs)
    for s in range(S):
        for n in range(N):
            out[s, n] = cv2.warpPerspective(
                imgs[s, n], M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    return out


# ============================================================
# 8. DATASET
# ============================================================

class StudyDataset(Dataset):
    def __init__(self, cfg: Cfg, cache, mask, cache_rows, y, weight, *, train: bool):
        self.cfg = cfg
        self.cache = cache
        self.mask = mask
        self.cache_rows = np.asarray(cache_rows, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.float32)
        self.weight = np.asarray(weight, dtype=np.float32)
        self.train = bool(train)

    @staticmethod
    def _depth_jitter_pm1(imgs: np.ndarray) -> np.ndarray:
        """Train-only ±1 jitter for consecutive triplets; supports 24 slices / 8 groups."""
        S, N, H, W = imgs.shape
        if N < 3 or N % 3:
            raise RuntimeError(f"depth jitter requires a positive multiple of 3, got {N}")

        out = np.empty_like(imgs)
        for s in range(S):
            for g in range(N // 3):
                base_start = g * 3
                u = random.random()
                if u < 0.25:
                    shift = -1
                elif u < 0.75:
                    shift = 0
                else:
                    shift = 1
                start = max(0, min(N - 3, base_start + shift))
                dst0 = g * 3
                out[s, dst0:dst0 + 3] = imgs[s, start:start + 3]
        return out

    @staticmethod
    def _seq_flip(imgs: np.ndarray, p: float) -> np.ndarray:
        """逐 slot 独立地反转 24 层的顺序 (RSNA 2024 2nd place)。

        作用于层轴而非图像平面，所以不交换内外侧 —— 与 HFlip 的破坏性
        完全不同。默认 p=0.0 (关闭)。
        """
        if p <= 0:
            return imgs
        out = imgs
        for s in range(imgs.shape[0]):
            if random.random() < p:
                if out is imgs:
                    out = imgs.copy()
                out[s] = out[s][::-1]
        return out

    @staticmethod
    def _intensity_aug(imgs: np.ndarray, mask: np.ndarray, cfg: Cfg) -> np.ndarray:
        """★ 强度增强 —— 逐 slot (= 逐序列)，slot 内 24 层共用同一组参数。

        为什么需要
        ----------
        体检出的采集异质性: Siemens 42% / Philips 31% / GE 20% / TOSHIBA 4%,
        1.5T 61% / 3.0T 39%, 16 个站点, 原生采集矩阵 960/896/800/640 四档
        (px 0.073~1.172 mm)。
        预处理端 `_to_uint8` 的 1%-99% 分位归一化只压掉了**全局线性增益**，
        压不掉以下四样:
            gamma / 对比度曲线   非线性，线性拉伸改不了曲线形状
            偏置场               空间缓变的乘性场，全局标量除不掉  <- MRI 特有
            噪声水平             分位数只看两个点，看不到噪声
            有效分辨率           原生矩阵差 1.5 倍，resize 到 224 后差异仍在

        为什么逐 slot 而不是逐 study
        ---------------------------
        几何增强必须整个 study 共享同一矩阵 —— 层间/序列间要对齐，否则 BiGRU
        沿组维扫到的是两张对不齐的图。强度不需要: 5 个 slot 是 5 条**不同的
        序列**(T1/PD/T2、加不加脂肪抑制)，采集参数本来就不同，独立扰动才对应
        真实的跨厂商 / 跨场强异质性。

        为什么 slot 内 24 层共用同一组参数
        ---------------------------------
        一次采集的偏置场和噪声水平在层间是连续的。逐层随机会让 BiGRU 扫到
        一串亮度乱跳的图，把层间连续性这个信号毁掉。

        为什么跳过空 slot
        -----------------
        mask=0 的 slot 在 cache 里是全零占位 (平均填充率 92.94%，即 7.06% 的
        slot 是空的)。给它加噪声会让常数图变成非零图，污染 BatchNorm 的 batch
        统计和 running_mean/var —— EfficientNet-B3 同样依赖 BN 统计量。

        施加顺序 (按物理链条)
        --------------------
            低分辨率 -> 模糊 -> 偏置场(乘性) -> 噪声(加性) -> gamma -> 亮度对比度
        噪声必须在 gamma 之前 (真实噪声产生于采集，不该被显示传递函数二次拉伸)；
        dropout 放最后，保证洞真的是 0。

        随机数
        ------
        一律走 python `random` —— DataLoader worker 只 seed python random / torch，
        **不 seed numpy 全局状态**，用 np.random 会让 12 个 worker 生成完全相同
        的增强序列。批量高斯用 default_rng(random 出来的种子)。
        """
        R = imgs.shape[-1]
        out = None

        for s in range(imgs.shape[0]):
            if mask[s] < 0.5:                     # 空 slot: 全零占位，别动
                continue

            x = None                              # 惰性转 float32，没触发就不复制

            # ---- (6) 低分辨率模拟: 降采样(最近邻) -> 升回去(双三次) ----
            if cfg.aug_lowres_p > 0 and random.random() < cfg.aug_lowres_p:
                sc = random.uniform(*cfg.aug_lowres_scale)
                r = int(round(R * sc))
                if 8 <= r < R:
                    x = np.stack([
                        cv2.resize(
                            cv2.resize(imgs[s, k], (r, r),
                                       interpolation=cv2.INTER_NEAREST),
                            (R, R), interpolation=cv2.INTER_CUBIC)
                        for k in range(imgs.shape[1])
                    ]).astype(np.float32)

            # ---- (5) 高斯模糊 ----
            if cfg.aug_blur_p > 0 and random.random() < cfg.aug_blur_p:
                if x is None:
                    x = imgs[s].astype(np.float32)
                sig = random.uniform(*cfg.aug_blur_sigma)
                x = np.stack([
                    cv2.GaussianBlur(x[k], (0, 0), sigmaX=sig, sigmaY=sig)
                    for k in range(x.shape[0])
                ])

            # ---- (3) 偏置场: 空间缓变的乘性场 ----
            if cfg.aug_bias_p > 0 and random.random() < cfg.aug_bias_p:
                if x is None:
                    x = imgs[s].astype(np.float32)
                k = cfg.aug_bias_grid
                low = np.asarray(
                    [[random.gauss(0.0, cfg.aug_bias_strength) for _ in range(k)]
                     for _ in range(k)], dtype=np.float32)
                field = cv2.resize(low, (R, R), interpolation=cv2.INTER_CUBIC)
                x = x * (1.0 + field)[None]       # 广播到 24 层，层间同一个场

            # ---- (4) 高斯噪声 ----
            if cfg.aug_noise_p > 0 and random.random() < cfg.aug_noise_p:
                if x is None:
                    x = imgs[s].astype(np.float32)
                sig = random.uniform(*cfg.aug_noise_sigma)
                rng = np.random.default_rng(random.getrandbits(63))
                x = x + rng.normal(0.0, sig, x.shape).astype(np.float32)

            # ---- (2) gamma ----
            if cfg.aug_gamma_p > 0 and random.random() < cfg.aug_gamma_p:
                if x is None:
                    x = imgs[s].astype(np.float32)
                g = random.uniform(*cfg.aug_gamma_range)
                x = np.power(np.clip(x, 0.0, 255.0) / 255.0, g) * 255.0

            # ---- (1) 亮度 / 对比度 ----
            #   对比度绕**均值**缩放 (不改整体亮度)，亮度是加性平移。
            #   与 albumentations.RandomBrightnessContrast 同形，但不是逐比特一致。
            if cfg.aug_bc_p > 0 and random.random() < cfg.aug_bc_p:
                if x is None:
                    x = imgs[s].astype(np.float32)
                c = random.uniform(-cfg.aug_contrast_limit, cfg.aug_contrast_limit)
                b = random.uniform(-cfg.aug_brightness_limit, cfg.aug_brightness_limit)
                mu = float(x.mean())
                x = (x - mu) * (1.0 + c) + mu + b * 255.0

            if x is not None:
                if out is None:
                    out = imgs.copy()
                out[s] = np.clip(x, 0.0, 255.0).astype(np.uint8)

            # ---- (7) 单像素 dropout: 放最后，保证洞是纯 0 ----
            if cfg.aug_dropout_p > 0 and random.random() < cfg.aug_dropout_p:
                if out is None:
                    out = imgs.copy()
                n_hole = random.randint(*cfg.aug_dropout_holes)
                ys = [random.randrange(R) for _ in range(n_hole)]
                xs = [random.randrange(R) for _ in range(n_hole)]
                out[s][:, ys, xs] = 0             # 24 层同一组洞

        return imgs if out is None else out

    def __len__(self) -> int:
        return len(self.cache_rows)

    def __getitem__(self, i):
        cfg = self.cfg
        r = int(self.cache_rows[i])

        # 和原 baseline 一样先从 mmap copy 出当前 study
        imgs_np = np.array(self.cache[r])                      # [5,24,224,224] uint8

        if self.train:
            # 顺序: 先选层 (depth jitter / seq flip)，再变换像素 (geo)。
            # 两者独立，但先选层可以只 warp 真正会用到的 24 层。
            if cfg.aug_depth_jitter:
                imgs_np = self._depth_jitter_pm1(imgs_np)
            imgs_np = self._seq_flip(imgs_np, cfg.aug_seq_flip_p)

            # ★ 强度增强放在几何变换**之前**：
            #   apply_geo 用 BORDER_CONSTANT=0 填充，视野外应当保持纯黑。
            #   放在之后会给补零区加上噪声 / 偏置场 / 亮度平移。
            if cfg.aug_intensity:
                imgs_np = self._intensity_aug(imgs_np, self.mask[r], cfg)

            # study 级共享的几何变换
            M = sample_geo_matrix(cfg, cfg.img_size)
            if M is not None:
                imgs_np = apply_geo(imgs_np, M)

        imgs = torch.from_numpy(np.ascontiguousarray(imgs_np))
        m = torch.from_numpy(np.ascontiguousarray(self.mask[r])).float()
        y = torch.from_numpy(np.ascontiguousarray(self.y[i])).float()
        w = torch.from_numpy(np.ascontiguousarray(self.weight[i])).float()
        return imgs, m, y, w


OAI_IMAGES = Path('/kaggle/input/datasets/minminandy/oai-baseline-2399-384px-images')
OAI_LABELS = Path('/kaggle/input/datasets/minminandy/oai-soft-labels-2399-v3/OAI_soft_labels_2399_v3.csv')
OAI_ACTIVE_TARGETS = ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis']
OAI_SOURCE_SLOTS = ['SAG_IW_TSE', 'COR_IW_TSE', 'AX_MPR', 'COR_MPR', 'SAG_3D_DESS']


class OAIThreePlaneCache(ShardedCache):
    def __getitem__(self, row):
        # Copy only one knee; never concatenate image volumes in RAM.
        image = np.array(super().__getitem__(row), copy=True)
        if image.shape[0] == 3:
            image = np.pad(image, ((0, 2), (0, 0), (0, 0), (0, 0)))
        image[3:] = 0
        return image


def oai_seven_targets(rows):
    y = np.zeros((len(rows), len(TARGETS)), dtype=np.float32)
    w = np.zeros_like(y)
    for target in OAI_ACTIVE_TARGETS:
        j = TARGETS.index(target)
        raw = rows[target].to_numpy(dtype=np.float32)
        mask = rows[target + '__mask'].to_numpy(dtype=np.float32)
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Invalid OAI mask: {target}')
        known = mask == 1
        values = raw[known]
        if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
            raise ValueError(f'Invalid known OAI target: {target}')
        y[known, j] = raw[known]
        w[known, j] = 1.0  # Binary validity only; ignore original __weight columns.
    return y, w


def make_oai_single_dataset(cfg):
    images = OAI_IMAGES
    manifest = json.loads((images / 'manifest.json').read_text())
    if manifest['slot_order'] not in (OAI_SOURCE_SLOTS, OAI_SOURCE_SLOTS[:3]):
        raise ValueError('OAI source slot order mismatch')
    df = pd.read_csv(OAI_LABELS, dtype={'knee_id': str})
    required = ['knee_id', *OAI_ACTIVE_TARGETS,
                *[t + '__mask' for t in OAI_ACTIVE_TARGETS]]
    if not set(required).issubset(df.columns) or df.knee_id.isna().any() or not df.knee_id.is_unique:
        raise ValueError('Missing OAI columns or duplicate knee IDs')
    want = (5, 24, 384, 384)
    if (cfg.n_slots, cfg.slices_per_slot, cfg.img_size, cfg.img_size) != want:
        raise ValueError('This OAI mix requires the 384px/24-slice/5-slot configuration')
    source_slots = len(manifest['slot_order'])
    paths, mappings, ids_parts, masks = [], [], [], []
    for p, n in enumerate(manifest['parts']):
        path = images / f'part{p}_cache.npy'
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        ids = np.load(images / f'part{p}_knee_ids.npy', allow_pickle=False).astype(str)
        sm = np.load(images / f'part{p}_series_mask.npy', allow_pickle=False)
        if a.shape != (n, source_slots, *want[1:]) or a.dtype != np.uint8 or ids.shape != (n,):
            raise ValueError(f'Invalid OAI image part {p}')
        if sm.shape != (n, source_slots) or not np.isin(sm, [0, 1]).all():
            raise ValueError(f'Invalid OAI series mask part {p}')
        paths.append(path); ids_parts.append(ids); masks.append(np.pad(sm, ((0, 0), (0, 5-source_slots))))
        mappings.extend((p, r) for r in range(n))
    ids = np.concatenate(ids_parts)
    if len(df) != len(ids) or len(np.unique(ids)) != len(ids) or set(ids) != set(df.knee_id):
        raise ValueError('Expected unique aligned OAI exam IDs; include visit in follow-up knee_id')
    rows = df.set_index('knee_id').loc[ids].copy()
    # Follow-up identity includes visit; validate uploaded index rather than trusting row order.
    if (images / 'index.csv').is_file():
        index = pd.read_csv(images / 'index.csv', dtype={'knee_id': str, 'src_subject_id': str, 'visit': str})
        if not index.knee_id.is_unique or set(index.knee_id) != set(ids):
            raise ValueError('OAI index/image IDs mismatch')
        aligned_index = index.set_index('knee_id').loc[ids]
        if 'visit' in aligned_index:
            if 'visit' not in rows or not np.array_equal(rows['visit'].astype(str), aligned_index['visit'].astype(str)):
                raise ValueError('OAI label/image visit mismatch')
        if 'src_subject_id' in aligned_index:
            if 'src_subject_id' in rows and not np.array_equal(rows['src_subject_id'].astype(str), aligned_index['src_subject_id'].astype(str)):
                raise ValueError('OAI label/image participant mismatch')
            rows['src_subject_id'] = aligned_index['src_subject_id']
    packing_indices = np.asarray(mappings, dtype=np.int64)
    rows['part'] = packing_indices[:, 0]
    rows['row'] = packing_indices[:, 1]
    y, w = oai_seven_targets(rows)
    sm = np.concatenate(masks).astype(np.float32)
    sm[:, 3:] = 0
    keep = np.flatnonzero((w.sum(1) > 0) & (sm.sum(1) > 0))
    if len(keep) == 0:
        raise ValueError('No OAI knees have usable images and seven-target supervision')
    cache = OAIThreePlaneCache(paths, mappings, (len(ids), *want))
    audit = rows.iloc[keep][['part', 'row']].copy()
    for j, target in enumerate(TARGETS):
        if target in OAI_ACTIVE_TARGETS:
            audit[target] = np.where(w[keep, j] > 0, y[keep, j], np.nan)
            audit[target + '__mask'] = w[keep, j].astype(np.uint8)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(cfg.work_dir / 'oai_mixed_labels_used.csv', index=True)
    report = {'source_knees': len(ids), 'used_knees': len(keep),
              'excluded_knees': len(ids) - len(keep), 'targets': OAI_ACTIVE_TARGETS,
              'label_weight': '1 for known, 0 for missing; no extra OAI weights',
              'sequence_mapping': dict(zip([s[0] for s in SLOTS], OAI_SOURCE_SLOTS[:3] + [None, None])),
              'note': 'First three experts shared by plane; MRI contrasts differ. Last two slots masked.',
              'per_target': {t: {'positive_ge_0_5': int(((y[keep, TARGETS.index(t)] >= 0.5) & (w[keep, TARGETS.index(t)] > 0)).sum()),
                                 'negative_lt_0_5': int(((y[keep, TARGETS.index(t)] < 0.5) & (w[keep, TARGETS.index(t)] > 0)).sum())}
                             for t in OAI_ACTIVE_TARGETS}}
    (cfg.work_dir / 'oai_mixed_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    log.info('OAI MIX: %s', json.dumps(report))
    ds = StudyDataset(cfg, cache, sm, keep, y[keep], w[keep], train=True)
    ds.oai_ids = ids[keep]
    # Preserve participant grouping across sides, visits, and datasets.
    if 'src_subject_id' in rows:
        ds.oai_groups = rows.iloc[keep]['src_subject_id'].astype(str).to_numpy()
    else:
        extracted = pd.Series(ids[keep]).str.extract(r'^(\d{7})(?:_|$)', expand=False)
        if extracted.isna().any():
            raise ValueError('Need src_subject_id or IDs starting with the seven-digit participant ID')
        ds.oai_groups = extracted.to_numpy()
    return ds



OAI_FOLLOWUP_IMAGES = Path('/kaggle/input/datasets/minminandy/oai-followup-6250-384px-images')
OAI_FOLLOWUP_LABELS = Path('/kaggle/input/datasets/minminandy/oai-soft-followup6250-v3/OAI_soft_followup6250_v3.csv')
OAI_DATASETS = []

def resolve_followup_inputs():
    candidates = []
    for manifest_path in Path('/kaggle/input').rglob('manifest.json'):
        try:
            meta = json.loads(manifest_path.read_text())
        except (ValueError, OSError):
            continue
        if (meta.get('checks') == 6250 and meta.get('resolution') == 384
                and meta.get('slot_order') == OAI_SOURCE_SLOTS[:3]):
            candidates.append(manifest_path.parent)
    labels = ([OAI_FOLLOWUP_LABELS] if OAI_FOLLOWUP_LABELS.is_file()
              else list(Path('/kaggle/input').rglob('OAI_soft_followup6250_v3.csv')))
    if len(candidates) != 1 or len(labels) != 1:
        raise FileNotFoundError('Attach ONE verified 6250/384 cache dataset and '
            'ONE final OAI_soft_followup6250_v3.csv. Found images=%s labels=%s'
            % (candidates, labels))
    label_rows = pd.read_csv(labels[0], dtype={'knee_id': str, 'src_subject_id': str})
    if len(label_rows) != 6250 or label_rows['knee_id'].nunique() != 6250:
        raise ValueError('Final follow-up label file must contain 6250 unique exams')
    return candidates[0], labels[0]

def make_oai_mixed_dataset(cfg):
    global OAI_IMAGES, OAI_LABELS
    original = OAI_IMAGES, OAI_LABELS
    datasets, seen = [], set()
    try:
        for i, (images, labels) in enumerate(OAI_DATASETS):
            OAI_IMAGES, OAI_LABELS = Path(images), Path(labels)
            ds = make_oai_single_dataset(replace(cfg, work_dir=cfg.work_dir / f'source{i}'))
            overlap = seen.intersection(ds.oai_ids)
            if overlap:
                raise ValueError('Repeated exam IDs across OAI datasets; check visit-qualified IDs')
            seen.update(ds.oai_ids)
            datasets.append(ds)
    finally:
        OAI_IMAGES, OAI_LABELS = original
    result = ConcatDataset(datasets)
    result.oai_groups = np.concatenate([d.oai_groups for d in datasets])
    return result

class OAIRatioSampler(Sampler):
    """All RSNA once; fresh participant-balanced OAI subset each epoch."""
    def __init__(self, n_rsna, groups, seed, fold):
        self.n_rsna = n_rsna
        self.groups = np.asarray(groups).astype(str)
        self.n_oai = int(round(n_rsna * 0.42 / 0.58))
        if self.n_oai > len(groups):
            raise ValueError('OAI pool too small for requested 58/42 ratio')
        self.seed, self.fold, self.epoch = seed, fold, 0
        self.by_subject = {g: np.flatnonzero(self.groups == g) for g in np.unique(self.groups)}
    def set_epoch(self, epoch):
        self.epoch = int(epoch)
    def __len__(self):
        return self.n_rsna + self.n_oai
    def __iter__(self):
        rng = np.random.default_rng(self.seed + 100003*self.fold + 1009*self.epoch)
        buckets = [rng.permutation(v).tolist() for v in self.by_subject.values()]
        chosen = []
        # One exam per participant before selecting additional visits/sides.
        while len(chosen) < self.n_oai:
            active = [i for i,b in enumerate(buckets) if b]
            for i in rng.permutation(active):
                chosen.append(buckets[i].pop())
                if len(chosen) == self.n_oai:
                    break
        assert len(set(chosen)) == self.n_oai
        idx = np.concatenate([np.arange(self.n_rsna), self.n_rsna+np.asarray(chosen)])
        return iter(rng.permutation(idx).tolist())

def make_loader(cfg: Cfg, data: FoldData, indices, device, *, train: bool) -> DataLoader:
    indices = np.asarray(indices, dtype=np.int64)
    ds = StudyDataset(
        cfg, data.cache, data.mask,
        data.cache_rows[indices], data.y[indices], data.weight[indices],
        train=train,
    )
    sampler = None
    if train:
        followup = resolve_followup_inputs()
        OAI_DATASETS[:] = [(OAI_IMAGES, OAI_LABELS), followup]
        oai_ds = make_oai_mixed_dataset(cfg)
        n_rsna = len(ds)
        sampler = OAIRatioSampler(n_rsna, oai_ds.oai_groups, cfg.seed, cfg.fold_to_run)
        ds = ConcatDataset([ds, oai_ds])
        log.info('OAI pool=%d; each epoch RSNA=%d OAI=%d (%.2f%% OAI)',
                 len(oai_ds), n_rsna, sampler.n_oai,
                 100*sampler.n_oai/len(sampler))
    return DataLoader(
        ds,
        batch_size=cfg.batch_studies if train else cfg.eval_batch_studies,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        # val_loader 也必须 persistent。若只对 train 生效，验证 worker 每个 epoch
        # 建销一次，进程和 fd 会逐 epoch 累积直至阻塞。
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )


# ============================================================
# 9. MODEL — shared DINOv3 ConvNeXt-Tiny + 5 independent sequence experts
# ============================================================

def resolve_timm_weight(path: Path | None) -> Path | None:
    """接受具体权重文件或 Kaggle Model 根目录，返回本地 checkpoint。"""
    if path is None:
        return None
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        return None

    allowed = {".safetensors", ".pth", ".pt", ".bin"}
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    if not files:
        return None
    priority = {
        "model.safetensors": 0,
        "pytorch_model.bin": 1,
    }
    files.sort(key=lambda p: (priority.get(p.name.lower(), 100), len(p.parts), str(p)))
    return files[0]


class KneeSharedDINOv3ConvNeXtNet(nn.Module):
    """Shared DINOv3 ConvNeXt-Tiny overlapping 3-slice 2.5D + label-wise spatial-temporal experts.

    Input:
        imgs: [B, 5, 24, H, W] uint8 grayscale

    For each available slot:
        24 grayscale slices
        -> 11 overlapping triplets (window=3, stride=2)
        -> adjacent slices become the three channels before normalization
        -> shared DINOv3 ConvNeXt-Tiny final spatial map
        -> label-wise spatial attention: [Bv, 11, 12, 768]
        -> slot-specific shared BiGRU for each label sequence
        -> label-wise temporal Attention + 12-logit head

    The three input channels represent neighboring slices.
    Five expert logits are fused with the same label-wise softmax gate as the baseline.
    """

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.seq_dim = cfg.gru_hidden * 2

        p = resolve_timm_weight(cfg.timm_weight_path)
        if p is None:
            if not cfg.allow_pretrained_download:
                raise FileNotFoundError(
                    f"找不到 DINOv3 ConvNeXt-Tiny 本地权重: {cfg.timm_weight_path}"
                )
            pretrained = True
            overlay = None
            self.pretrained_source = "timm official download/cache"
        else:
            pretrained = True
            overlay = dict(file=str(p))
            self.pretrained_source = str(p)

        kwargs = dict(
            pretrained=pretrained,
            num_classes=0,
            in_chans=cfg.input_channels,
            drop_path_rate=0.0,
        )
        if overlay is not None:
            kwargs["pretrained_cfg_overlay"] = overlay

        self.backbone = timm.create_model(cfg.model_name, **kwargs)
        self.feature_dim = int(self.backbone.num_features)
        if self.feature_dim != 768:
            log.warning("预期 DINOv3 ConvNeXt-Tiny feature_dim=768，实际得到 %d", self.feature_dim)

        # DINOv3 ConvNeXt-Tiny 全量微调，保持最新训练单元的解冻策略。
        for p_ in self.backbone.parameters():
            p_.requires_grad_(True)

        if cfg.convnext_grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        # 每个 slot 的空间注意力在 ConvNeXt 全局池化之前工作。最后一层置零，
        # 初始 softmax 在全部空间位置上均匀，等价于原来的 global average pooling。
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, cfg.attn_hidden),
                nn.Tanh(),
                nn.Linear(cfg.attn_hidden, len(TARGETS)),
            )
            for _ in range(cfg.n_slots)
        ])
        for attention in self.spatial_attentions:
            nn.init.zeros_(attention[-1].weight)
            nn.init.zeros_(attention[-1].bias)

        self.grus = nn.ModuleList([
            nn.GRU(
                input_size=self.feature_dim,
                hidden_size=cfg.gru_hidden,
                num_layers=cfg.gru_layers,
                batch_first=True,
                bidirectional=True,
            )
            for _ in range(cfg.n_slots)
        ])
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_dim, cfg.attn_hidden),
                nn.Tanh(),
                nn.Linear(cfg.attn_hidden, len(TARGETS)),
            )
            for _ in range(cfg.n_slots)
        ])
        # 所有标签从均匀的 11-token 注意力开始，避免随机初始化时过早锁定某些切片。
        for attention in self.attentions:
            nn.init.zeros_(attention[-1].weight)
            nn.init.zeros_(attention[-1].bias)

        self.expert_heads = nn.ModuleList([
            nn.Linear(self.seq_dim, len(TARGETS))
            for _ in range(cfg.n_slots)
        ])

        self.fusion_gate_logits = nn.Parameter(
            torch.zeros(len(TARGETS), cfg.n_slots, dtype=torch.float32)
        )
        self.fusion_bias = nn.Parameter(torch.zeros(len(TARGETS), dtype=torch.float32))

        pcfg = self.backbone.pretrained_cfg
        mean = tuple(pcfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(pcfg.get("std", (0.229, 0.224, 0.225)))
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        total_backbone = sum(p_.numel() for p_ in self.backbone.parameters())
        train_backbone = sum(
            p_.numel() for p_ in self.backbone.parameters() if p_.requires_grad
        )
        log.info(
            "shared DINOv3 ConvNeXt-Tiny | %s | feature_dim=%d | trainable backbone %.2fM / %.2fM",
            cfg.model_name, self.feature_dim,
            train_backbone / 1e6, total_backbone / 1e6,
        )
        log.info(
            "input: 24 grayscale slices -> 11 overlapping 3-slice tokens -> "
            "shared ConvNeXt spatial map -> per-label spatial pooling -> [11,12,%d]",
            self.feature_dim,
        )
        log.info(
            "5 independent shared-across-label BiGRU(hidden=%d) + label-wise "
            "spatial/temporal Attention(hidden=%d) + 12-logit heads",
            cfg.gru_hidden, cfg.attn_hidden,
        )

    def fusion_weights(self) -> torch.Tensor:
        return self.fusion_gate_logits.float().softmax(dim=-1)

    def backbone_parameters(self):
        for p in self.backbone.parameters():
            if p.requires_grad:
                yield p

    def head_parameters(self):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in backbone_ids:
                yield p

    def _run_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run shared ConvNeXt to its final spatial map, with optional chunking."""
        c = int(self.cfg.backbone_chunk_size)
        if c <= 0 or x.shape[0] <= c:
            feat = self.backbone.forward_features(x)
        else:
            parts = []
            for i in range(0, x.shape[0], c):
                parts.append(self.backbone.forward_features(x[i:i + c]))
            feat = torch.cat(parts, dim=0)

        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat) or feat.ndim != 4:
            raise RuntimeError(
                f"ConvNeXt forward_features 预期返回4维空间图，实际类型/shape: "
                f"{type(feat)} / "
                f"{getattr(feat, 'shape', None)}"
            )

        # timm ConvNeXt 默认 NCHW；同时兼容可能的 NHWC output_fmt。
        if feat.shape[1] == self.feature_dim:
            pass
        elif feat.shape[-1] == self.feature_dim:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(
                f"ConvNeXt 空间图找不到 feature_dim={self.feature_dim} 的通道轴: "
                f"{tuple(feat.shape)}"
            )
        return feat

    def _apply_backbone_head(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply the original ConvNeXt post-pool norm/dropout to [N,T,D]."""
        if feat.ndim != 3 or feat.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"backbone head 预期 [N,T,{self.feature_dim}]，得到 {tuple(feat.shape)}"
            )
        n, t, d = feat.shape
        # 1x1 spatial maps let timm's own forward_head preserve the same
        # post-pooling normalization path used by self.backbone(x).
        x = feat.reshape(n * t, d, 1, 1)
        try:
            x = self.backbone.forward_head(x, pre_logits=True)  # ConvNeXt / EfficientNet
        except TypeError:
            x = self.backbone.head(x, pre_logits=True)          # TinyViT: forward_head 不接受 pre_logits
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[-1] != d:
            raise RuntimeError(
                f"ConvNeXt forward_head 预期返回 [{n * t},{d}]，实际 "
                f"{type(x)} / {getattr(x, 'shape', None)}"
            )
        return x.reshape(n, t, d)

    def _encode_slot(
        self,
        slot: int,
        imgs: torch.Tensor,
        valid_idx: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        C, R = cfg.group_size, cfg.img_size
        x = imgs.index_select(0, valid_idx)[:, slot]            # [Bv,24,H,W]
        bv = x.shape[0]

        # ★ 2.5D 重叠三元组：窗口 3、步长 2，起点 0,2,...,20 -> (24-3)//2+1 = 11 个 token；
        #   三个 channel 是相邻 MRI 切片 [2g, 2g+1, 2g+2]，不是同一张灰度图复制。
        x = x.unfold(1, C, 2)                                   # [Bv,11,H,W,3]
        G = x.shape[1]                                          # 11
        x = x.permute(0, 1, 4, 2, 3)                            # [Bv,11,3,H,W]
        x = x.reshape(bv * G, C, R, R).float().div(255.0)
        x = (x - self.mean) / self.std
        x = x.contiguous(memory_format=torch.channels_last)

        feat_map = self._run_backbone_features(x)               # [Bv*11,768,h,w]

        # 空间注意力 / RNN / 时间注意力 / expert head 显式 fp32。
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = bv * G
            pixels = feat_map.float().flatten(2).transpose(1, 2)  # [N,P,768]

            # 每个标签独立选择当前单张切片 内的空间位置。
            spatial_a = self.spatial_attentions[slot](pixels)
            spatial_a = spatial_a.softmax(dim=1)                # [N,P,12]
            label_tokens = torch.einsum(
                "npt,npd->ntd", spatial_a, pixels
            )                                                    # [N,12,768]
            label_tokens = self._apply_backbone_head(label_tokens)
            label_tokens = label_tokens.reshape(
                bv, G, len(TARGETS), self.feature_dim
            ).permute(0, 2, 1, 3).contiguous()                  # [Bv,12,11,768]

            # 12 个标签复用该 slot 的同一个 BiGRU 参数，但输入是各自的空间特征。
            h = label_tokens.reshape(
                bv * len(TARGETS), G, self.feature_dim
            )
            h, _ = self.grus[slot](h)                           # [Bv*12,24,2H]
            h = h.reshape(bv, len(TARGETS), G, self.seq_dim)    # [Bv,12,11,2H]

            # 继续使用原有逐标签时间注意力；只取输入标签与输出标签一致的对角项。
            temporal_all = self.attentions[slot](h)             # [Bv,12,11,12]
            temporal_a = temporal_all.diagonal(
                dim1=1, dim2=3
            ).permute(0, 2, 1).softmax(dim=-1)                  # [Bv,12,6]
            label_feat = torch.einsum(
                "btg,btgd->btd", temporal_a, h
            )                                                    # [Bv,12,2H]

            # 复用原 nn.Linear(2H,12) 的逐标签权重，但不产生 12x12 交叉输出。
            # 每个标签仅用自己的 pooled feature 和自己的分类向量得到一个 logit。
            head = self.expert_heads[slot]
            return torch.einsum(
                "btd,td->bt", label_feat, head.weight
            ) + head.bias                                      # [Bv,12]

    def forward(
        self,
        imgs: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_expert_logits: bool = False,
    ):
        cfg = self.cfg
        B = imgs.shape[0]
        expected = (cfg.n_slots, cfg.slices_per_slot, cfg.img_size, cfg.img_size)
        if imgs.ndim != 5 or tuple(imgs.shape[1:]) != expected:
            raise RuntimeError(
                f"期望 imgs [B,{cfg.n_slots},{cfg.slices_per_slot},{cfg.img_size},{cfg.img_size}]，"
                f"得到 {tuple(imgs.shape)}"
            )

        available = mask > 0.5
        if (~available.any(dim=1)).any():
            raise RuntimeError("发现 5 个 slot 全部缺失的 study，无法进行融合。")

        per_slot = []
        for s in range(cfg.n_slots):
            valid_idx = torch.nonzero(available[:, s], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                full = torch.zeros(
                    B, len(TARGETS), device=imgs.device, dtype=torch.float32
                )
            else:
                valid_logits = self._encode_slot(s, imgs, valid_idx)
                full = torch.zeros(
                    B, len(TARGETS), device=imgs.device, dtype=valid_logits.dtype
                ).index_copy(0, valid_idx, valid_logits)
            per_slot.append(full)

        expert_logits = torch.stack(per_slot, dim=1)           # [B,S,T]

        with torch.autocast(device_type=imgs.device.type, enabled=False):
            gate = self.fusion_gate_logits.float().unsqueeze(0).expand(B, -1, -1)
            gate = gate.masked_fill(~available[:, None, :], -1e4)
            gate = gate.softmax(dim=-1)
            logits = (
                gate * expert_logits.float().permute(0, 2, 1)
            ).sum(dim=-1) + self.fusion_bias.float()

        if return_expert_logits:
            return logits, expert_logits
        return logits


# ============================================================
# 9b. ★ 本次新增: 权重 EMA
# ============================================================

class ModelEma:
    """权重指数滑动平均 (timm ModelEmaV2 的最小实现)。

    几个和本脚本相关的点:

    1. 只在真正的 optimizer.step() 之后调用 update()，不是每个 micro-batch。
       所以 decay 的时间尺度是"有效 batch"(batch_studies * accum_steps)。
       本配置每 epoch 约 200 步，decay=0.995 -> 平均窗口 1/(1-d) = 200 步 ≈ 1 epoch。
       若沿用常见的 0.999，窗口是 1000 步 ≈ 5 epoch，10 epoch 的训练里 EMA
       会被早期权重严重拖住，反而掉点 —— 这是小步数训练最常见的 EMA 翻车方式。

    2. 浮点参数和 buffer 一起做 EMA；整数 buffer（如果存在）直接 copy_。
       ConvNeXt 主要使用 LayerNorm，不依赖 BatchNorm running statistics。

    3. warmup: d = min(decay, (1+n)/(warmup+n))。前几步 head 还是随机初始化，
       不 warmup 的话 EMA 要几百步才能摆脱噪声起点。

    4. EMA 会额外保存一份模型权重；DINOv3 ConvNeXt-Tiny 约 28M 骨干参数，另有任务头参数，需预留相应显存/内存。
    """

    def __init__(self, model: nn.Module, decay: float = 0.995,
                 warmup_steps: int = 10):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

        # deepcopy 后 GRU 的 _flat_weights 不再连续，cuDNN 会 warn；这里重排一次。
        for m in self.module.modules():
            if isinstance(m, nn.RNNBase):
                try:
                    m.flatten_parameters()
                except Exception:
                    pass

        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.updates = 0

    def current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (self.warmup_steps + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.current_decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            mv = msd[k]
            if v.is_floating_point():
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)          # num_batches_tracked 等整数 buffer

    def state_dict(self):
        return self.module.state_dict()


# ============================================================
# 10. METRIC / PREDICT
# ============================================================

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict]:
    scores, valid = {}, []
    for j, target in enumerate(TARGETS):
        if len(np.unique(y_true[:, j])) < 2:
            scores[target] = None
            continue
        auc = float(roc_auc_score(y_true[:, j], y_pred[:, j]))
        scores[target] = auc
        valid.append(auc)
    return (float(np.mean(valid)) if valid else float("nan")), scores


@torch.inference_mode()
def predict(cfg: Cfg, model, loader, device, desc: str = "Validation") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_y = [], []

    bar = pbar(loader, total=len(loader), desc=desc, unit="batch", leave=False)
    try:
        for imgs, mask, y, _w in bar:
            imgs = imgs.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(cfg.use_amp and device.type == "cuda"),
            ):
                logits = model(imgs, mask)

            all_pred.append(torch.sigmoid(logits.float()).cpu().numpy())
            all_y.append(y.numpy())
    finally:
        bar.close()

    return np.concatenate(all_y, axis=0), np.concatenate(all_pred, axis=0)


def run_validation(cfg: Cfg, model, val_loader, gold_pos: np.ndarray, device,
                   desc: str = "Validation (GRU+Attn)") -> tuple[ValMetrics, np.ndarray]:
    """gold_pos 是 gold study 在 val_loader 输出中的行位置。

    val_loader 用 shuffle=False / drop_last=False，输出行序与 val_idx 一致；
    data.y 的 gold 行存的就是 gold 值 (combine_first 优先 gold)，直接切片即可。
    验证集不做任何 augment (train=False)。
    """
    val_y_soft, val_pred = predict(cfg, model, val_loader, device, desc=desc)
    derived_auc, derived_per_label = macro_auc(
        (val_y_soft >= 0.5).astype(np.uint8), val_pred
    )

    gold_auc = float("nan")
    gold_per_label = {t: None for t in TARGETS}

    if len(gold_pos) >= cfg.min_gold_val:
        gold_auc, gold_per_label = macro_auc(
            val_y_soft[gold_pos].astype(np.uint8), val_pred[gold_pos]
        )

    selection = SELECTORS[cfg.selection](derived_auc, gold_auc)

    return ValMetrics(
        derived_auc=derived_auc,
        gold_auc=gold_auc,
        selection=selection,
        n_gold_val=int(len(gold_pos)),
        derived_per_label=derived_per_label,
        gold_per_label=gold_per_label,
    ), val_pred


def save_ckpt(cfg: Cfg, model, path: Path, epoch: int, m: ValMetrics,
              *, is_ema: bool) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "targets": TARGETS,
        "architecture": "shared_dinov3_convnext_tiny_2p5d11_oai58_42_labelwise_spatiotemporal_attention_v3",
        "model_name": cfg.model_name,
        "pretrained_source": model.pretrained_source,
        "feature_dim": model.feature_dim,
        "seq_dim": model.seq_dim,
        "gru_hidden": cfg.gru_hidden,
        "gru_layers": cfg.gru_layers,
        "attn_hidden": cfg.attn_hidden,
        "shared_backbone_across_slots": True,
        "input_mode": "3slice_2p5d_overlap_window3_stride2",
        "sequence_tokens_per_slot": (cfg.slices_per_slot - cfg.group_size) // 2 + 1,
        "group_size": cfg.group_size,
        "n_groups": cfg.n_groups,
        "convnext_grad_checkpointing": cfg.convnext_grad_checkpointing,
        "backbone_chunk_size": cfg.backbone_chunk_size,
        "backbone_finetune": "full",
        "expert_aux_weight": cfg.expert_aux_weight,
        "spatial_attention": "per_slot_per_label_softmax_over_final_convnext_spatial_map",
        "spatial_attention_zero_initialized": True,
        "temporal_attention": "per_slot_per_label_softmax_over_11_tokens",
        "fusion": "per_label_global_softmax_gate_with_missing_slot_mask",
        "fusion_weights_no_missing": model.fusion_weights().detach().cpu(),
        "img_size": cfg.img_size,
        "slice_order": "physical_position_with_instance_fallback",
        "n_slots": cfg.n_slots,
        "slices_per_slot": cfg.slices_per_slot,
        "crop_mm": cfg.crop_mm,
        "label_mix": cfg.label_desc,
        "aug": cfg.aug_desc,
        "aug_rotate_p": cfg.aug_rotate_p,
        "aug_rotate_limit": cfg.aug_rotate_limit,
        "aug_perspective_p": cfg.aug_perspective_p,
        "aug_perspective_scale": tuple(cfg.aug_perspective_scale),
        "aug_seq_flip_p": cfg.aug_seq_flip_p,
        "aug_depth_jitter": cfg.aug_depth_jitter,
        "aug_intensity": bool(cfg.aug_intensity),
        "aug_intensity_desc": cfg.intensity_desc,
        "aug_bc_p": cfg.aug_bc_p,
        "aug_brightness_limit": cfg.aug_brightness_limit,
        "aug_contrast_limit": cfg.aug_contrast_limit,
        "aug_gamma_p": cfg.aug_gamma_p,
        "aug_gamma_range": tuple(cfg.aug_gamma_range),
        "aug_bias_p": cfg.aug_bias_p,
        "aug_bias_strength": cfg.aug_bias_strength,
        "aug_bias_grid": cfg.aug_bias_grid,
        "aug_noise_p": cfg.aug_noise_p,
        "aug_noise_sigma": tuple(cfg.aug_noise_sigma),
        "aug_blur_p": cfg.aug_blur_p,
        "aug_blur_sigma": tuple(cfg.aug_blur_sigma),
        "aug_lowres_p": cfg.aug_lowres_p,
        "aug_lowres_scale": tuple(cfg.aug_lowres_scale),
        "aug_dropout_p": cfg.aug_dropout_p,
        "aug_dropout_holes": tuple(cfg.aug_dropout_holes),
        "ema": bool(is_ema),
        "ema_decay": cfg.ema_decay if is_ema else None,
        "ema_warmup_steps": cfg.ema_warmup_steps if is_ema else None,
        "train_sequence_strategy": "11_overlap_3slice_2p5d_tokens_shared_dinov3_convnext_tiny_labelwise_spatial_attn_per_slot_gru_labelwise_temporal_attn",
        "val_sequence_strategy": "11_fixed_overlap_3slice_2p5d_tokens_noaug",
        "memory_format": "channels_last",
        "selection": cfg.selection,
        "lr": cfg.lr,
        "backbone_lr": cfg.backbone_lr,
        "head_lr": cfg.lr,
        "batch_studies": cfg.batch_studies,
        "accum_steps": cfg.accum_steps,
        "epoch": epoch,
        "derived_auc": m.derived_auc,
        "gold_auc": m.gold_auc,
        "selection_score": m.selection,
        "n_gold_val": m.n_gold_val,
    }, path)


# ============================================================
# 11. TRAIN
# ============================================================

def train_one_epoch(cfg: Cfg, model, loader, optimizer, loss_fn,
                    device, epoch: int, ema: ModelEma | None = None) -> dict:
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    model.train()
    total_loss = total_main = total_aux = 0.0
    n_batch = 0
    optimizer.zero_grad(set_to_none=True)

    n_steps = len(loader)
    bar = pbar(loader, total=n_steps, desc=f"Epoch {epoch:02d}/{cfg.epochs}",
               unit="batch", leave=False)
    try:
        for step, (imgs, m, yb, wb) in enumerate(bar):
            imgs = imgs.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(cfg.use_amp and device.type == "cuda"),
            ):
                logits, expert_logits = model(
                    imgs, m, return_expert_logits=True
                )

                main_raw = loss_fn(logits, yb)                 # [B,T]
                main_loss = (main_raw * wb).sum() / wb.sum().clamp_min(1e-6)

                # 每个真实存在的 slot 都接受同一个 study 标签监督。
                # 这是防止 gate 早期塌缩的 deep supervision；缺失 slot 不计 loss。
                aux_raw = loss_fn(
                    expert_logits,
                    yb[:, None, :].expand_as(expert_logits),
                )                                               # [B,S,T]
                valid = (m > 0.5).float().unsqueeze(-1)         # [B,S,1]
                aux_weight = wb[:, None, :] * valid             # [B,S,T]
                aux_loss = (
                    (aux_raw * aux_weight).sum()
                    / aux_weight.sum().clamp_min(1e-6)
                )
                loss = main_loss + cfg.expert_aux_weight * aux_loss
                # ★ B4: 批内逐标签成对 logistic 排序项（阳性 logit 应大于阴性），权重 0.1
                _pos = (yb >= 0.5).float()
                _pair = _pos[:, None, :] * (1.0 - _pos)[None, :, :]
                _diff = logits.float()[:, None, :] - logits.float()[None, :, :]
                rank_loss = (F.softplus(-_diff) * _pair).sum() / _pair.sum().clamp_min(1.0)
                loss = loss + 0.1 * rank_loss

            (loss / cfg.accum_steps).backward()

            # 最后一个不完整的累积窗口也要 step，否则那部分梯度被丢弃
            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == n_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # ★ EMA 跟着 optimizer step 走，不是跟着 micro-batch 走
                if ema is not None:
                    ema.update(model)

            total_loss += float(loss.item())
            total_main += float(main_loss.item())
            total_aux += float(aux_loss.item())
            n_batch += 1
            bar.set_postfix(
                loss=f"{total_loss / n_batch:.4f}",
                main=f"{total_main / n_batch:.4f}",
                aux=f"{total_aux / n_batch:.4f}",
            )
    finally:
        bar.close()

    den = max(n_batch, 1)
    return {
        "loss": total_loss / den,
        "main_loss": total_main / den,
        "expert_aux_loss": total_aux / den,
    }


def train_fold(cfg: Cfg, data: FoldData, device) -> None:
    splits = list(GroupKFold(n_splits=cfg.n_folds).split(
        np.zeros(len(data.y)), groups=data.groups
    ))
    train_idx, val_idx = splits[cfg.fold_to_run]
    gold_pos = np.flatnonzero(data.is_gold[val_idx])

    log.info(
        "fold%d | train %d | val %d | gold in train %d | gold in val %d",
        cfg.fold_to_run, len(train_idx), len(val_idx),
        int(data.is_gold[train_idx].sum()), len(gold_pos),
    )

    train_loader = val_loader = None
    try:
        train_loader = make_loader(cfg, data, train_idx, device, train=True)
        val_loader = make_loader(cfg, data, val_idx, device, train=False)

        model = KneeSharedDINOv3ConvNeXtNet(cfg).to(
            device, memory_format=torch.channels_last
        )

        backbone_params = list(model.backbone_parameters())
        head_params = list(model.head_parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.backbone_lr},
                {"params": head_params, "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        log.info(
            "optimizer: AdamW | backbone %.2fM @ %.2e | head %.2fM @ %.2e",
            sum(p.numel() for p in backbone_params) / 1e6,
            cfg.backbone_lr,
            sum(p.numel() for p in head_params) / 1e6,
            cfg.lr,
        )

        # ---- ★ EMA ----
        ema = None
        if cfg.use_ema:
            ema = ModelEma(model, decay=cfg.ema_decay,
                           warmup_steps=cfg.ema_warmup_steps)
            steps_per_epoch = int(np.ceil(len(train_loader) / cfg.accum_steps))
            horizon = 1.0 / (1.0 - cfg.ema_decay)
            log.info(
                "EMA: decay %g | warmup %d 步 | 每 epoch %d 次更新 | "
                "平均窗口 ≈ %.0f 步 ≈ %.2f epoch",
                cfg.ema_decay, cfg.ema_warmup_steps, steps_per_epoch,
                horizon, horizon / max(steps_per_epoch, 1),
            )
            if horizon > 3.0 * steps_per_epoch:
                log.warning(
                    "EMA 窗口超过 3 个 epoch，10 epoch 的训练里可能被早期权重拖住，"
                    "考虑把 ema_decay 调小。"
                )

        # 验证 / 存盘用的模型：开 EMA 就是 EMA 权重
        eval_model = ema.module if ema is not None else model
        eval_tag = "EMA" if ema is not None else "raw"

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("trainable params: %.2fM | backbone LR = %g | head LR = %g | channels_last | "
                 "cudnn.benchmark %s | eval on %s",
                 trainable / 1e6, cfg.backbone_lr, cfg.lr, torch.backends.cudnn.benchmark, eval_tag)

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        history, best_score = [], -np.inf
        best_derived_score = -np.inf
        best_derived_epoch = None

        epoch_bar = pbar(range(1, cfg.epochs + 1), total=cfg.epochs,
                         desc=f"Training fold{cfg.fold_to_run}", unit="epoch")
        try:
            for epoch in epoch_bar:
                train_stats = train_one_epoch(
                    cfg, model, train_loader, optimizer, loss_fn, device, epoch,
                    ema=ema,
                )
                mean_loss = train_stats["loss"]

                # 对照用的 raw 权重验证 (可关，关掉验证耗时减半)
                raw_metrics = None
                if ema is not None and cfg.ema_eval_raw:
                    raw_metrics, _ = run_validation(
                        cfg, model, val_loader, gold_pos, device,
                        desc="Validation (raw)",
                    )

                metrics, _ = run_validation(
                    cfg, eval_model, val_loader, gold_pos, device,
                    desc=f"Validation ({eval_tag})",
                )

                # 注: derived_auc / gold_auc / selection_score 这几列一律是
                #     "被用来选 ckpt 的那个模型"的指标，raw_* 才是对照。
                row = {
                    "epoch": epoch,
                    "loss": mean_loss,
                    "main_loss": train_stats["main_loss"],
                    "expert_aux_loss": train_stats["expert_aux_loss"],
                    "derived_auc": metrics.derived_auc,
                    "gold_auc": metrics.gold_auc,
                    "selection_score": metrics.selection,
                    "n_gold_val": metrics.n_gold_val,
                    "eval_on": eval_tag,
                }
                if raw_metrics is not None:
                    row.update({
                        "raw_derived_auc": raw_metrics.derived_auc,
                        "raw_gold_auc": raw_metrics.gold_auc,
                        "raw_selection_score": raw_metrics.selection,
                    })
                history.append(row)

                gold_txt = (
                    f"{metrics.gold_auc:.4f}" if np.isfinite(metrics.gold_auc) else "NA"
                )
                postfix = dict(
                    loss=f"{mean_loss:.4f}",
                    scored=f"{metrics.derived_auc:.4f}",
                    gold=gold_txt,
                )
                if raw_metrics is not None:
                    postfix["raw"] = f"{raw_metrics.derived_auc:.4f}"
                epoch_bar.set_postfix(**postfix)

                raw_txt = ""
                if raw_metrics is not None:
                    raw_gold = (
                        f"{raw_metrics.gold_auc:.5f}"
                        if np.isfinite(raw_metrics.gold_auc) else "NA"
                    )
                    raw_txt = (
                        f" || raw derived {raw_metrics.derived_auc:.5f} "
                        f"| raw gold {raw_gold}"
                    )
                log.info(
                    "epoch %02d/%d | loss %.5f (main %.5f + %.3f*aux %.5f) | "
                    "[%s] derived AUC %.5f | "
                    "gold AUC %s | select %.5f%s",
                    epoch, cfg.epochs, mean_loss,
                    train_stats["main_loss"], cfg.expert_aux_weight,
                    train_stats["expert_aux_loss"], eval_tag, metrics.derived_auc,
                    gold_txt, metrics.selection, raw_txt,
                )

                if np.isfinite(metrics.selection) and metrics.selection > best_score:
                    best_score = metrics.selection
                    save_ckpt(cfg, eval_model, cfg.out_dir / "fold0_best.pt",
                              epoch, metrics, is_ema=(ema is not None))
                # 保留 final 文件名供现有推理使用，内容改为 derived AUC 最高轮。
                if np.isfinite(metrics.derived_auc) and metrics.derived_auc > best_derived_score:
                    best_derived_score = metrics.derived_auc
                    best_derived_epoch = epoch
                    save_ckpt(cfg, eval_model, cfg.out_dir / "fold0_final.pt",
                              epoch, metrics, is_ema=(ema is not None))
        finally:
            epoch_bar.close()

        if best_derived_epoch is None:
            raise RuntimeError("没有有效的 derived AUC，无法保存最佳 derived 权重。")
        # 导出的验证预测和融合权重与 final 文件实际保存的模型保持一致。
        derived_ckpt = torch.load(
            cfg.out_dir / "fold0_final.pt", map_location="cpu", weights_only=False,
        )
        eval_model.load_state_dict(derived_ckpt["state_dict"])
        del derived_ckpt
        log.info("final 文件使用最佳 derived 权重: epoch %d | derived AUC %.5f",
                 best_derived_epoch, best_derived_score)

        final, val_pred = run_validation(
            cfg, eval_model, val_loader, gold_pos, device,
            desc=f"Validation ({eval_tag})",
        )

        pd.DataFrame(history).to_csv(cfg.out_dir / "history.csv", index=False)

        pred_df = pd.DataFrame(val_pred, columns=TARGETS)
        pred_df.insert(0, "StudyInstanceUID", data.study_ids[val_idx])
        pred_df.insert(1, "is_gold", data.is_gold[val_idx].astype(np.uint8))
        pred_df.to_csv(cfg.out_dir / "fold0_val_predictions.csv", index=False)

        fusion_df = pd.DataFrame(
            eval_model.fusion_weights().detach().cpu().numpy(),
            index=TARGETS,
            columns=[slot[0] for slot in SLOTS],
        )
        fusion_df.index.name = "target"
        fusion_df.to_csv(cfg.out_dir / "fusion_weights_final.csv")

        gold_txt = f"{final.gold_auc:.5f}" if np.isfinite(final.gold_auc) else "NA"
        print(f"\n[eval on {eval_tag}]")
        print(f"FINAL derived AUC : {final.derived_auc:.5f}")
        print(f"FINAL gold AUC    : {gold_txt}")
        print(f"BEST selection    : {best_score:.5f}")

        for title, per_label in [
            ("Per-label derived AUC", final.derived_per_label),
            ("Per-label GOLD AUC (official held-out only)", final.gold_per_label),
        ]:
            print(f"\n{title}:")
            for target, auc in per_label.items():
                print(f"{target:18s} {'NA' if auc is None else f'{auc:.5f}'}")

        print("\nFINAL label-wise fusion weights (无缺失 slot 时):")
        print(fusion_df.round(4).to_string())

    finally:
        shutdown_loader(train_loader)
        shutdown_loader(val_loader)
        del train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 12. MULTI-FOLD
# ============================================================

def collect_oof(cfg: Cfg) -> pd.DataFrame | None:
    frames = []
    for fold in range(cfg.n_folds):
        path = cfg.work_dir / f"fold{fold}" / "fold0_val_predictions.csv"
        if not path.exists():
            log.warning("缺少 %s，跳过该 fold", path)
            continue
        df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
        df.insert(0, "fold", fold)
        frames.append(df)

    if not frames:
        log.warning("没有任何 fold 的预测文件，跳过 OOF 汇总。")
        return None

    oof = pd.concat(frames, ignore_index=True)

    dup = oof["StudyInstanceUID"].duplicated().sum()
    if dup:
        log.warning("OOF 中有 %d 条重复 StudyInstanceUID", dup)

    oof_path = cfg.work_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    log.info("OOF 预测已写出: %s (%d 条, 来自 %d 个 fold)",
             oof_path, len(oof), len(frames))

    # ---- derived per-label AUC: 本次实验的主判据 ----
    # 4400 样本，SE ~0.005，读得出。macro 会把只受几何变异影响的 label
    # 稀释掉，所以必须看 per-label。
    llm, _, _ = load_vlm_labels(cfg)
    soft = llm.reindex(oof["StudyInstanceUID"])[TARGETS].to_numpy(dtype=np.float64)
    ok = np.isfinite(soft).all(axis=1)
    if ok.sum() >= 100:
        d_macro, d_per = macro_auc(
            (soft[ok] >= 0.5).astype(np.uint8),
            oof.loc[ok, TARGETS].to_numpy(dtype=np.float64),
        )
        print("\n" + "=" * 64)
        print(f"OOF DERIVED macro AUC ({int(ok.sum())} studies): {d_macro:.5f}")
        print("=" * 64)
        for target, auc in d_per.items():
            print(f"{target:18s} {'NA' if auc is None else f'{auc:.5f}'}")
        pd.DataFrame(
            [{"target": t, "oof_derived_auc": a} for t, a in d_per.items()]
        ).to_csv(cfg.work_dir / "oof_derived_auc.csv", index=False)

    # ---- official gold (58 个样本，SE ~0.010-0.018，仅供参考) ----
    gold_pred = oof[oof["is_gold"] == 1].drop_duplicates("StudyInstanceUID")

    if len(gold_pred) < cfg.min_gold_val:
        log.warning("OOF 中仅 %d 条 gold，少于 min_gold_val=%d，不计算 AUC。",
                    len(gold_pred), cfg.min_gold_val)
        return oof

    truth = (
        pd.read_csv(str(cfg.train_csv), dtype={"StudyInstanceUID": str})
        .drop_duplicates("StudyInstanceUID")
        .set_index("StudyInstanceUID")
        .reindex(gold_pred["StudyInstanceUID"])[TARGETS]
    )

    macro, per_label = macro_auc(
        truth.to_numpy(dtype=np.float64).astype(np.uint8),
        gold_pred[TARGETS].to_numpy(dtype=np.float64),
    )

    print("\n" + "=" * 64)
    print(f"OOF GOLD macro AUC ({len(gold_pred)} studies, {len(frames)} folds): "
          f"{macro:.5f}   [SE ~0.010-0.018，别据此下结论]")
    print("=" * 64)
    for target, auc in per_label.items():
        print(f"{target:18s} {'NA' if auc is None else f'{auc:.5f}'}")

    pd.DataFrame(
        [{"target": t, "oof_gold_auc": a} for t, a in per_label.items()]
    ).to_csv(cfg.work_dir / "oof_gold_auc.csv", index=False)

    return oof


# ============================================================
# 13. AUG 自检 (可选，跑训练前建议先看一眼)
# ============================================================

def preview_aug(cfg: Cfg | None = None, n: int = 6, row: int = 0,
                out_path: Path | None = None) -> Path:
    """把同一个 study 的中间层在 n 种随机几何变换下画出来。

    确认三件事:
      1. 旋转角度落在 ±25 度内，不是把膝关节转成奇怪的朝向
      2. 背景是黑的 (BORDER_CONSTANT=0)，不是镜像/复制边缘
      3. 5 个 slot 的变换完全一致
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or Cfg()
    cache, _, _ = load_cache(cfg)
    imgs0 = np.array(cache[row])                     # [5,24,H,W]

    fig, axes = plt.subplots(cfg.n_slots, n + 1, figsize=(2.0 * (n + 1), 2.0 * cfg.n_slots))
    mid = cfg.slices_per_slot // 2

    for s in range(cfg.n_slots):
        axes[s, 0].imshow(imgs0[s, mid], cmap="gray")
        axes[s, 0].set_ylabel(SLOTS[s][0], fontsize=7)
        axes[s, 0].set_xticks([]); axes[s, 0].set_yticks([])
        if s == 0:
            axes[s, 0].set_title("original", fontsize=8)

    for k in range(n):
        M = sample_geo_matrix(cfg, cfg.img_size)
        aug = imgs0 if M is None else apply_geo(imgs0, M)
        for s in range(cfg.n_slots):
            ax = axes[s, k + 1]
            ax.imshow(aug[s, mid], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if s == 0:
                ax.set_title("identity" if M is None else f"aug {k}", fontsize=8)

    plt.tight_layout()
    out_path = out_path or (cfg.work_dir / "aug_preview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info("aug 预览已写出: %s (每一列应当 5 个 slot 变换一致)", out_path)
    return out_path


def preview_intensity(cfg: Cfg | None = None, n: int = 6, row: int = 0,
                      out_path: Path | None = None) -> Path:
    """★ 只画强度增强 (不叠加几何)，用来判断参数是不是太猛。

    跑训练前建议先看一眼，确认三件事:
      1. 膝关节结构还认得出来 —— gamma 作用在已经分位归一化过的图上，
         视觉效果可能比预期强。太猛就把 aug_gamma_range 收到 (0.8, 1.25)
      2. 背景仍然接近黑 —— 亮度平移和偏置场会把背景抬起来一点，正常；
         但如果背景变成灰白一片，说明 aug_brightness_limit 太大
      3. 5 个 slot 的扰动**各不相同** —— 这是设计如此(逐序列)，
         与几何增强的「5 个 slot 完全一致」正好相反

    与 preview_aug 分开是为了排除旋转的干扰: 两个一起看判断不了是哪一项太猛。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or Cfg()
    cache, mask, _ = load_cache(cfg)
    imgs0 = np.array(cache[row])                     # [5,24,H,W]
    m0 = mask[row].astype(np.float32)

    fig, axes = plt.subplots(cfg.n_slots, n + 1,
                             figsize=(2.0 * (n + 1), 2.0 * cfg.n_slots))
    mid = cfg.slices_per_slot // 2

    # 全部用同一个 vmin/vmax，否则 imshow 每格自动拉伸，看不出亮度变化
    for s in range(cfg.n_slots):
        axes[s, 0].imshow(imgs0[s, mid], cmap="gray", vmin=0, vmax=255)
        axes[s, 0].set_ylabel(f"{SLOTS[s][0]}\nmask={m0[s]:.0f}", fontsize=6)
        axes[s, 0].set_xticks([]); axes[s, 0].set_yticks([])
        if s == 0:
            axes[s, 0].set_title("original", fontsize=8)

    for k in range(n):
        aug = StudyDataset._intensity_aug(imgs0, m0, cfg)
        for s in range(cfg.n_slots):
            ax = axes[s, k + 1]
            ax.imshow(aug[s, mid], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if s == 0:
                ax.set_title(f"aug {k}", fontsize=8)
            d = float(np.abs(aug[s, mid].astype(np.int16)
                             - imgs0[s, mid].astype(np.int16)).mean())
            ax.set_xlabel(f"|Δ|={d:.1f}", fontsize=6)

    plt.tight_layout()
    out_path = out_path or (cfg.work_dir / "intensity_preview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info("强度增强预览: %s", out_path)
    log.info("  每一行是一个 slot，扰动应当各不相同(逐序列)；空 slot 那行应当全无变化")
    return out_path


# ============================================================
# 14. MAIN
# ============================================================

def main(cfg: Cfg | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(TqdmLoggingHandler())
    root.setLevel(logging.INFO)

    cfg = cfg or Cfg()
    seed_everything(cfg.seed)

    if not cfg.cache_dirs:
        raise ValueError("cache_dirs 不能为空")
    for directory in cfg.cache_dirs:
        for filename in ("train_cache.npy", "train_mask.npy", "study_ids.npy", "manifest.json"):
            path = Path(directory) / filename
            if not path.is_file():
                raise FileNotFoundError(f"找不到缓存分片文件: {path}")

    for name, p in [
        ("train.csv", cfg.train_csv),
        ("scored_path (VLM)", cfg.scored_path),
        ("v1_path", cfg.v1_path),
        ("v2_path", cfg.v2_path),
        ("ConvNeXt weights", cfg.timm_weight_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"找不到 {name}: {p}")

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 64)
    log.info("Knee MRI Training | shared DINOv3 ConvNeXt-Tiny overlapping 3-slice 2.5D + "
             "label-wise spatial-temporal attention + static label-wise fusion | EMA | bf16")
    log.info("=" * 64)
    log.info("device %s | timm %s | model %s", device, timm.__version__, cfg.model_name)
    log.info("实验变量: 训练标签 = %s；其余与 VLM 标签版完全相同", cfg.label_desc)
    log.info("DINOv3 ConvNeXt-Tiny 全量微调 | feature_dim=768")
    log.info("专家辅助 BCE 权重 %.3f；缺失 slot 不进 encoder、不计辅助 loss、不参与 gate",
             cfg.expert_aux_weight)
    log.info("cache_dirs %s", cfg.cache_dirs)
    log.info(
        "img %d | crop %.0fmm | slots %d | slices/slot %d | "
        "seq_len %d | group_size %d | input=overlapping 3-slice 2.5D",
        cfg.img_size, cfg.crop_mm, cfg.n_slots, cfg.slices_per_slot,
        (cfg.slices_per_slot - cfg.group_size) // 2 + 1, cfg.group_size,
    )
    log.info(
        "batch %d x accum %d = 有效 %d | epochs %d | backbone lr %.2e | head lr %.2e | amp bf16 | selection %s",
        cfg.batch_studies, cfg.accum_steps, cfg.effective_batch,
        cfg.epochs, cfg.backbone_lr, cfg.lr, cfg.selection,
    )
    log.info(
        "每 micro-batch 最多 DINOv3 ConvNeXt 2.5D 三元组数 %d (=B*slots*11, shared backbone) | "
        "chunk=%d | grad_ckpt=%s",
        cfg.batch_studies * cfg.n_slots * ((cfg.slices_per_slot - cfg.group_size) // 2 + 1),
        cfg.backbone_chunk_size, cfg.convnext_grad_checkpointing,
    )
    log.info("train aug: %s", cfg.aug_desc)
    log.info("           几何变换整个 study (5 slots x 24 slices) 共用同一矩阵")
    log.info("           不含 HFlip/VFlip —— 会交换内外侧/股胫，破坏 label")
    log.info("intensity: %s", cfg.intensity_desc)
    log.info("           逐 slot(=逐序列)独立采样，slot 内 24 层共用同一组参数")
    log.info("           空 slot (mask=0) 跳过 —— 不送入 ConvNeXt，也不计辅助 loss")
    log.info("           施加于几何变换之前 —— 保证 warp 的补零区是纯黑")
    log.info("val aug:   none")
    log.info("EMA:       %s | 验证/选 ckpt/存盘 %s | raw 对照验证 %s",
             cfg.ema_desc,
             "走 EMA 权重" if cfg.use_ema else "走 raw 权重",
             "on" if (cfg.use_ema and cfg.ema_eval_raw) else "off")
    log.info("=" * 64)

    cache, mask, study_ids = load_cache(cfg)

    data = load_labels(cfg, study_ids)
    data.cache = cache
    data.mask = mask

    train_fold(cfg, data, device)


def run_all_folds(cfg: Cfg | None = None, folds: Sequence[int] | None = None) -> None:
    cfg = cfg or Cfg()
    folds = tuple(range(cfg.n_folds)) if folds is None else tuple(folds)

    bad = [f for f in folds if not 0 <= f < cfg.n_folds]
    if bad:
        raise ValueError(f"fold 必须在 0..{cfg.n_folds - 1}，得到 {bad}")

    t_start = time.time()
    for i, fold in enumerate(folds, 1):
        t_fold = time.time()
        print("\n" + "#" * 64)
        print(f"# FOLD {fold}  ({i}/{len(folds)})  aug={cfg.aug_desc}  ema={cfg.ema_desc}")
        print("#" * 64)

        main(replace(cfg, fold_to_run=fold))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_fold
        log.info("fold%d 耗时 %.1f 分钟 | 累计 %.1f 分钟",
                 fold, elapsed / 60, (time.time() - t_start) / 60)

    collect_oof(cfg)
    log.info("全部完成，总耗时 %.1f 分钟", (time.time() - t_start) / 60)


if __name__ == "__main__":
    # 默认依次训练 fold0、fold1、fold2、fold3，每折 16 epochs。
    run_all_folds(Cfg(), folds=(0, 1, 2, 3))
