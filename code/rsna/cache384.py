#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knee MRI: 中央 80% + 24 slices, 384px / crop130, 四分包。
使用：在四个 Kaggle Notebook 中分别设置 PART_ID = 0、1、2、3，运行全文。
按 train.csv 中唯一 StudyInstanceUID 的字典序，np.array_split 分成四份。
4407 studies -> 1102 / 1102 / 1102 / 1101；最大分片约 19.50 GB（18.16 GiB）。
输出各自目录下 train_cache.npy / train_mask.npy / study_ids.npy / manifest.json。
保持原物理排序、laterality、slot 选择、裁剪和强度处理逻辑。
中央 80% 不足24层时居中扩展；全序列不足24层才重复索引。
训练与推理均需适配24层，并按每份 study_ids.npy 映射数据。
依赖：pip install pydicom pandas numpy tqdm torch
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pydicom
from numpy.lib.format import open_memmap
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

log = logging.getLogger("knee_preprocess")


# ------------------------------------------------------------
# tqdm.notebook 半初始化告警的静默处理
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

COLUMN_ALIASES = {
    "study": ["StudyInstanceUID", "Study Instance UID"],
    "series": ["SeriesInstanceUID", "Series Instance UID"],
    "plane": ["Plane", "AnatomicalPlane", "Anatomical Plane", "Anatomical_Plane"],
    "fluid": ["Fluid_Sensitive", "Fluid Sensitive", "FluidSensitive"],
    "fatsat": ["Fat_Suppression", "Fat Suppression", "FatSuppression"],
}


# ============================================================
# 2. CONFIG
# ============================================================

# 分别运行四次：PART_ID 依次为 0、1、2、3，每次只生成一份。
PART_ID = 0
N_PARTS = 4


@dataclass(frozen=True)
class PreprocessCfg:
    # paths
    data_root: Path = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
    part_id: int = PART_ID
    n_parts: int = N_PARTS
    output_root: Path = Path("/kaggle/working")

    @property
    def cache_dir(self) -> Path:
        return self.output_root / (
            f"knee_v5_{self.img_size}px_{self.slices_per_slot}sl_central80_"
            f"crop{self.crop_mm:g}_physicalsort_part{self.part_id}"
        )

    # image
    img_size: int = 384
    crop_mm: float = 130.0
    n_slots: int = 5
    slices_per_slot: int = 24

    # laterality
    # |x| 小于此值时认为关节位于等中心附近、符号无信息 -> 不做兜底。
    # 判不出就不翻是安全的: 翻错比不翻更糟。
    lat_min_offset_mm: float = 5.0

    # runtime
    header_workers: int = 12
    pixel_workers: int = 8

    def __post_init__(self) -> None:
        if self.n_parts < 1 or not 0 <= self.part_id < self.n_parts:
            raise ValueError("n_parts 必须为正整数，part_id 必须在 0 到 n_parts-1 之间")
        if self.n_slots != len(SLOTS):
            raise ValueError(f"n_slots={self.n_slots} 与 SLOTS 的 {len(SLOTS)} 项不符")

    @property
    def train_dicom_dir(self) -> Path:
        return self.data_root / "train_series"

    @property
    def train_csv(self) -> Path:
        return self.data_root / "train.csv"

    @property
    def train_series_csv(self) -> Path:
        return self.data_root / "train_series.csv"

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "train_cache.npy"

    @property
    def mask_path(self) -> Path:
        return self.cache_dir / "train_mask.npy"

    @property
    def ids_path(self) -> Path:
        return self.cache_dir / "study_ids.npy"


# ============================================================
# 3. UTILS
# ============================================================

def pbar(iterable=None, **kwargs):
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(iterable, **kwargs)


def find_col(df: pd.DataFrame, names: Sequence[str]) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name.lower().strip() in lower:
            return lower[name.lower().strip()]
    raise KeyError(f"找不到列 {list(names)}；实际 columns = {list(df.columns)}")


def to_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(int(x))
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def normalize_plane(x) -> str | None:
    s = str(x).strip().lower()
    if "sag" in s:
        return "Sagittal"
    if "cor" in s:
        return "Coronal"
    if "axi" in s or "tra" in s:
        return "Axial"
    return None


def report_hash(text) -> str:
    if not isinstance(text, str):
        text = ""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# 4. SERIES TABLE
# ============================================================

def load_series_csv(cfg: PreprocessCfg) -> pd.DataFrame:
    df = pd.read_csv(
        cfg.train_series_csv,
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    out = pd.DataFrame({
        "StudyInstanceUID": df[find_col(df, COLUMN_ALIASES["study"])].astype(str),
        "SeriesInstanceUID": df[find_col(df, COLUMN_ALIASES["series"])].astype(str),
        "plane": df[find_col(df, COLUMN_ALIASES["plane"])].map(normalize_plane),
        "fluid": df[find_col(df, COLUMN_ALIASES["fluid"])].map(to_bool),
        "fatsat": df[find_col(df, COLUMN_ALIASES["fatsat"])].map(to_bool),
    })
    return out.dropna(subset=["plane"]).reset_index(drop=True)


def dicom_files(cfg: PreprocessCfg, study_uid: str, series_uid: str) -> list[Path]:
    folder = cfg.train_dicom_dir / study_uid / series_uid
    return sorted(folder.glob("*.dcm")) if folder.exists() else []


_ORDER_TAGS = [
    (0x0020, 0x0013),  # InstanceNumber（仅作为回退与稳定 tie-break）
    (0x0020, 0x0032),  # ImagePositionPatient
    (0x0020, 0x0037),  # ImageOrientationPatient
]


def _physical_slice_normal(iop: np.ndarray, plane: str | None) -> np.ndarray | None:
    """由 IOP 得到单位层法向，并统一为患者坐标主轴的正方向。

    DICOM 患者坐标为 LPS: +X=L, +Y=P, +Z=H。
    因而三个平面固定为 Sagittal +X、Coronal +Y、Axial +Z。
    """
    normal = np.cross(iop[:3], iop[3:6])
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    normal = normal / norm

    plane_axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
    axis = plane_axis.get(plane, int(np.argmax(np.abs(normal))))
    if normal[axis] < 0:
        normal = -normal
    return normal


def sorted_dicom_files(
    cfg: PreprocessCfg,
    study_uid: str,
    series_uid: str,
    plane: str | None,
) -> tuple[list[Path], bool]:
    """按患者物理坐标排序；几何信息不完整时整条回退 InstanceNumber。

    position = dot(ImagePositionPatient,
                   cross(IOP_row, IOP_col))

    不能逐张混用两种排序，否则缺 tag 的单张图可能被插入错误位置。
    """
    files = dicom_files(cfg, study_uid, series_uid)
    if len(files) < 2:
        return files, False

    records = []
    for p in files:
        try:
            ds = pydicom.dcmread(
                str(p), force=True, stop_before_pixels=True,
                specific_tags=_ORDER_TAGS,
            )
            n = getattr(ds, "InstanceNumber", None)
            instance = float(n) if n is not None else float("inf")

            ipp_raw = getattr(ds, "ImagePositionPatient", None)
            iop_raw = getattr(ds, "ImageOrientationPatient", None)
            ipp = (
                np.asarray([float(v) for v in ipp_raw], dtype=np.float64)
                if ipp_raw is not None and len(ipp_raw) >= 3 else None
            )
            iop = (
                np.asarray([float(v) for v in iop_raw], dtype=np.float64)
                if iop_raw is not None and len(iop_raw) >= 6 else None
            )
            records.append((p, instance, ipp, iop))
        except Exception:
            records.append((p, float("inf"), None, None))

    def instance_fallback() -> list[Path]:
        return [
            rec[0] for rec in sorted(records, key=lambda r: (r[1], r[0].name))
        ]

    # 任意一张缺几何信息，就让整个 series 回退，避免混合排序。
    if any(ipp is None or iop is None for _, _, ipp, iop in records):
        return instance_fallback(), False

    normal = _physical_slice_normal(records[0][3], plane)
    if normal is None:
        return instance_fallback(), False

    positioned = []
    for p, instance, ipp, _iop in records:
        position = float(np.dot(ipp, normal))
        if not np.isfinite(position):
            return instance_fallback(), False
        positioned.append((position, instance, p.name, p))

    return [p for _, _, _, p in sorted(positioned)], True


def probe_series(cfg: PreprocessCfg, row) -> dict:
    """读 header 取 PixelSpacing / laterality tag / x 坐标 / 层进方向。

    修正 [A]: x_pos 改为**图像中心** x。

        ImagePositionPatient 给的是左上角第一个体素的坐标。要得到图像中心，
        沿两个面内方向各走半幅:
            center = ipp + iop_row * col_spacing * (Columns-1)/2
                         + iop_col * row_spacing * (Rows-1)/2
        DICOM 约定: ImageOrientationPatient = (行方向余弦, 列方向余弦)
                    PixelSpacing            = [行间距, 列间距]

        x_raw 保留原始 ipp[0]，只在中心算不出来时给矢状位兜底用
        (矢状位 x 就是层轴，本来无偏)。
    """
    study, series = str(row.StudyInstanceUID), str(row.SeriesInstanceUID)
    files = dicom_files(cfg, study, series)

    out = {
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "n_slices": len(files),
        "px": np.nan,
        "laterality": None,
        "x_pos": np.nan,
        "x_raw": np.nan,
        "slice_dir": 1,
    }
    if not files:
        return out

    try:
        ds = pydicom.dcmread(
            str(files[len(files) // 2]), stop_before_pixels=True, force=True
        )
        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None:
            out["px"] = float(spacing[0])

        side = getattr(ds, "ImageLaterality", None) or getattr(ds, "Laterality", None)
        if side:
            side = str(side).strip().upper()
            if side and side[0] in ("L", "R"):
                out["laterality"] = side[0]

        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            ipp_v = np.array([float(v) for v in ipp])
            out["x_raw"] = float(ipp_v[0])
            try:
                iop_m = np.array(
                    [float(v) for v in ds.ImageOrientationPatient]
                ).reshape(2, 3)
                ps = [float(v) for v in ds.PixelSpacing]
                center = (
                    ipp_v
                    + iop_m[0] * ps[1] * (int(ds.Columns) - 1) / 2.0
                    + iop_m[1] * ps[0] * (int(ds.Rows) - 1) / 2.0
                )
                out["x_pos"] = float(center[0])
            except Exception:
                pass

        # 层进方向: 把首尾两张的位置投影到层法向，看 InstanceNumber 递增
        # 时投影是增是减。
        if len(files) >= 3:
            iop = [float(v) for v in ds.ImageOrientationPatient]
            nv = np.cross(iop[:3], iop[3:])
            nv = nv / (np.linalg.norm(nv) + 1e-9)
            proj, inst = [], []
            for f in (files[0], files[-1]):
                d2 = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                proj.append(float(np.dot(
                    [float(v) for v in d2.ImagePositionPatient], nv)))
                inst.append(float(d2.InstanceNumber))
            d_inst = inst[1] - inst[0]
            if d_inst != 0:
                out["slice_dir"] = int(np.sign((proj[1] - proj[0]) * np.sign(d_inst)))
    except Exception:
        pass

    return out


def add_dicom_header_info(cfg: PreprocessCfg, series_df: pd.DataFrame) -> pd.DataFrame:
    rows = list(series_df.itertuples(index=False))
    log.info("读取 %d 个 series 的 DICOM header...", len(rows))

    with ThreadPoolExecutor(max_workers=cfg.header_workers) as pool:
        info = list(pbar(
            pool.map(lambda r: probe_series(cfg, r), rows),
            total=len(rows), desc="DICOM headers", unit="series",
        ))

    merged = series_df.merge(
        pd.DataFrame(info), on=["StudyInstanceUID", "SeriesInstanceUID"], how="left"
    )
    merged["n_slices"] = (
        pd.to_numeric(merged["n_slices"], errors="coerce").fillna(0).astype(int)
    )
    merged["slice_dir"] = (
        pd.to_numeric(merged["slice_dir"], errors="coerce").fillna(1).astype(int)
    )
    return merged[merged["n_slices"] > 0].reset_index(drop=True)


def resolve_laterality(cfg: PreprocessCfg,
                       series_df: pd.DataFrame) -> dict[str, str | None]:
    """study 级的左右判定。

    必须在 study 级做 —— 同一次检查的所有 series 扫的是同一个膝盖，
    逐 series 判断只会引入不必要的噪声。

    修正 [A] 的兜底顺序:
        1. ImageLaterality tag        厂商写死，最可信 (覆盖 50%)
        2. 图像中心 x 的中位数         对所有 plane 无偏
        3. 矢状位原始 ipp[0] 的中位数  中心算不出来时的最后一档

    LPS 约定: 右膝位于负 x。判不出时返回 None，read_slot 不翻转 ——
    翻错比不翻更糟。
    """
    out: dict[str, str | None] = {}
    n_tag = n_ctr = n_sag = n_none = 0

    def side_of(x: float) -> str | None:
        if not np.isfinite(x) or abs(x) < cfg.lat_min_offset_mm:
            return None
        return "R" if x < 0 else "L"

    for study, g in series_df.groupby("StudyInstanceUID"):
        tag = g["laterality"].dropna()
        if len(tag):
            out[str(study)] = str(tag.iloc[0])
            n_tag += 1
            continue

        xs = g["x_pos"].dropna()
        if len(xs):
            s = side_of(float(xs.median()))
            if s is not None:
                out[str(study)] = s
                n_ctr += 1
                continue

        xr = g.loc[g["plane"] == "Sagittal", "x_raw"].dropna()
        if len(xr):
            s = side_of(float(xr.median()))
            if s is not None:
                out[str(study)] = s
                n_sag += 1
                continue

        out[str(study)] = None
        n_none += 1

    total = max(len(out), 1)
    log.info(
        "laterality: tag %d (%.1f%%) | 中心x %d (%.1f%%) | 矢状兜底 %d (%.1f%%) | "
        "判不出 %d (%.1f%%) -> 覆盖 %.1f%%",
        n_tag, 100 * n_tag / total, n_ctr, 100 * n_ctr / total,
        n_sag, 100 * n_sag / total, n_none, 100 * n_none / total,
        100 * (total - n_none) / total,
    )
    return out


def pick_slots(series_df: pd.DataFrame) -> dict[str, dict]:
    """每个 study 为 5 个 slot 各选一条 series（切片数最多者）。"""
    result = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fatsat in SLOTS:
            cand = g[(g["plane"] == plane) & (g["fluid"] == fluid) & (g["fatsat"] == fatsat)]

            # 注: 官方 Fluid_Sensitive 与 Fat_Suppression 数值完全相同，
            # 故此分支实际永不触发。保留以防将来换用恢复的加权判据。
            if len(cand) == 0 and fluid is False:
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]

            if len(cand):
                chosen[name] = (
                    cand.sort_values("n_slices", ascending=False).iloc[0].to_dict()
                )
        result[str(study)] = chosen
    return result


# ============================================================
# 5. SLOT -> uint8 [24, H, W]
# ============================================================
# 处理顺序不可调换:
#   sort -> sample -> crop -> percentile -> clip -> pad -> resize -> uint8 -> flip

def _sample_indices(n: int, k: int) -> np.ndarray:
    """中央80%均匀取k层；不足时居中扩展，全序列不足k才重复。"""
    if n < 1 or k < 1:
        raise ValueError("n 和 k 必须为正整数")
    lo, hi = int(0.10 * (n - 1)), int(0.90 * (n - 1))
    if hi - lo >= k - 1:
        return np.linspace(lo, hi, k).astype(int)

    # 端点直接取整数并居中，全程不做浮点舍入。
    # 注意别用 np.round(): n 为奇数时端点落在 x.5、间距恰为 1.0，
    # 银行家舍入 (round-half-to-even) 会把相邻两个塌成同一个
    # (1.5->2, 2.5->2, 3.5->4 ...)，n=19..27 的奇数反而比 v3 更差。
    span = min(n - 1, k - 1)                 # 需要多宽就放多宽，但不超出边界
    lo = (n - 1 - span) // 2                 # 居中
    return np.linspace(lo, lo + span, k).astype(int)


def _read_volume(files: list[Path], idx: np.ndarray) -> np.ndarray | None:
    """读取指定 index 的切片并应用 rescale；形状不一致或读失败返回 None。"""
    slices = []
    for i in idx:
        try:
            ds = pydicom.dcmread(str(files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            slices.append(a * slope + intercept)
        except Exception:
            return None

    if any(s.shape != slices[0].shape for s in slices):
        return None
    return np.stack(slices)


def _crop_mm(vol: np.ndarray, px, crop_mm: float) -> tuple[np.ndarray, int | None]:
    """按物理尺寸中心裁剪，返回 (裁剪结果, 目标边长)。"""
    try:
        px = float(px)
    except Exception:
        px = float("nan")

    h, w = vol.shape[1:]
    if not (np.isfinite(px) and px > 0):
        s = min(h, w)
        return vol[
            :,
            h // 2 - s // 2:h // 2 + s // 2,
            w // 2 - s // 2:w // 2 + s // 2,
        ], None

    want = int(round(crop_mm / px))
    if want <= 16:
        return vol, None

    cy, cx, half = h // 2, w // 2, want // 2
    return vol[
        :,
        max(0, cy - half):min(h, cy + half),
        max(0, cx - half):min(w, cx + half),
    ], want


def _to_uint8(vol: np.ndarray, img_size: int, pad_to: int | None = None) -> torch.Tensor:
    """1%-99% 归一化 -> clip -> 补零到 pad_to -> 抗锯齿 resize -> uint8。"""
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / max(float(p99 - p1), 1e-6), 0, 1)

    if pad_to is not None and vol.shape[1:] != (pad_to, pad_to):
        k, h, w = vol.shape
        out = np.zeros((k, pad_to, pad_to), dtype=vol.dtype)
        out[
            :,
            (pad_to - h) // 2:(pad_to - h) // 2 + h,
            (pad_to - w) // 2:(pad_to - w) // 2 + w,
        ] = vol
        vol = out

    x = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    x = F.interpolate(
        x, size=(img_size, img_size), mode="bilinear",
        align_corners=False, antialias=True,
    )
    return x.squeeze(0).mul(255).round().clamp(0, 255).to(torch.uint8)


def _canonicalize_side(
    x: torch.Tensor,
    side,
    plane,
    physical_ordered: bool,
    slice_dir: int = 1,
) -> torch.Tensor:
    """把每个膝关节映射到统一朝向。

    冠状 / 轴位: 内外方向在图像平面内 -> 右膝水平镜像。
        (体检确认冠状 8609 / 轴位 5898 条的 IOP 行方向余弦 x 全部 > 0，
         无一例外，故无条件翻转是安全的。)

    矢状位: sorted_dicom_files 已固定按患者 +X 排序。
        左膝: +X 即 medial -> lateral，保持不变。
        右膝: medial -> lateral 为 -X，因此反转层轴。

    side 为 None 时保持 +X 物理顺序，不做无法确认的左右侧反转。
    若几何信息缺失而回退 InstanceNumber，则保留 v4 的 slice_dir 异或规则。
    """
    if plane == "Sagittal":
        if physical_ordered:
            flip = side == "R"
        else:
            flip = (int(slice_dir) > 0) ^ (side == "R")
        return torch.flip(x, dims=[0]) if flip else x

    if side == "R" and plane in ("Coronal", "Axial"):
        return torch.flip(x, dims=[-1])

    return x


def read_slot(cfg: PreprocessCfg, rec: dict) -> np.ndarray | None:
    files, physical_ordered = sorted_dicom_files(
        cfg,
        str(rec["StudyInstanceUID"]),
        str(rec["SeriesInstanceUID"]),
        rec.get("plane"),
    )
    if not files:
        return None

    vol = _read_volume(files, _sample_indices(len(files), cfg.slices_per_slot))
    if vol is None:
        return None

    vol, want = _crop_mm(vol, rec.get("px", np.nan), cfg.crop_mm)
    x = _to_uint8(vol, cfg.img_size, want)
    x = _canonicalize_side(
        x,
        rec.get("laterality"),
        rec.get("plane"),
        physical_ordered,
        rec.get("slice_dir", 1),
    )
    return x.numpy()


# ============================================================
# 6. CACHE 构建与加载
# ============================================================

def partition_study_ids(cfg: PreprocessCfg) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(cfg.train_csv, dtype={"StudyInstanceUID": str})
    if df["StudyInstanceUID"].isna().any():
        raise ValueError("train.csv 存在空 StudyInstanceUID")
    all_ids = df["StudyInstanceUID"].astype(str).drop_duplicates().sort_values().to_numpy(dtype=str)
    if len(all_ids) < cfg.n_parts:
        raise ValueError("study 数少于分包数")
    return all_ids, np.array_split(all_ids, cfg.n_parts)[cfg.part_id]


def cache_manifest(cfg: PreprocessCfg) -> dict:
    all_ids, ids = partition_study_ids(cfg)
    return {
        "format_version": "physicalsort-central80-multipart-v1",
        "part_id": cfg.part_id, "n_parts": cfg.n_parts,
        "total_studies": len(all_ids), "part_studies": len(ids),
        "all_ids_sha256": hashlib.sha256("\n".join(all_ids).encode()).hexdigest(),
        "part_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "shape": [len(ids), cfg.n_slots, cfg.slices_per_slot, cfg.img_size, cfg.img_size],
        "dtype": "uint8", "central_fraction": 0.8,
        "crop_mm": cfg.crop_mm, "lat_min_offset_mm": cfg.lat_min_offset_mm,
        "slots": [x[0] for x in SLOTS],
    }


def build_cache(cfg: PreprocessCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_ids, study_ids = partition_study_ids(cfg)
    log.info("part%d/%d: %d / %d studies", cfg.part_id, cfg.n_parts, len(study_ids), len(all_ids))
    series_df = load_series_csv(cfg)
    series_df = series_df[series_df["StudyInstanceUID"].isin(study_ids)].reset_index(drop=True)
    series_df = add_dicom_header_info(cfg, series_df)
    side_map = resolve_laterality(cfg, series_df)          # 修正 [A]
    slot_map = pick_slots(series_df)
    slot_index = {name: i for i, (name, *_) in enumerate(SLOTS)}

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    (cfg.cache_dir / "manifest.json").unlink(missing_ok=True)
    shape = (len(study_ids), cfg.n_slots, cfg.slices_per_slot, cfg.img_size, cfg.img_size)
    cache = open_memmap(cfg.cache_path, mode="w+", dtype=np.uint8, shape=shape)
    cache[:] = 0  # 缺失/读取失败 slot 显式保持零
    mask = np.zeros((len(study_ids), cfg.n_slots), dtype=np.float32)
    log.info("cache shape %s (%.2f GiB, memmap 增量写)", shape, cache.nbytes / 1024 ** 3)

    # laterality 用 study 级结果覆盖 series 级的 tag
    jobs = []
    for i, sid in enumerate(study_ids):
        for name, rec in slot_map.get(sid, {}).items():
            rec = dict(rec)
            rec["laterality"] = side_map.get(sid)
            jobs.append((i, slot_index[name], rec))
    log.info("准备读取 %d 个 selected series...", len(jobs))

    n_sag_flip = sum(
        1 for _, _, r in jobs
        if r.get("plane") == "Sagittal"
        and r.get("laterality") == "R"
    )
    n_sag = sum(1 for _, _, r in jobs if r.get("plane") == "Sagittal")
    log.info("矢状位 %d 条，其中需翻转层序 %d 条 (%.1f%%)",
             n_sag, n_sag_flip, 100 * n_sag_flip / max(n_sag, 1))

    unique_counts = [len(np.unique(_sample_indices(int(r["n_slices"]), cfg.slices_per_slot)))
                     for _, _, r in jobs if int(r.get("n_slices", 0)) > 0]
    if unique_counts:
        log.info("取样: %d/%d 条有重复索引，平均唯一层 %.2f / %d",
                 sum(u < cfg.slices_per_slot for u in unique_counts), len(unique_counts),
                 float(np.mean(unique_counts)), cfg.slices_per_slot)

    def worker(job):
        i, k, rec = job
        return i, k, read_slot(cfg, rec)

    with ThreadPoolExecutor(max_workers=cfg.pixel_workers) as pool:
        for i, k, x in pbar(
            pool.map(worker, jobs), total=len(jobs), desc="DICOM -> cache", unit="series"
        ):
            if x is not None:
                cache[i, k] = x
                mask[i, k] = 1.0

    cache.flush()
    np.save(cfg.mask_path, mask)
    np.save(cfg.ids_path, study_ids)
    log.info("各 slot 填充率: %s",
             {name: f"{mask[:, i].mean():.1%}" for i, (name, *_) in enumerate(SLOTS)})

    # 输出侧自检: 有 mask 但内容全零 = read_slot 出了问题
    filled = mask > 0.5
    zero = np.zeros_like(filled)
    for i in range(len(study_ids)):
        for k in range(cfg.n_slots):
            if filled[i, k] and cache[i, k].max() == 0:
                zero[i, k] = True
    n_zero = int(zero.sum())
    log.info("自检: mask=1 但内容全零的 slot %d 个 (应为 0)", n_zero)
    log.info("自检: 平均亮度 首层 %.1f / 中层 %.1f / 末层 %.1f (中层应最亮)",
             cache[:, :, 0][filled].mean(),
             cache[:, :, cfg.slices_per_slot // 2][filled].mean(),
             cache[:, :, -1][filled].mean())

    # 最后写完成标记；中断后不会复用半成品。
    manifest_path = cfg.cache_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache_manifest(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    return cache, mask, study_ids


def load_or_build_cache(cfg: PreprocessCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = cfg.cache_dir / "manifest.json"
    if all(p.exists() for p in (cfg.cache_path, cfg.mask_path, cfg.ids_path, manifest_path)):
        expected = cache_manifest(cfg)
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("已有缓存的配置或 study 划分不一致，请更换输出目录。")
        cache = np.load(cfg.cache_path, mmap_mode="r")
        mask = np.load(cfg.mask_path)
        ids = np.load(cfg.ids_path, allow_pickle=False).astype(str)
        _, expected_ids = partition_study_ids(cfg)
        if (list(cache.shape) != expected["shape"] or cache.dtype != np.uint8
                or mask.shape != (len(ids), cfg.n_slots)
                or not np.array_equal(ids, expected_ids)):
            raise RuntimeError("已有缓存形状、类型或 ID 不一致，请重新生成。")
        log.info("复用已完成的 part%d cache。", cfg.part_id)
        return cache, mask, ids
    log.info("没有完整缓存，从 DICOM 构建当前分包。")
    return build_cache(cfg)


# ============================================================
# 7. 标签加载（纯数据，不含训练逻辑）
# ============================================================

def load_labels(
    cfg: PreprocessCfg,
    study_ids: np.ndarray,
    scored_path: Path,
    gold_weight: float = 3.0,
) -> dict:
    """
    返回 dict:
      study_ids  [n] str        — 有标签的 study（子集）
      cache_rows [n] int64      — 对应 cache 第 0 维行号
      y          [n, 12] float32
      weight     [n, 12] float32
      is_gold    [n] bool
      groups     [n] str        — GroupKFold 分组键
    """
    # ---- LLM soft labels ----
    llm = pd.read_csv(scored_path, dtype={"StudyInstanceUID": str})
    missing = [c for c in ["StudyInstanceUID", *TARGETS] if c not in llm.columns]
    if missing:
        raise ValueError(f"train_scored.csv 缺少 columns: {missing}")

    llm["StudyInstanceUID"] = llm["StudyInstanceUID"].astype(str)
    for t in TARGETS:
        llm[t] = pd.to_numeric(llm[t], errors="coerce")
    llm = llm.drop_duplicates("StudyInstanceUID", keep="first").set_index("StudyInstanceUID")

    # ---- official gold ----
    train_df = pd.read_csv(cfg.train_csv, dtype={"StudyInstanceUID": str})
    train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)

    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    log.info("official fully-annotated studies: %d", len(gold))

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
    per_study_w = np.where(is_gold, gold_weight, 1.0).astype(np.float32)
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
        "supervised: %d | gold: %d (w=%.1f) | LLM: %d (w=1.0)",
        len(ids), n_gold, gold_weight, len(ids) - n_gold,
    )

    return {
        "study_ids": ids,
        "cache_rows": cache_rows,
        "y": y,
        "weight": weight,
        "is_gold": is_gold,
        "groups": groups,
    }


# ============================================================
# 8. MAIN
# ============================================================

def main(cfg: PreprocessCfg | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[TqdmLoggingHandler()],
        force=True,
    )
    cfg = cfg or PreprocessCfg()

    for p in (cfg.train_dicom_dir, cfg.train_csv, cfg.train_series_csv):
        if not p.exists():
            raise FileNotFoundError(f"找不到: {p}")

    log.info("=" * 64)
    log.info("Knee MRI 预处理 (中央80%% + %d层，%d分包，物理排序)",
             cfg.slices_per_slot, cfg.n_parts)
    log.info("=" * 64)
    log.info("cache_dir %s", cfg.cache_dir)
    log.info(
        "img %d | crop %.0fmm | slots %d | slices/slot %d | lat_min_offset %.0fmm",
        cfg.img_size, cfg.crop_mm, cfg.n_slots, cfg.slices_per_slot,
        cfg.lat_min_offset_mm,
    )
    log.info("=" * 64)

    cache, mask, study_ids = load_or_build_cache(cfg)
    log.info("cache 就绪: %d studies, shape %s", len(study_ids), cache.shape)


if __name__ == "__main__":
    main()
