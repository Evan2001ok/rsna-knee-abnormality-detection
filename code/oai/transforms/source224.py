# Original pixel functions only; loaded via AST by OAI cache scripts.

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
