"""Load complete studies as fixed sets of views and neighbouring slice groups."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .constants import PLANES, RAPTOR_VARIANT_NAMES, SERIES_ID, STUDY_ID, TARGETS


def make_dataset(manifest, studies, config, training=False, seed=42):
    """Select the checkpoint's recorded image recipe without changing old runs."""
    if config.get("recipe", "standard") == "raptor":
        from .raptor_data import RaptorDataset

        return RaptorDataset(manifest, studies, config, training=training, seed=seed)
    return KneeDataset(manifest, studies, config, training=training, seed=seed)


def validate_dataset_manifest(manifest, config, study_ids=None, verify_checksums=False):
    if config.get("recipe", "standard") == "raptor":
        from .raptor_data import validate_raptor_manifest

        return validate_raptor_manifest(
            manifest, expected_size=config["cache_size"], study_ids=study_ids, verify_checksums=verify_checksums,
            version=config.get("raptor_variant", RAPTOR_VARIANT_NAMES[0]),
        )
    from .prepare import validate_manifest

    return validate_manifest(
        manifest, expected_size=config["cache_size"], study_ids=study_ids, verify_checksums=verify_checksums,
    )


def dataset_preprocess_version(config):
    if config.get("recipe", "standard") == "raptor":
        from .raptor_data import raptor_variant

        version = config.get("raptor_variant", RAPTOR_VARIANT_NAMES[0])
        raptor_variant(version)
        return version
    from .imaging import PREPROCESS_VERSION

    return PREPROCESS_VERSION


def _flag(value: object) -> float:
    """Keep missing sequence information distinct from a known negative flag."""
    if pd.isna(value):
        return -1.0
    value = str(value).strip().lower()
    if value in {"1", "1.0", "true", "yes", "y"}:
        return 1.0
    if value in {"0", "0.0", "false", "no", "n"}:
        return 0.0
    return -1.0


def _plane(value: object) -> str | None:
    name = str(value).strip().lower()
    aliases = {"sag": "Sagittal", "cor": "Coronal", "ax": "Axial", "transverse": "Axial"}
    return aliases.get(name, next((p for p in PLANES if p.lower() == name), None))


class KneeDataset(Dataset):
    """Use one target per study; absent views and unknown labels have explicit masks.

    The manifest points to uint16 ``[depth, height, width]`` NumPy volumes in
    physical slice order. Image preparation, including intensity scaling, happens
    once before training. All random choices depend only on seed, study, and epoch.
    """

    def __init__(
        self,
        manifest: str | Path,
        studies: pd.DataFrame,
        config: dict,
        training: bool = False,
        seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.training = bool(training)
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.image_size = int(config.get("image_size", 224))
        self.slices_per_series = int(config.get("slices_per_series", 8))
        self.series_per_plane = int(config.get("series_per_plane", 1))
        self.neighbor_stride = int(config.get("neighbor_stride", 1))
        for name in ("image_size", "slices_per_series", "series_per_plane", "neighbor_stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        augmentation = config.get("augmentation", {})
        if not isinstance(augmentation, dict):
            raise ValueError("augmentation must be a dictionary")
        self.augmentation = {
            "enabled": True,
            "rotation_deg": 7.0,
            "translate": 0.04,
            "scale": 0.08,
            "contrast": 0.15,
            "brightness": 0.05,
            "noise_std": 0.015,
            **augmentation,
        }
        for name in ("rotation_deg", "translate", "scale", "contrast", "brightness", "noise_std"):
            value = float(self.augmentation[name])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"augmentation.{name} must be finite and non-negative")
            self.augmentation[name] = value
        if self.augmentation["scale"] >= 1 or self.augmentation["contrast"] >= 1:
            raise ValueError("augmentation scale and contrast must be smaller than one")

        # DataLoader workers can remain alive between epochs without keeping an
        # obsolete copy of the epoch number.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        if STUDY_ID not in studies:
            raise ValueError(f"studies must contain {STUDY_ID}")
        self.studies = studies.reset_index(drop=True).copy()
        if self.studies[STUDY_ID].isna().any():
            raise ValueError("Study IDs cannot be missing")
        self.studies[STUDY_ID] = self.studies[STUDY_ID].astype(str)
        if self.studies[STUDY_ID].duplicated().any():
            raise ValueError("studies must contain each study only once")

        labels = self.studies.reindex(columns=TARGETS).apply(pd.to_numeric, errors="raise")
        values = labels.to_numpy(dtype=np.float32, na_value=np.nan)
        known = ~np.isnan(values)
        if not np.isfinite(values[known]).all() or ((values[known] < 0) | (values[known] > 1)).any():
            raise ValueError("Known targets must be probabilities between zero and one")
        self.targets = np.where(known, values, 0.0).astype(np.float32)
        self.target_masks = known

        table = pd.read_csv(self.manifest_path, dtype={STUDY_ID: str, SERIES_ID: str})
        required = {STUDY_ID, SERIES_ID, "Anatomical_Plane", "cache_path", "num_slices"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        if table[[STUDY_ID, SERIES_ID, "cache_path"]].isna().any().any():
            raise ValueError("Manifest study IDs, series IDs, and cache paths must be present")
        if table.duplicated([STUDY_ID, SERIES_ID]).any():
            raise ValueError("Manifest contains a repeated study/series pair")
        self._series: dict[str, dict[str, list[dict]]] = {}
        study_ids = set(self.studies[STUDY_ID])
        for row in table.to_dict(orient="records"):
            study_id = row[STUDY_ID]
            if study_id not in study_ids:
                continue
            plane = _plane(row["Anatomical_Plane"])
            if plane is None:
                continue
            count = float(row["num_slices"])
            if not np.isfinite(count) or count < 1 or count != int(count):
                raise ValueError(f"Invalid num_slices for {row[SERIES_ID]}")
            row["num_slices"] = int(count)
            row["fluid"] = _flag(row.get("Fluid_Sensitive"))
            row["fat"] = _flag(row.get("Fat_Suppression"))
            # Flags take priority; depth breaks ties. Training may choose another
            # series with the same useful sequence flags to vary its coverage.
            row["quality"] = (
                2.0 * (row["fluid"] == 1)
                + 1.0 * (row["fat"] == 1)
                + 0.25 * min(row["num_slices"], 96) / 96
            )
            path = Path(row["cache_path"])
            row["path"] = path if path.is_absolute() else self.manifest_path.parent / path
            self._series.setdefault(study_id, {}).setdefault(plane, []).append(row)
        absent = [sid for sid in self.studies[STUDY_ID] if sid not in self._series]
        if absent:
            raise ValueError(f"No usable MRI series for {len(absent)} studies, including {absent[:3]}")
        for planes in self._series.values():
            for rows in planes.values():
                rows.sort(key=lambda row: (-row["quality"], str(row[SERIES_ID])))

    def __len__(self) -> int:
        return len(self.studies)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch.fill_(int(epoch))

    def _rng(self, study_id: str) -> np.random.Generator:
        digest = hashlib.blake2b(study_id.encode("utf-8"), digest_size=8).digest()
        identity = int.from_bytes(digest, "little")
        epoch = int(self._epoch.item()) if self.training else 0
        return np.random.default_rng(np.random.SeedSequence([
            self.seed, epoch, identity & 0xFFFFFFFF, identity >> 32,
        ]))

    def _select_series(self, rows: list[dict], rng: np.random.Generator) -> list[dict]:
        if not self.training:
            return rows[: self.series_per_plane]
        remaining = list(rows)
        selected = []
        while remaining and len(selected) < self.series_per_plane:
            best = remaining[0]["quality"]
            candidates = [i for i, row in enumerate(remaining) if row["quality"] >= best - 0.25]
            weights = np.sqrt([min(remaining[i]["num_slices"], 96) for i in candidates])
            chosen = int(rng.choice(candidates, p=weights / weights.sum()))
            selected.append(remaining.pop(chosen))
        return selected

    def _centers(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if self.training:
            edges = np.linspace(0, count, self.slices_per_series + 1)
            return np.minimum(rng.uniform(edges[:-1], edges[1:]).astype(np.int64), count - 1)
        if self.slices_per_series == 1:
            return np.array([(count - 1) // 2], dtype=np.int64)
        return np.rint(np.linspace(0, count - 1, self.slices_per_series)).astype(np.int64)

    def _augment(self, groups: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        a = self.augmentation
        if not self.training or not a["enabled"]:
            return groups
        shift = a["translate"] * self.image_size
        # A single affine transform is applied to every slice in this series.
        groups = TF.affine(
            groups,
            angle=float(rng.uniform(-a["rotation_deg"], a["rotation_deg"])),
            translate=[int(round(v)) for v in rng.uniform(-shift, shift, size=2)],
            scale=float(rng.uniform(1 - a["scale"], 1 + a["scale"])),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        contrast = float(rng.uniform(1 - a["contrast"], 1 + a["contrast"]))
        brightness = float(rng.uniform(-a["brightness"], a["brightness"]))
        groups = (groups - 0.5) * contrast + 0.5 + brightness
        if a["noise_std"]:
            noise = rng.normal(0, a["noise_std"], size=tuple(groups.shape)).astype(np.float32)
            groups = groups + torch.from_numpy(noise)
        return groups.clamp_(0, 1)

    def __getitem__(self, index: int) -> dict:
        study_id = self.studies.iloc[index][STUDY_ID]
        rng = self._rng(study_id)
        slots = len(PLANES) * self.series_per_plane
        images = torch.zeros(slots, self.slices_per_series, 3, self.image_size, self.image_size)
        mask = torch.zeros(slots, dtype=torch.bool)
        metadata = torch.zeros(slots, 5, dtype=torch.float32)
        metadata[:, 3:] = -1
        positions = torch.zeros(slots, self.slices_per_series, dtype=torch.float32)
        for plane_index, plane in enumerate(PLANES):
            rows = self._series[study_id].get(plane, [])
            for series_index, row in enumerate(self._select_series(rows, rng)):
                slot = plane_index * self.series_per_plane + series_index
                volume = np.load(row["path"], mmap_mode="r", allow_pickle=False)
                if volume.ndim != 3 or volume.dtype != np.uint16 or min(volume.shape) < 1:
                    raise ValueError(f"Expected a non-empty uint16 [depth, height, width] volume: {row['path']}")
                if volume.shape[0] != row["num_slices"]:
                    raise ValueError(f"Cached depth differs from manifest: {row['path']}")
                centers = self._centers(volume.shape[0], rng)
                neighbors = np.clip(
                    centers[:, None] + np.array([-self.neighbor_stride, 0, self.neighbor_stride]),
                    0, volume.shape[0] - 1,
                )
                pixels = np.asarray(volume[neighbors], dtype=np.float32) / 65535.0
                groups = torch.from_numpy(pixels)
                if tuple(groups.shape[-2:]) != (self.image_size, self.image_size):
                    groups = TF.resize(
                        groups, [self.image_size, self.image_size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True,
                    )
                images[slot] = self._augment(groups, rng)
                mask[slot] = True
                metadata[slot, plane_index] = 1
                metadata[slot, 3:] = torch.tensor([row["fluid"], row["fat"]])
                positions[slot] = torch.from_numpy(centers.astype(np.float32) / max(volume.shape[0] - 1, 1))
        return {
            "images": images,
            "series_mask": mask,
            "metadata": metadata,
            "positions": positions,
            "targets": torch.from_numpy(self.targets[index].copy()),
            "target_mask": torch.from_numpy(self.target_masks[index].copy()),
            "study_id": study_id,
        }
