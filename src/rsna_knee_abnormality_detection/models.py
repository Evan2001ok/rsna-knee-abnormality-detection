"""A shared image encoder that predicts twelve findings for a whole study."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50

from .constants import TARGETS, WINDOW_FEATURES

RAPTOR_BACKBONE = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
# OrthoFoundation-L: a DINOv3 ViT-L/16 further pretrained on knee X-ray and MRI images. Its public weights (MIT
# license) come from https://github.com/ytrsk/OrthoFoundation; our download script is not included here.
ORTHOFOUNDATION_BACKBONE = "orthofoundation_l"
ORTHOFOUNDATION_TIMM_NAME = "vit_large_patch16_dinov3"
# DINOv3 ViTs (Meta, natural-image pretraining) with timm's downloadable weights.
DINOV3_BACKBONES = {"dinov3_small": "vit_small_patch16_dinov3.lvd1689m", "dinov3_base": "vit_base_patch16_dinov3.lvd1689m"}
# Convolutional timm encoders take any image size, so they need no input_size.
CONVNEXT_BACKBONES = {"convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k"}
# EfficientNetV2-S (ImageNet-21k pretrained, fine-tuned on ImageNet-1k, evaluated at 384 px); convolutional as above.
EFFICIENTNET_BACKBONES = {"efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k"}
TIMM_BACKBONES = {RAPTOR_BACKBONE, ORTHOFOUNDATION_BACKBONE, *DINOV3_BACKBONES, *CONVNEXT_BACKBONES,
                  *EFFICIENTNET_BACKBONES}


def _dinov3_filter(state: dict, model: nn.Module) -> dict:
    """Convert official DINOv3 key names to timm's, as timm does for Meta's own weights."""
    from timm.models.eva import checkpoint_filter_fn

    return checkpoint_filter_fn(state, model)


def load_orthofoundation_weights(encoder: nn.Module, path) -> None:
    """Load the released backbone (keys prefixed "backbone.") strictly: any missing or extra key is an error."""
    if not path:
        raise ValueError("orthofoundation_l needs model.pretrained_checkpoint when pretrained weights are requested")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = {key.removeprefix("backbone."): value for key, value in state.items()}
    encoder.load_state_dict(_dinov3_filter(state, encoder), strict=True)
# How many leading window descriptors each mode uses (see constants.WINDOW_FEATURES).
WINDOW_FEATURE_MODES = {"none": 0, "types": 5, "types_and_positions": len(WINDOW_FEATURES)}


class ClassAttentionHead(nn.Module):
    """The public Raptor head, with one slice attention distribution per finding.

    This follows ``RaptorClassifier.head`` in the public trainer, version
    347608187. The only addition is a mask for absent series in our study batches.
    Each finding has its own attention output and final classifier vector.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.att = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(256, len(TARGETS)),
        )
        self.clsW = nn.Parameter(torch.zeros(len(TARGETS), feature_dim))
        self.clsb = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim == 4:
            return self._forward_per_finding(features, mask, extra)
        if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Class attention needs features [batch, groups, channels] and a boolean group mask")
        if not mask.any(dim=1).all():
            raise ValueError("Every study needs at least one real slice group")
        # Clear masked values before normalization as well as excluding their
        # attention scores, so even nonfinite padding cannot affect a result.
        h = self.norm(features.masked_fill(~mask.unsqueeze(-1), 0))
        if extra is not None:
            # Per-window descriptors shift both the attention and the pooled features.
            h = h + extra.masked_fill(~mask.unsqueeze(-1), 0)
        scores = self.att(h).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def _forward_per_finding(self, features: torch.Tensor, mask: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        """Same head when each window has one feature vector per finding (in-image attention, Oct 15, 2026):
        finding n's attention across windows and its pooled vector use only that finding's window features. With
        identical vectors for every finding this equals the ordinary forward."""
        if features.shape[2] != len(TARGETS) or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Per-finding class attention needs features [batch, groups, findings, channels] and a group mask")
        if not mask.any(dim=1).all():
            raise ValueError("Every study needs at least one real slice group")
        keep = mask[:, :, None, None]
        h = self.norm(features.masked_fill(~keep, 0))
        if extra is not None:
            h = h + extra.masked_fill(~mask.unsqueeze(-1), 0).unsqueeze(2)
        scores = torch.diagonal(self.att(h), dim1=-2, dim2=-1).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bkn,bknf->bnf", attention, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


class StudyModel(nn.Module):
    """Encode neighbouring slice groups, then combine series into one prediction.

    Padded series are removed before the encoder, including during training, so
    they neither waste encoder memory nor affect its batch-normalization values.
    """

    def __init__(self, config: dict, pretrained: bool | None = None) -> None:
        super().__init__()
        name = config.get("backbone", "resnet34")
        pretrained = bool(config.get("pretrained", True)) if pretrained is None else bool(pretrained)
        choices = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
        }
        if name not in {*choices, *TIMM_BACKBONES}:
            raise ValueError(f"Unsupported backbone {name!r}; choose resnet18, resnet34, {RAPTOR_BACKBONE}, "
                             f"{ORTHOFOUNDATION_BACKBONE}, resnet50, a DINOv3 name ({sorted(DINOV3_BACKBONES)}) "
                             f", a ConvNeXt name ({sorted(CONVNEXT_BACKBONES)}) or an EfficientNet name "
                             f"({sorted(EFFICIENTNET_BACKBONES)})")
        self.pooling = config.get("pooling", "mean")
        if self.pooling not in {"mean", "max", "attention", "class_attention"}:
            raise ValueError("pooling must be mean, max, attention, or class_attention")
        self.encoder_chunk_size = int(config.get("encoder_chunk_size", 32))
        if self.encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")
        # Inference only (Oct 16, 2026): pad the last encoder chunk to the full chunk size, so a compiled encoder sees one
        # input shape. Windows are encoded independently, so the real windows' outputs do not change.
        self.pad_last_chunk = False
        self.freeze_encoder = bool(config.get("freeze_encoder", False))
        is_class_attention = self.pooling == "class_attention"
        self.use_metadata = bool(config.get("use_metadata", not is_class_attention))
        self.use_positions = bool(config.get("use_positions", not is_class_attention))
        if is_class_attention and (self.use_metadata or self.use_positions):
            raise ValueError("class_attention requires use_metadata=False and use_positions=False")
        self.grad_checkpointing = bool(config.get("grad_checkpointing", False))
        self.input_size = None
        if name == RAPTOR_BACKBONE:
            self.input_size = config.get("input_size", 384)
            if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size < 1:
                raise ValueError("CoAtNet input_size must be a positive integer")
            # ResNet-only use remains independent of timm, and checkpoint reloads
            # explicitly construct this architecture without fetching weights.
            import timm

            self.encoder = timm.create_model(
                name, pretrained=pretrained, num_classes=0, in_chans=3,
                global_pool="avg", img_size=self.input_size,
            )
            self.feature_dim = self.encoder.num_features
        elif name in DINOV3_BACKBONES:
            self.input_size = config.get("input_size", 224)
            if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size < 16 \
                    or self.input_size % 16:
                raise ValueError("DINOv3 input_size must be a positive multiple of 16")
            import timm

            self.encoder = timm.create_model(DINOV3_BACKBONES[name], pretrained=pretrained, num_classes=0,
                                             global_pool="token", img_size=self.input_size)
            self.feature_dim = self.encoder.num_features
        elif name in CONVNEXT_BACKBONES or name in EFFICIENTNET_BACKBONES:
            import timm

            self.encoder = timm.create_model({**CONVNEXT_BACKBONES, **EFFICIENTNET_BACKBONES}[name],
                                             pretrained=pretrained, num_classes=0, global_pool="avg")
            self.feature_dim = self.encoder.num_features
        elif name == ORTHOFOUNDATION_BACKBONE:
            self.input_size = config.get("input_size", 224)
            if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size < 16 \
                    or self.input_size % 16:
                raise ValueError("OrthoFoundation input_size must be a positive multiple of 16")
            import timm

            # Built empty, then loaded from the local release; trained checkpoints reload without it.
            self.encoder = timm.create_model(ORTHOFOUNDATION_TIMM_NAME, pretrained=False, num_classes=0,
                                             global_pool="token", img_size=self.input_size)
            if pretrained:
                load_orthofoundation_weights(self.encoder, config.get("pretrained_checkpoint"))
            self.feature_dim = self.encoder.num_features
        else:
            builder, weights = choices[name]
            self.encoder = builder(weights=weights if pretrained else None)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        if self.grad_checkpointing:
            if not callable(getattr(self.encoder, "set_grad_checkpointing", None)):
                raise ValueError(f"Gradient checkpointing is not supported for {name}")
            self.encoder.set_grad_checkpointing(True)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if is_class_attention:
            self.head = ClassAttentionHead(self.feature_dim, dropout=float(config.get("dropout", 0.2)))
        else:
            self.metadata_embedding = nn.Sequential(
                nn.Linear(5, self.feature_dim), nn.Tanh(),
            )
            # Position changes how image features are weighted, so mean pooling can
            # still distinguish the same feature near different ends of a series.
            self.position_modulation = nn.Sequential(
                nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 2 * self.feature_dim),
            )
            nn.init.zeros_(self.position_modulation[-1].weight)
            nn.init.zeros_(self.position_modulation[-1].bias)
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(float(config.get("dropout", 0.2))),
                nn.Linear(self.feature_dim, len(TARGETS)),
            )
        if self.pooling == "attention":
            # Keep shared initial weights and the subsequent random stream
            # identical when comparing pooling methods with the same seed.
            with torch.random.fork_rng(devices=[]):
                self.slice_attention = self._attention()
                self.series_attention = self._attention()
        self.window_feature_mode = config.get("window_features", "none")
        if self.window_feature_mode not in WINDOW_FEATURE_MODES:
            raise ValueError(f"window_features must be one of {sorted(WINDOW_FEATURE_MODES)}")
        if self.window_feature_mode != "none":
            if not is_class_attention:
                raise ValueError("window_features require class_attention pooling")
            # Zero initial weights start from the same predictions as a model
            # without descriptors; fork_rng keeps every other random draw unchanged.
            with torch.random.fork_rng(devices=[]):
                self.window_feature_projection = nn.Linear(len(WINDOW_FEATURES), self.feature_dim)
            nn.init.zeros_(self.window_feature_projection.weight)
            nn.init.zeros_(self.window_feature_projection.bias)
            used = torch.zeros(len(WINDOW_FEATURES))
            used[:WINDOW_FEATURE_MODES[self.window_feature_mode]] = 1
            self.register_buffer("window_feature_mask", used)
        self.spatial_attention = bool(config.get("spatial_attention", False))
        if self.spatial_attention:
            if name != RAPTOR_BACKBONE or not is_class_attention:
                raise ValueError("spatial_attention needs the CoAtNet backbone and class_attention pooling")
            # Per-finding attention over the final feature map inside each window (the teammates' idea, Oct 15,
            # 2026). The last layer starts at zero, so every finding starts from the plain average over positions,
            # which is exactly what the encoder's own head computes; fork_rng keeps every other random draw unchanged.
            with torch.random.fork_rng(devices=[]):
                self.spatial_scores = nn.Sequential(
                    nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, len(TARGETS)),
                )
            nn.init.zeros_(self.spatial_scores[-1].weight)
            nn.init.zeros_(self.spatial_scores[-1].bias)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def _attention(self) -> nn.Sequential:
        return nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def train(self, mode: bool = True) -> StudyModel:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        chunks = []
        for chunk in images.split(self.encoder_chunk_size):
            real = chunk.shape[0]
            if self.pad_last_chunk and real < self.encoder_chunk_size:
                chunk = torch.cat([chunk, chunk.new_zeros((self.encoder_chunk_size - real, *chunk.shape[1:]))])
            normalized = (chunk - self.image_mean) / self.image_std
            if self.spatial_attention:
                if self.freeze_encoder:
                    with torch.no_grad():
                        maps = self.encoder.forward_features(normalized)
                else:
                    maps = self.encoder.forward_features(normalized)
                tokens = maps.flatten(2).transpose(1, 2)  # [windows, positions, channels]
                weights = torch.softmax(self.spatial_scores(tokens).float(), dim=1).to(tokens.dtype)
                encoded = torch.einsum("npt,npf->ntf", weights, tokens)  # [windows, findings, channels]
            elif self.freeze_encoder:
                with torch.no_grad():
                    encoded = self.encoder(normalized)
            else:
                encoded = self.encoder(normalized)
            chunks.append(encoded[:real])
        return torch.cat(chunks, dim=0)

    def _pool_slices(self, features: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return features.mean(dim=1)
        if self.pooling == "max":
            return features.amax(dim=1)
        weights = torch.softmax(self.slice_attention(features).float(), dim=1).to(features.dtype)
        return (features * weights).sum(dim=1)

    def _pool_series(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1)
        if self.pooling == "mean":
            return (features * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
        if self.pooling == "max":
            return features.masked_fill(~expanded, float("-inf")).amax(dim=1)
        scores = self.series_attention(features).float().masked_fill(~expanded, float("-inf"))
        weights = torch.softmax(scores, dim=1).to(features.dtype)
        return (features * weights).sum(dim=1)

    def forward(
        self,
        images: torch.Tensor,
        series_mask: torch.Tensor,
        metadata: torch.Tensor,
        positions: torch.Tensor | None = None,
        window_features: torch.Tensor | None = None,
        window_slots: torch.Tensor | None = None,
        return_series: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Study logits [batch, findings]. With return_series (class_attention only), also each series' own logits
        through the same head from its windows alone, [rows, findings], and each row's study index [rows]."""
        if return_series and self.pooling != "class_attention":
            raise ValueError("Per-series logits need class_attention pooling")
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError("images must have shape [batch, series, groups, 3, height, width]")
        batch, series, groups = images.shape[:3]
        if min(batch, series, groups, images.shape[-2], images.shape[-1]) < 1:
            raise ValueError("Image dimensions cannot be empty")
        if self.input_size is not None and images.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"CoAtNet expects images with height and width {self.input_size}")
        if series_mask.shape != (batch, series) or metadata.shape != (batch, series, 5):
            raise ValueError("series_mask and metadata shapes must match the image batch")
        if series_mask.dtype != torch.bool:
            raise ValueError("series_mask must be boolean")
        if not series_mask.any(dim=1).all():
            raise ValueError("Every study needs at least one real series")
        if positions is not None and positions.shape != (batch, series, groups):
            raise ValueError("positions must have shape [batch, series, groups]")

        real_images = images[series_mask]
        real_count = real_images.shape[0]
        encoded = self._encode(real_images.flatten(0, 1))
        rest = encoded.shape[1:]  # (channels,) or, with spatial attention, (findings, channels)
        features = encoded.reshape(real_count, groups, *rest)
        if self.pooling == "class_attention":
            indices = series_mask.flatten().nonzero(as_tuple=False).flatten()
            padded = features.new_zeros(batch * series, groups, *rest)
            padded = padded.index_copy(0, indices, features).reshape(batch, series * groups, *rest)
            group_mask = series_mask.unsqueeze(-1).expand(batch, series, groups).reshape(batch, series * groups)
            extra = None
            if self.window_feature_mode != "none":
                if window_features is None or window_features.shape != (batch, series, groups, len(WINDOW_FEATURES)):
                    raise ValueError(f"This model needs window_features [batch, series, groups, {len(WINDOW_FEATURES)}]")
                extra = self.window_feature_projection(window_features.float() * self.window_feature_mask)
                extra = extra.reshape(batch, series * groups, self.feature_dim)
            logits = self.head(padded, group_mask, extra)
            if not return_series:
                return logits
            if window_slots is None or window_slots.shape != (batch, series, groups):
                raise ValueError("Per-series logits need window_slots [batch, series, groups]")
            slots = window_slots.reshape(batch, series * groups).to(padded.device)
            series_logits, rows = [], []
            for slot in torch.unique(slots[group_mask]).tolist():
                windows = group_mask & (slots == slot)
                present = windows.any(dim=1)  # the head needs at least one window per row
                series_logits.append(self.head(padded[present], windows[present],
                                               None if extra is None else extra[present]))
                rows.append(present.nonzero(as_tuple=False).flatten())
            return logits, torch.cat(series_logits), torch.cat(rows)
        if self.use_positions and positions is not None:
            p = positions[series_mask].to(features.dtype).unsqueeze(-1)
            scale, offset = self.position_modulation(p).chunk(2, dim=-1)
            features = features * (1 + 0.1 * scale.tanh()) + 0.1 * offset
        series_features = self._pool_slices(features)
        if self.use_metadata:
            real_metadata = metadata[series_mask].to(series_features.dtype)
            series_features = series_features + 0.1 * self.metadata_embedding(real_metadata)

        indices = series_mask.flatten().nonzero(as_tuple=False).flatten()
        padded = series_features.new_zeros(batch * series, self.feature_dim)
        padded = padded.index_copy(0, indices, series_features).reshape(batch, series, self.feature_dim)
        study_features = self._pool_series(padded, series_mask)
        return self.head(study_features)
