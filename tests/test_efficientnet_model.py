import torch

from rsna_knee_abnormality_detection.config import DEFAULTS
from rsna_knee_abnormality_detection.constants import TARGETS
from rsna_knee_abnormality_detection.models import EFFICIENTNET_BACKBONES, TIMM_BACKBONES, StudyModel


def test_efficientnetv2_s_builds_checkpoints_and_reloads():
    config = {**DEFAULTS["model"], "backbone": "efficientnetv2_s", "pooling": "class_attention", "use_metadata": False,
              "use_positions": False, "grad_checkpointing": True}
    model = StudyModel(config, pretrained=False)
    assert "efficientnetv2_s" in EFFICIENTNET_BACKBONES and "efficientnetv2_s" in TIMM_BACKBONES
    assert model.feature_dim == 1280 and model.input_size is None
    reloaded = StudyModel(config, pretrained=False)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    images = torch.rand(1, 2, 2, 3, 64, 64)  # batch, series, window groups, channels, height, width
    mask = torch.ones(1, 2, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out = model(images, mask, torch.zeros(1, 2, 5))
    assert out.shape == (1, len(TARGETS))
