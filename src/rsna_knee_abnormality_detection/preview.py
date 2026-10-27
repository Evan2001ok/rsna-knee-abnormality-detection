"""Render actual dataset inputs for visual inspection before training."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .constants import PLANES
from .dataset import KneeDataset
from .train import study_partitions


def preview_inputs(config: dict, output: str | Path, count: int = 4, training: bool = False) -> Path:
    if count < 1:
        raise ValueError("count must be positive")
    train, _ = study_partitions(config)
    dataset = KneeDataset(config["data"]["manifest"], train.head(count), config["data"], training=training,
                          seed=config["seed"])
    width = config["data"]["image_size"]
    slots = 3 * config["data"]["series_per_plane"]
    canvas = Image.new("RGB", (slots * width, len(dataset) * (width + 44)), "#222222")
    draw = ImageDraw.Draw(canvas)
    for row in range(len(dataset)):
        item = dataset[row]
        for slot in range(slots):
            y, x = row * (width + 44), slot * width
            plane = PLANES[slot // config["data"]["series_per_plane"]]
            if item["series_mask"][slot]:
                group = config["data"]["slices_per_series"] // 2
                values = item["images"][slot, group, 1].numpy()
                rendered = Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8)).convert("RGB")
                canvas.paste(rendered, (x, y + 40))
                text = plane
            else:
                text = plane + " (missing)"
            draw.text((x + 4, y + 3), text, fill="white")
            draw.text((x + 4, y + 20), item["study_id"][-25:], fill="white")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
