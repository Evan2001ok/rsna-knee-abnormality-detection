"""Local TensorBoard charts without changing the training data or random state."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .constants import PLANES, TARGETS


def input_image(batch: dict) -> np.ndarray:
    """Show up to four sampled positions per real series in the first study."""
    images = batch["images"][0].detach().cpu()
    slots = batch["series_mask"][0].nonzero(as_tuple=False).flatten().tolist()
    positions = batch["positions"][0].detach().cpu()
    metadata = batch["metadata"][0].detach().cpu()
    if not slots:
        raise ValueError("Cannot display a study with no real MRI series")
    selected = np.unique(np.linspace(0, images.shape[1] - 1, min(4, images.shape[1]), dtype=int))
    height, width = images.shape[-2:]
    canvas = Image.new("RGB", (len(selected) * width, len(slots) * (height + 24)), "#181818")
    draw = ImageDraw.Draw(canvas)
    for row, slot in enumerate(slots):
        plane = PLANES[int(metadata[slot, :3].argmax())]
        for column, group in enumerate(selected):
            if "window_planes" in batch:
                plane = PLANES[int(batch["window_planes"][0, slot, group])]
            # The middle channel is the central MRI slice, not an RGB photograph.
            pixels = images[slot, group, 1].clamp(0, 1).numpy()
            tile = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
            x, y = column * width, row * (height + 24)
            canvas.paste(tile, (x, y + 24))
            draw.text((x + 3, y + 4), f"{plane} | {positions[slot, group]:.0%}", fill="white")
    return np.asarray(canvas)


class TrainingMonitor:
    def __init__(self, output_dir, config: dict, resume_step: int | None = None):
        self.settings = {
            "enabled": True, "log_every_updates": 10, "flush_seconds": 10, "images": True,
            "validation_every_updates": 0, "validation_studies": 32,
        } | config.get("monitoring", {})
        self.writer = None
        if self.settings["enabled"]:
            # Keep the training-only dependency out of offline inference imports.
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=str(Path(output_dir) / "tensorboard"), flush_secs=self.settings["flush_seconds"],
                purge_step=resume_step + 1 if resume_step is not None else None,
            )
            self.writer.add_custom_scalars({
                "Learning curves": {
                    "Epoch losses": ["Multiline", ["Loss/train_epoch", "Loss/validation"]],
                    "Training updates": ["Multiline", ["Loss/train_recent", "Loss/train_running"]],
                    "Per-finding AUC": ["Multiline", [f"AUC_by_finding/{target.replace(' ', '_')}" for target in TARGETS]],
                },
            })
            sample_interval = self.settings["validation_every_updates"]
            sample_size = self.settings["validation_studies"]
            if sample_interval > 0:
                sample_description = (
                    "All validation studies (validation_studies=0) are evaluated " if sample_size == 0 else
                    f"A fixed sample of up to {sample_size} validation studies is evaluated "
                ) + (
                    f"every {sample_interval} optimizer updates. The Validation_sample charts are for monitoring "
                    "only; they do not select the best checkpoint. The label counts and study IDs describe "
                    "the actual monitored studies. "
                )
            else:
                sample_description = (
                    "Validation sampling between epochs is disabled "
                    f"(validation_every_updates={sample_interval}, validation_studies={sample_size}). "
                )
            description = (
                f"**{'VERIFICATION RUN' if any(config.get('debug', {}).values()) else 'Training run'}**\n\n"
                f"Model: {config['model']['backbone']}; fold: {config['data']['fold']}; seed: {config['seed']}.\n\n"
                "The horizontal step axis counts optimizer updates. Full validation is recorded after each epoch "
                "and its score selects the best checkpoint. "
                f"{sample_description}"
                "Macro AUC requires all twelve findings; missing values are omitted, not replaced by zero.\n\n"
                f"Training measurements are logged every {self.settings['log_every_updates']} optimizer updates "
                "and at the final update of an epoch. Loss/train_running is the mean since the start of the "
                "current epoch. Loss/train_recent covers the current logging interval. Loss/train_epoch is the "
                "completed epoch's mean. Logged updates are flushed immediately.\n\n"
                f"Training loss weighting: {config['training'].get('target_weighting', 'observed')}. "
                f"Positive-term weighting: {config['training'].get('positive_weighting', 'none')}. "
                "Validation loss always averages known target labels. Compare validation AUC across weighting modes; "
                "their training losses use different objectives.\n\n"
                "Configuration:\n```yaml\n" + yaml.safe_dump(config, sort_keys=False) + "```"
            )
            self.writer.add_text("Run/settings", description, global_step=resume_step or 0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.writer:
            try:
                if exc_type is not None:
                    self.writer.add_text("Run/error", f"Stopped with {exc_type.__name__}: {exc}")
            finally:
                self.writer.close()

    def scalar(self, tag, value, step):
        if self.writer and value is not None and math.isfinite(float(value)):
            self.writer.add_scalar(tag, float(value), global_step=step)

    def support(self, metrics: dict, step: int):
        if not self.writer:
            return
        lines = ["Reference-label counts; unknown findings are excluded from the loss and AUC.\n",
                 "| Finding | Positive | Negative | Soft | Unknown |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for name, row in metrics["per_target"].items():
            lines.append(f"| {name} | {row['positive']} | {row['negative']} | {row['soft']} | {row['missing']} |")
        self.writer.add_text("Validation/label_counts", "\n".join(lines), global_step=step)

    def sample_support(self, metrics: dict, study_ids, step: int = 0):
        if not self.writer:
            return
        ids = [str(study_id) for study_id in study_ids]
        selection = "Full validation between epochs" if self.settings["validation_studies"] == 0 else "Fixed validation sample"
        context = (
            "These measurements use the entire validation partition between epoch boundaries. "
            if self.settings["validation_studies"] == 0 else
            "Sample AUC can change more sharply than full validation AUC. "
        )
        lines = [
            f"{selection}: **{len(ids)} studies**. These measurements are for monitoring only. "
            "Full validation at each epoch selects the best checkpoint.\n",
            f"{context}The twelve-finding macro AUC is "
            "reported only when every finding has both positive and negative labels. Otherwise, "
            "available_findings_only averages just the findings that can be scored. Unknown labels are "
            "excluded from loss and AUC; soft labels contribute to loss but not AUC.\n",
            "| Finding | Positive | Negative | Soft | Unknown |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, row in metrics["per_target"].items():
            lines.append(f"| {name} | {row['positive']} | {row['negative']} | {row['soft']} | {row['missing']} |")
        lines.extend(["\nFixed study IDs:", "```json", json.dumps(ids, indent=2), "```"])
        self.writer.add_text("Validation_sample/label_counts", "\n".join(lines), global_step=step)
        self.writer.flush()

    def batch_images(self, batch: dict, split: str, step: int):
        if self.writer and self.settings["images"]:
            self.writer.add_image(f"Inputs/{split}", input_image(batch), global_step=step, dataformats="HWC")

    def should_log(self, step: int, force: bool = False) -> bool:
        return bool(self.writer and (force or step % self.settings["log_every_updates"] == 0))

    def update(self, step: int, loss: float, learning_rates: dict, *, force: bool = False, measurements: dict | None = None):
        """Log epoch-cumulative loss, learning rates, and supplied interval measurements."""
        if not self.should_log(step, force):
            return
        self.scalar("Loss/train_running", loss, step)
        for name, value in learning_rates.items():
            self.scalar(f"Learning_rate/{name}", value, step)
        for tag, value in (measurements or {}).items():
            self.scalar(tag, value, step)
        self.writer.flush()

    def validation_sample(self, metrics: dict, step: int, seconds: float):
        """Log a monitoring-only sample without changing full-validation charts."""
        if not self.writer:
            return
        self.scalar("Validation_sample/loss", metrics["loss"], step)
        macro_auc = metrics["macro_auc"]
        self.scalar("Validation_sample/macro_12_findings", macro_auc, step)
        if macro_auc is None or not math.isfinite(float(macro_auc)):
            self.scalar("Validation_sample/available_findings_only", metrics["available_auc"], step)
        for name, values in metrics["per_target"].items():
            self.scalar(f"Validation_sample/AUC_by_finding/{name.replace(' ', '_')}", values["auc"], step)
        self.scalar("Validation_sample/auc_findings_available", metrics["auc_targets"], step)
        self.scalar("Validation_sample/seconds", seconds, step)
        self.scalar("Validation_sample/studies", metrics["studies"], step)
        self.writer.flush()

    def epoch(self, record: dict):
        step, metrics = record["global_step"], record["validation"]
        self.scalar("Loss/train_epoch", record["train_loss"], step)
        self.scalar("Loss/validation", metrics["loss"], step)
        self.scalar("AUC/macro_12_findings", metrics["macro_auc"], step)
        if metrics["macro_auc"] is None:
            self.scalar("AUC/available_findings_only", metrics["available_auc"], step)
        for name, values in metrics["per_target"].items():
            self.scalar(f"AUC_by_finding/{name.replace(' ', '_')}", values["auc"], step)
        self.scalar("Progress/completed_epoch", record["epoch"], step)
        self.scalar("Progress/auc_findings_available", metrics["auc_targets"], step)
        self.scalar("Performance/epoch_seconds", record["seconds"], step)
        self.scalar("Performance/peak_GPU_GiB", record["peak_gpu_gb"], step)
        self.scalar("Optimization/skipped_updates", record.get("skipped_updates", 0), step)
        if self.writer:
            self.writer.flush()

    def training_epoch(self, record: dict):
        """Epoch summary when the validation pass is skipped."""
        step = record["global_step"]
        self.scalar("Loss/train_epoch", record["train_loss"], step)
        self.scalar("Progress/completed_epoch", record["epoch"], step)
        self.scalar("Performance/epoch_seconds", record["seconds"], step)
        self.scalar("Performance/peak_GPU_GiB", record["peak_gpu_gb"], step)
        self.scalar("Optimization/skipped_updates", record.get("skipped_updates", 0), step)
        if self.writer:
            self.writer.flush()

    def status(self, completed: bool, epoch: int, step: int):
        if self.writer:
            message = f"{'Finished' if completed else 'Stopped at checkpoint after'} epoch {epoch}."
            self.writer.add_text("Run/status", message, global_step=step)


def import_histories(runs_dir: str | Path) -> dict:
    """Import saved epoch summaries once; do not duplicate existing event logs."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    paths = [root / "history.json"] if (root / "history.json").is_file() else sorted(root.rglob("history.json"))
    imported, skipped = [], []
    for path in paths:
        run = path.parent
        if any((run / "tensorboard").glob("events.out.tfevents.*")):
            skipped.append(str(run))
            continue
        history = json.loads(path.read_text())
        if not isinstance(history, list) or not history:
            raise ValueError(f"No completed epochs to import in {path}")
        config = yaml.safe_load((run / "config.yaml").read_text())
        config["monitoring"] = {
            "enabled": True, "log_every_updates": 10, "flush_seconds": 10, "images": False,
            "validation_every_updates": 0, "validation_studies": 0,
        }
        # Publish only complete event files, so interruption can be retried and
        # never leaves a partial import that looks like an already-logged run.
        with tempfile.TemporaryDirectory(prefix=".tensorboard-import-", dir=run) as temporary:
            with TrainingMonitor(temporary, config) as monitor:
                monitor.writer.add_text(
                    "Run/history_import", "Imported saved epoch summaries only. Original optimizer-update measurements, "
                    "validation-sample measurements, and input images are unavailable.",
                )
                for record in history:
                    monitor.epoch(record)
                    rates = record["learning_rates"]
                    names = ["head"] if config["model"].get("freeze_encoder", False) else ["encoder", "head"]
                    for name, value in zip(names, rates, strict=True):
                        monitor.scalar(f"Learning_rate/{name}", value, record["global_step"])
                monitor.support(history[-1]["validation"], history[-1]["global_step"])
            Path(temporary, "tensorboard").rename(run / "tensorboard")
        imported.append(str(run))
    return {"imported": imported, "already_logged": skipped}
