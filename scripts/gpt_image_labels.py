#!/usr/bin/env python3
"""Ask a hosted model (GPT-6 Astra, Claude Opus 5.5 or Gemini 4 Argon) to label knee MRI studies from the images alone.

Steps (each writes under --out, which lives in artifacts/ and stays out of git):
  prepare   Save every slice of every series as 8-bit grayscale PNG (longest side capped, 512 px by
            default), ordered by physical position and turned to standard viewing orientation, plus
            pack.json. Offline.
  estimate  Count images, input tokens and request size, and print a cost range. Offline.
  run       Send one request per study and save the raw replies. --provider openai uses the Responses
            API; --provider anthropic uses the Messages API, uploading a study's images through the
            Files API (and deleting them after the reply is saved) only when the request would
            exceed 32 MB; --provider google uses the Gemini API's generateContent, uploading a study's images
            through the Gemini Files API (and deleting them after the reply is saved) when the request would exceed
            19 MB. Refuses without --yes. API keys are read from .env and never printed.
  cleanup   Delete any of these studies' images still uploaded to the Files API.
  parse     Check saved replies against the answer format and write labels.csv (probabilities 0-1).

The radiology report is never read or sent. The instructions are fixed in configs/llm/gpt_image_labels_v1.md (the
exact prompt we used); --prompt and --findings (OpenAI only) send another instruction file that asks about fewer
findings (we once tried a focused reading of synovitis, PF OA and lateral OA; that prompt is not included). A run
records its prompt, findings, effort and tier in run-settings.json in its replies folder and refuses to mix settings.
The GPT image labels in our training labels were made with --provider openai --effort medium (the default effort is
high) on the default 512 px images.
Example: uv run python scripts/gpt_image_labels.py prepare --studies <id> <id>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageFilter

from rsna_knee_abnormality_detection.constants import STUDY_ID
from rsna_knee_abnormality_detection.imaging import load_dicom_series

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = {
    "acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA", "lateral_oa": "Lateral OA", "pf_oa": "PF OA", "effusion": "Effusion",
    "synovitis": "Synovitis", "bakers_cyst": "Baker's", "contusion": "Contusion", "fracture": "Fracture",
}
MODEL = "gpt-6-astra"
PROMPT = ROOT / "configs/llm/gpt_image_labels_v1.md"
API_URL = "https://api.openai.com/v1/responses"
PLANE_ORDER = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
PATCH, HIGH_PATCH_BUDGET, TOKEN_MULTIPLIER = 32, 2500, 1.2
PRICE_PER_MILLION = {"input": 10.0, "output": 50.0}  # standard tier; the Batch API is half price
# Measured on the 2-study pilot (medium about 1,200, high about 4,400, max about 9,400); low and xhigh are guesses.
OUTPUT_TOKEN_GUESSES = {"low": 800, "medium": 1300, "high": 4500, "xhigh": 7000, "max": 9500}
# Claude rejects the nested per-finding schema as too large a grammar, so it answers as a list of entries.
CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {"finding": {"type": "string", "enum": list(FINDINGS)}, "probability": {"type": "integer"},
                           "assessable": {"type": "boolean"}, "evidence": {"type": "array", "items": {"type": "string"}},
                           "note": {"type": "string"}},
            "required": ["finding", "probability", "assessable", "evidence", "note"], "additionalProperties": False}},
        "image_quality_note": {"type": "string"},
    },
    "required": ["findings", "image_quality_note"], "additionalProperties": False,
}
CLAUDE_FORMAT_NOTE = ("Answer format: give exactly one entry in \"findings\" for each of the twelve labels, naming it by its "
                      "identifier (" + ", ".join(FINDINGS) + ").")
CLAUDE_MODEL = "claude-opus-5-5"  # default; --model picks another Claude model
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_FILES_URL = "https://api.anthropic.com/v1/files"
CLAUDE_PATCH, CLAUDE_MAX_EDGE, CLAUDE_MAX_IMAGE_TOKENS, CLAUDE_MAX_IMAGES = 28, 2576, 4784, 600
# Standard tier per million tokens (Message Batches are half price); cache reads cost this share of the input price.
CLAUDE_PRICES = {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.05},
                 "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.025}}
CLAUDE_REQUEST_LIMIT = 30_000_000  # bytes; the Messages API rejects requests over 32 MB
CLAUDE_OUTPUT_TOKEN_GUESSES = {"low": 1000, "medium": 2000, "high": 5000, "xhigh": 8000, "max": 12000}  # unmeasured
GEMINI_MODEL = "gemini-4-argon"  # released Oct 13, 2026
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart"
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/"
# Per million tokens; introductory price until Nov 30, 2026 ($4 / $20 after). Thinking is billed as output.
GEMINI_PRICES = {"gemini-4-argon": {"input": 2.0, "output": 10.0}}
GEMINI_REQUEST_LIMIT = 19_000_000  # bytes; the Gemini API takes about 20 MB of inline data per request
FILE_EXPIRY_SECONDS = 172_800  # uploaded images also delete themselves after two days
RETRY_CODES = (408, 429, 500, 502, 503, 504, 529)  # flex replies "too busy" with 429 and may time out with 408
# Uploads in flight across all studies at once. Four studies uploading together (16 at once) all failed on the first run.
UPLOAD_SLOTS = threading.Semaphore(4)
FILES_TIMEOUT = 120  # seconds to wait on one upload or delete; on the first run a connection hung for 45 minutes


def make_schema(keys: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "object",
                "properties": {key: {
                    "type": "object",
                    "properties": {"probability": {"type": "integer"}, "assessable": {"type": "boolean"},
                                   "evidence": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string"}},
                    "required": ["probability", "assessable", "evidence", "note"], "additionalProperties": False,
                } for key in keys},
                "required": list(keys), "additionalProperties": False,
            },
            "image_quality_note": {"type": "string"},
        },
        "required": ["findings", "image_quality_note"], "additionalProperties": False,
    }


SCHEMA = make_schema(tuple(FINDINGS))


def to_uint8(images: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
    """Window a whole series together (0.5-99.5 percentiles) so slices stay comparable."""
    finite = np.concatenate([image[np.isfinite(image)].ravel() for image in images])
    low, high = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    high = high if high > low else low + 1.0
    scaled = [np.round(np.clip(np.nan_to_num((image - low) / (high - low), nan=0.0), 0, 1) * 255).astype(np.uint8)
              for image in images]
    return scaled, [low, high]


def shrink(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Scale an 8-bit image down so its longest side is at most max_side, keeping its shape."""
    height, width = image.shape
    if not max_side or max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))


def tissue_crop(scaled: list[np.ndarray], min_share: float = 0.03, margin: float = 0.04, full_frame: float = 30.0,
                above_noise: float = 15.0) -> tuple[list[int], tuple[int, int, int, int] | None]:
    """Slices to keep and one crop box (top, left, bottom, right; exclusive ends) for a whole series.

    Each slice is smoothed with a 9-pixel box blur. The series' noise level is the median brightness of the outer ring
    (4% of each side) of its slices; tissue is anything brighter than that level plus `above_noise` (at least 20 of
    255), so scanner noise does not count. When the ring itself is brighter than `full_frame`, tissue fills the frame
    and the series is left as it is. Slices with less than `min_share` tissue are dropped only from the two ends of the
    series (edge slices that miss the knee); the box is the union of the kept slices' tissue plus a `margin` share of
    each side, so no anatomy is cut and every slice keeps the same scale. Mixed slice shapes or no tissue at all leave
    the series as it is.
    """
    if len({image.shape for image in scaled}) != 1:
        return list(range(len(scaled))), None
    blurred = [np.asarray(Image.fromarray(image).filter(ImageFilter.BoxBlur(4))).astype(np.float32) for image in scaled]
    height, width = blurred[0].shape
    ring_rows, ring_columns = max(2, round(0.04 * height)), max(2, round(0.04 * width))
    ring = np.ones((height, width), bool)
    ring[ring_rows:height - ring_rows, ring_columns:width - ring_columns] = False
    noise = float(np.median([np.median(image[ring]) for image in blurred]))
    if noise > full_frame:
        return list(range(len(scaled))), None
    masks = [image > max(20.0, noise + above_noise) for image in blurred]
    shares = [mask.mean() for mask in masks]
    first, last = 0, len(scaled) - 1
    while first <= last and shares[first] < min_share:
        first += 1
    while last >= first and shares[last] < min_share:
        last -= 1
    if first > last:
        return list(range(len(scaled))), None
    union = np.any(masks[first:last + 1], axis=0)
    rows, columns = np.where(union.any(1))[0], np.where(union.any(0))[0]
    pad_rows, pad_columns = max(4, round(margin * height)), max(4, round(margin * width))
    box = (max(0, int(rows[0]) - pad_rows), max(0, int(columns[0]) - pad_columns),
           min(height, int(rows[-1]) + 1 + pad_rows), min(width, int(columns[-1]) + 1 + pad_columns))
    return list(range(first, last + 1)), box


def knee_side(series_meta: list[tuple[dict, dict]]) -> tuple[str | None, str | None]:
    """Left or right knee: DICOM laterality when recorded, else which side of the patient a front or top view is centered on."""
    for _, header in series_meta:
        tag = str(header.get("Laterality", "") or header.get("ImageLaterality", "")).strip().upper()
        if tag in ("L", "R"):
            return ("left" if tag == "L" else "right"), "dicom_laterality"
    for meta, _ in series_meta:
        if meta["anatomical_plane"] not in ("Coronal", "Axial"):
            continue
        middle = meta["num_slices"] // 2
        position, orientation = meta["image_positions_lps"][middle], meta["original_orientations_lps"][middle]
        spacing, shape = meta["original_pixel_spacings_mm"][middle], meta["native_shapes"][middle]
        if position is None or orientation is None or spacing is None:
            continue
        rows, columns = shape
        center = (np.asarray(position) + np.asarray(orientation[:3]) * columns / 2 * spacing[1]
                  + np.asarray(orientation[3:]) * rows / 2 * spacing[0])
        return ("left" if center[0] > 0 else "right"), "geometry"
    return None, None


def study_header(side: str | None, source: str | None, series_count: int, image_count: int) -> str:
    text = f"Knee MRI study with {series_count} series and {image_count} images. "
    if side is None:
        return text + "The scanned side is unknown; judge medial and lateral from the anatomy (the fibula is lateral)."
    how = "recorded in the scan header" if source == "dicom_laterality" else "estimated from the scanner geometry"
    medial = "left" if side == "left" else "right"
    return text + (f"Scanned knee: {side} ({how}). In coronal and axial images the medial side is on the image {medial}. "
                   f"In sagittal series the first slices are {'medial' if side == 'left' else 'lateral'} "
                   f"and the last slices {'lateral' if side == 'left' else 'medial'}.")


def series_text(entry: dict) -> str:
    order = {"Sagittal": "from the patient's right to left", "Coronal": "from front to back",
             "Axial": "from foot to head"}[entry["plane"]]
    spacing = f", about {entry['slice_spacing_mm']:.1f} mm apart" if entry["slice_spacing_mm"] else ""
    text = (f"Series S{entry['number']}: {entry['plane']}, {entry['description'] or 'no description'}; "
            f"fluid-sensitive: {'yes' if entry['fluid_sensitive'] else 'no'}, "
            f"fat-suppressed: {'yes' if entry['fat_suppressed'] else 'no'}; {len(entry['images'])} slices{spacing}, "
            f"ordered {order}.")
    if entry["ordering"] != "physical_position":
        text += " Slice order follows the stored numbering because positions are missing."
    if not entry["orientation_standardized"]:
        text += " This series could not be turned to standard orientation and is shown as stored."
    return text


def prepare_study(study: str, data_root: Path, out_root: Path, max_side: int | None = 512, crop: bool = False) -> Path:
    series = pd.read_csv(data_root / "train_series.csv", dtype={STUDY_ID: str, "SeriesInstanceUID": str})
    rows = series[series[STUDY_ID].eq(study)].copy()
    if rows.empty:
        raise ValueError(f"{study}: no series listed")
    rows["order"] = rows["Anatomical_Plane"].map(PLANE_ORDER)
    rows = rows.sort_values(["order", "Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"],
                            ascending=[True, False, False, True])
    out = out_root / study
    out.mkdir(parents=True, exist_ok=True)
    loaded, entries = [], []
    for number, row in enumerate(rows.itertuples(index=False), 1):
        series_dir = data_root / "train_series" / study / row.SeriesInstanceUID
        images, meta = load_dicom_series(series_dir, row.Anatomical_Plane)
        header = pydicom.dcmread(series_dir / meta["source_frames"][0]["file"], stop_before_pixels=True)
        scaled, window = to_uint8(images)
        kept, box = tissue_crop(scaled) if crop else (list(range(len(scaled))), None)
        files = []
        for index, image in enumerate(scaled, 1):
            if index - 1 not in kept:
                continue
            label = f"S{number}-{index:02d}"
            if box is not None:
                image = image[box[0]:box[2], box[1]:box[3]]
            small = shrink(image, max_side)
            Image.fromarray(small).save(out / f"{label}.png", optimize=True)
            files.append({"label": label, "file": f"{label}.png", "shape": list(small.shape),
                          "source_shape": list(image.shape)})
        loaded.append((meta, header))
        entries.append({"number": number, "series": row.SeriesInstanceUID, "plane": row.Anatomical_Plane,
                        "description": str(header.get("SeriesDescription", "")).strip(),
                        "fluid_sensitive": bool(int(row.Fluid_Sensitive)), "fat_suppressed": bool(int(row.Fat_Suppression)),
                        "slice_spacing_mm": meta["slice_spacing_mm"], "ordering": meta["ordering"],
                        "orientation_standardized": meta["orientation_standardized"], "window": window,
                        "warnings": meta["warnings"], "images": files,
                        **({"crop_box": box, "frame_shape": list(scaled[0].shape),
                            "dropped_edge_slices": len(scaled) - len(kept)} if crop else {})})
    side, source = knee_side(loaded)
    pack = {"study": study, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "knee_side": side, "knee_side_source": source, "series": entries,
            "image_count": sum(len(entry["images"]) for entry in entries), "max_side": max_side,
            "prompt": {"path": str(PROMPT.relative_to(ROOT)), "sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest()}}
    (out / "pack.json").write_text(json.dumps(pack, indent=1) + "\n")
    return out


def image_tokens(width: int, height: int) -> int:
    """Billable tokens at detail "high": 32-pixel patches, a 2,500-patch budget and a 1.2 multiplier."""
    patches = math.ceil(width / PATCH) * math.ceil(height / PATCH)
    if patches > HIGH_PATCH_BUDGET:
        scale = math.sqrt(HIGH_PATCH_BUDGET * PATCH * PATCH / (width * height))
        patches = min(HIGH_PATCH_BUDGET, math.ceil(int(width * scale) / PATCH) * math.ceil(int(height * scale) / PATCH))
    return math.ceil(patches * TOKEN_MULTIPLIER)


def build_request(pack_dir: Path, effort: str, embed_images: bool = True, service_tier: str | None = None,
                  prompt: Path = PROMPT, keys: tuple[str, ...] = tuple(FINDINGS)) -> dict:
    pack = json.loads((pack_dir / "pack.json").read_text())
    content = [{"type": "input_text", "text": study_header(pack["knee_side"], pack["knee_side_source"],
                                                           len(pack["series"]), pack["image_count"])}]
    for entry in pack["series"]:
        content.append({"type": "input_text", "text": series_text(entry)})
        for image in entry["images"]:
            content.append({"type": "input_text", "text": image["label"]})
            data = (base64.b64encode((pack_dir / image["file"]).read_bytes()).decode() if embed_images
                    else f"<{image['file']}>")
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{data}", "detail": "high"})
    tier = {"service_tier": service_tier} if service_tier else {}
    return {
        "model": MODEL,
        **tier,
        "reasoning": {"effort": effort},
        "input": [{"role": "developer", "content": [{"type": "input_text", "text": prompt.read_text()}]},
                  {"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "knee_mri_labels", "strict": True, "schema": make_schema(keys)}},
        "max_output_tokens": 100000,
        "store": False,
    }


def estimate(pack_dir: Path, effort: str, provider: str = "openai", model: str = CLAUDE_MODEL,
             prompt: Path = PROMPT) -> dict:
    pack = json.loads((pack_dir / "pack.json").read_text())
    shapes = [image["shape"] for entry in pack["series"] for image in entry["images"]]
    request = build_request(pack_dir, effort, embed_images=False)
    text = prompt.read_text() + "".join(part["text"] for part in request["input"][1]["content"] if part["type"] == "input_text")
    count = claude_image_tokens if provider == "anthropic" else image_tokens
    input_tokens = sum(count(width, height) for height, width in shapes) + math.ceil(len(text) / 4)
    payload = sum(math.ceil((pack_dir / image["file"]).stat().st_size / 3) * 4
                  for entry in pack["series"] for image in entry["images"]) + len(json.dumps(request))
    guesses, prices = ((CLAUDE_OUTPUT_TOKEN_GUESSES, CLAUDE_PRICES[model]) if provider == "anthropic"
                       else (OUTPUT_TOKEN_GUESSES, PRICE_PER_MILLION))
    output_tokens = guesses[effort]
    standard = input_tokens / 1e6 * prices["input"] + output_tokens / 1e6 * prices["output"]
    return {"study": pack["study"], "images": len(shapes), "shapes": sorted({f"{w}x{h}" for h, w in shapes}),
            "input_tokens": input_tokens, "assumed_output_tokens": output_tokens, "payload_mb": round(payload / 1e6, 1),
            "standard_usd": round(standard, 2), "batch_usd": round(standard / 2, 2)}


def api_key(name: str = "OPENAI_API_KEY") -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name and value.strip():
            return value.strip().strip("'\"")
    raise SystemExit(f"{name} is missing from .env")


def request_json(url: str, headers: dict, data: bytes | None = None, method: str = "POST", attempts: int = 6,
                 timeout: float = 3600) -> dict:
    """Call a JSON API, waiting and retrying when it is busy, rate-limited or the connection drops."""
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            if error.code in RETRY_CODES and attempt + 1 < attempts:
                wait = error.headers.get("retry-after")
                time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 20 * 2 ** attempt)
                continue
            raise RuntimeError(f"API error {error.code} from {url}: {detail}") from None
        except (OSError, http.client.HTTPException) as error:
            if attempt + 1 < attempts:
                time.sleep(20 * 2 ** attempt)
                continue
            raise RuntimeError(f"Connection problem with {url}: {error!r}") from None
    raise RuntimeError(f"No answer from {url}")


def send(body: dict, key: str) -> dict:
    return request_json(API_URL, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json.dumps(body).encode())


def claude_headers(key: str) -> dict:
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}


def claude_image_tokens(width: int, height: int) -> int:
    """Claude counts 28-pixel patches; images over 2,576 px or 4,784 patches are scaled down first."""
    scale = min(1.0, CLAUDE_MAX_EDGE / max(width, height))
    tokens = math.ceil(width * scale / CLAUDE_PATCH) * math.ceil(height * scale / CLAUDE_PATCH)
    if tokens > CLAUDE_MAX_IMAGE_TOKENS:
        scale *= math.sqrt(CLAUDE_MAX_IMAGE_TOKENS / tokens)
        tokens = math.ceil(width * scale / CLAUDE_PATCH) * math.ceil(height * scale / CLAUDE_PATCH)
    return min(tokens, CLAUDE_MAX_IMAGE_TOKENS)


def build_claude_request(pack_dir: Path, effort: str, file_ids: dict | None = None, embed_images: bool = True,
                         model: str = CLAUDE_MODEL) -> dict:
    pack = json.loads((pack_dir / "pack.json").read_text())
    if pack["image_count"] > CLAUDE_MAX_IMAGES:
        raise ValueError(f"{pack['study']}: {pack['image_count']} images exceed the {CLAUDE_MAX_IMAGES}-image request limit")
    content = [{"type": "text", "text": study_header(pack["knee_side"], pack["knee_side_source"],
                                                     len(pack["series"]), pack["image_count"])}]
    for entry in pack["series"]:
        content.append({"type": "text", "text": series_text(entry)})
        for image in entry["images"]:
            content.append({"type": "text", "text": image["label"]})
            if file_ids is not None:
                source = {"type": "file", "file_id": file_ids[image["label"]]}
            else:
                data = (base64.b64encode((pack_dir / image["file"]).read_bytes()).decode() if embed_images
                        else f"<{image['file']}>")
                source = {"type": "base64", "media_type": "image/png", "data": data}
            content.append({"type": "image", "source": source})
    content.append({"type": "text", "text": CLAUDE_FORMAT_NOTE})
    return {"model": model, "max_tokens": 64000, "system": PROMPT.read_text(),
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}}}


def build_gemini_request(pack_dir: Path, effort: str, file_uris: dict | None = None, embed_images: bool = True,
                         prompt: Path = PROMPT, keys: tuple[str, ...] = tuple(FINDINGS),
                         thinking_budget: int | None = None) -> dict:
    """The same instructions, labels and images as build_request, in the Gemini API's generateContent form.

    A thinking budget (tokens) replaces the thinking level: the API accepts only one of the two.
    """
    pack = json.loads((pack_dir / "pack.json").read_text())
    parts = [{"text": study_header(pack["knee_side"], pack["knee_side_source"], len(pack["series"]), pack["image_count"])}]
    for entry in pack["series"]:
        parts.append({"text": series_text(entry)})
        for image in entry["images"]:
            parts.append({"text": image["label"]})
            if file_uris:
                parts.append({"fileData": {"mimeType": "image/png", "fileUri": file_uris[image["file"]]}})
            else:
                data = (base64.b64encode((pack_dir / image["file"]).read_bytes()).decode() if embed_images
                        else f"<{image['file']}>")
                parts.append({"inlineData": {"mimeType": "image/png", "data": data}})
    return {
        "systemInstruction": {"parts": [{"text": prompt.read_text()}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": make_schema(keys),
                             "thinkingConfig": {"thinkingBudget": thinking_budget} if thinking_budget else {"thinkingLevel": effort},
                             "maxOutputTokens": 32768},
    }


def upload_gemini_pack(pack_dir: Path, key: str, workers: int = 4) -> dict[str, str]:
    """Upload a study's images to the Gemini Files API; returns {image file: file name} (files/...)."""
    pack = json.loads((pack_dir / "pack.json").read_text())

    def upload(image: dict) -> tuple[str, str]:
        boundary = uuid.uuid4().hex
        meta = json.dumps({"file": {"display_name": f"{pack_dir.name[-12:]}-{Path(image['file']).name}"}})
        body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
                f"--{boundary}\r\nContent-Type: image/png\r\n\r\n").encode() + (pack_dir / image["file"]).read_bytes() \
            + f"\r\n--{boundary}--\r\n".encode()
        with UPLOAD_SLOTS:
            reply = request_json(GEMINI_UPLOAD_URL, {"x-goog-api-key": key, "Content-Type": f"multipart/related; boundary={boundary}"},
                                 body, timeout=FILES_TIMEOUT)
        return image["file"], reply["file"]["name"]

    images = [image for entry in pack["series"] for image in entry["images"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(upload, images))


def delete_gemini_files(names: list[str], key: str, workers: int = 4) -> list[str]:
    """Delete uploaded Gemini files; returns the names that could not be deleted (they expire after two days)."""
    def delete(name: str) -> str | None:
        try:
            request_json(GEMINI_API_ROOT + name, {"x-goog-api-key": key}, method="DELETE", attempts=3, timeout=FILES_TIMEOUT)
            return None
        except RuntimeError:
            return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [name for name in pool.map(delete, names) if name]


def gemini_usage_cost(usage: dict, model: str = GEMINI_MODEL) -> dict:
    input_tokens = usage.get("promptTokenCount", 0)
    thinking, answer = usage.get("thoughtsTokenCount", 0), usage.get("candidatesTokenCount", 0)
    price = GEMINI_PRICES[model]
    standard = (input_tokens * price["input"] + (thinking + answer) * price["output"]) / 1e6
    return {"input_tokens": input_tokens, "output_tokens": thinking + answer, "reasoning_tokens": thinking,
            "standard_usd": round(standard, 4)}


def payload_bytes(pack_dir: Path) -> int:
    """Approximate request size with images sent inline as base64."""
    pack = json.loads((pack_dir / "pack.json").read_text())
    return sum(math.ceil((pack_dir / image["file"]).stat().st_size / 3) * 4
               for entry in pack["series"] for image in entry["images"]) + 100_000


def multipart_body(fields: dict[str, str], filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
             for name, value in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n")
    return "".join(parts).encode() + content + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def upload_pack(pack_dir: Path, key: str, workers: int = 4) -> dict[str, str]:
    """Upload a study's images to the Files API; each file also expires on its own after two days.

    If any upload fails, the images already uploaded are deleted before the error is passed on."""
    pack = json.loads((pack_dir / "pack.json").read_text())
    images = [image for entry in pack["series"] for image in entry["images"]]

    def upload(image: dict) -> tuple[str, str]:
        body, content_type = multipart_body({"expires_in_seconds": str(FILE_EXPIRY_SECONDS)},
                                            f"{pack['study'][-12:]}-{image['file']}",
                                            (pack_dir / image["file"]).read_bytes(), "image/png")
        with UPLOAD_SLOTS:
            reply = request_json(CLAUDE_FILES_URL, {**claude_headers(key), "Content-Type": content_type}, body,
                                 timeout=FILES_TIMEOUT)
        return image["label"], reply["id"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload, image) for image in images]
    uploaded, errors = {}, []
    for future in futures:
        try:
            label, file_id = future.result()
            uploaded[label] = file_id
        except Exception as error:  # report after the clean-up below
            errors.append(error)
    if errors:
        delete_files(list(uploaded.values()), key)
        raise errors[0]
    return uploaded


def delete_files(file_ids: list[str], key: str, workers: int = 4) -> list[str]:
    """Delete uploaded files; returns the ones that could not be deleted (they still expire on their own)."""

    def delete(file_id: str) -> str | None:
        try:
            request_json(f"{CLAUDE_FILES_URL}/{file_id}", claude_headers(key), method="DELETE", timeout=FILES_TIMEOUT)
        except RuntimeError as error:
            return None if str(error).startswith("API error 404") else file_id  # 404: already gone
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [file_id for file_id in pool.map(delete, file_ids) if file_id]


def list_files(key: str) -> list[dict]:
    files, after = [], None
    while True:
        page = request_json(f"{CLAUDE_FILES_URL}?limit=1000" + (f"&after_id={after}" if after else ""),
                            claude_headers(key), method="GET", timeout=FILES_TIMEOUT)
        files += page.get("data", [])
        if not page.get("has_more"):
            return files
        after = page["last_id"]


def reply_text(response: dict) -> str:
    if "candidates" in response:  # Gemini API
        candidate = (response["candidates"] or [{}])[0]
        parts = [part["text"] for part in (candidate.get("content") or {}).get("parts", [])
                 if "text" in part and not part.get("thought")]
        if not parts:
            raise ValueError(f"No answer text (finish reason {candidate.get('finishReason')})")
        return "".join(parts)
    if response.get("type") == "message":  # Claude Messages API
        parts = [block["text"] for block in response.get("content", []) if block.get("type") == "text"]
        if not parts:
            raise ValueError(f"No answer text (stop reason {response.get('stop_reason')})")
        return "".join(parts)
    parts = [content["text"] for item in response.get("output", []) if item.get("type") == "message"
             for content in item.get("content", []) if content.get("type") == "output_text"]
    if not parts:
        raise ValueError(f"No answer text (status {response.get('status')}, details {response.get('incomplete_details')})")
    return "".join(parts)


def parse_reply(study: str, text: str, keys: tuple[str, ...] = tuple(FINDINGS)) -> dict:
    answer = json.loads(text)
    findings = answer["findings"]
    if isinstance(findings, list):  # Claude's list form: one entry per finding
        names = [item["finding"] for item in findings]
        if len(names) != len(set(names)):
            raise ValueError(f"{study}: a finding appears more than once")
        findings = {item["finding"]: item for item in findings}
    if set(findings) != set(keys):
        raise ValueError(f"{study}: findings differ from the expected {'twelve' if len(keys) == 12 else len(keys)}")
    row = {STUDY_ID: study}
    for key in keys:
        target = FINDINGS[key]
        item = findings[key]
        probability = item["probability"]
        if isinstance(probability, bool) or not isinstance(probability, int) or not 0 <= probability <= 100:
            raise ValueError(f"{study}: {key} probability must be an integer from 0 to 100")
        if not isinstance(item["assessable"], bool):
            raise ValueError(f"{study}: {key} assessable must be true or false")
        row[target] = probability / 100
        row[f"{target}__assessable"] = item["assessable"]
        row[f"{target}__evidence"] = " ".join(item["evidence"][:3])
        row[f"{target}__note"] = item["note"]
    row["image_quality_note"] = answer["image_quality_note"]
    return row


def pending_studies(replies_dir: Path, studies: list[str]) -> list[str]:
    """Studies without a saved reply, so a rerun never pays twice for the same study."""
    return [study for study in studies if not (replies_dir / f"{study}.json").exists()]


def labels_path(out: Path, replies: str) -> Path:
    """labels.csv for the default replies folder, labels-<folder>.csv for any other."""
    return out / ("labels.csv" if replies == "replies" else f"labels-{replies}.csv")


def usage_cost(usage: dict, service_tier: str | None = None) -> dict:
    """standard_usd is the normal price; charged_usd is half of it when the reply says it ran on flex."""
    input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    standard = ((input_tokens - cached) * PRICE_PER_MILLION["input"] + cached * PRICE_PER_MILLION["input"] / 10
                + output_tokens * PRICE_PER_MILLION["output"]) / 1e6
    return {"input_tokens": input_tokens, "cached_input_tokens": cached, "output_tokens": output_tokens,
            "reasoning_tokens": reasoning, "standard_usd": round(standard, 4),
            "charged_usd": round(standard / 2 if service_tier == "flex" else standard, 4)}


def claude_usage_cost(usage: dict, model: str = CLAUDE_MODEL) -> dict:
    """Thinking is billed inside output tokens; cache writes cost 1.25x input."""
    input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    written, read = usage.get("cache_creation_input_tokens") or 0, usage.get("cache_read_input_tokens") or 0
    price = CLAUDE_PRICES[model]
    standard = ((input_tokens + 1.25 * written + price["cache_read"] * read) * price["input"]
                + output_tokens * price["output"]) / 1e6
    return {"input_tokens": input_tokens + written + read, "output_tokens": output_tokens, "standard_usd": round(standard, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "estimate", "run", "parse", "cleanup"])
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/rsna-knee-abnormality-detection")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/gpt-image-labels/v1")
    parser.add_argument("--effort", choices=list(OUTPUT_TOKEN_GUESSES), default="high")
    parser.add_argument("--yes", action="store_true", help="Required for run: confirms sending images and paying")
    parser.add_argument("--replies", default="replies", help="Folder (under --out) for replies, so runs do not overwrite each other")
    parser.add_argument("--workers", type=int, default=1, help="Requests sent at the same time")
    parser.add_argument("--max-side", type=int, default=512, help="Longest image side in pixels when preparing (0 keeps full size)")
    parser.add_argument("--crop-tissue", action="store_true",
                        help="prepare: crop each series to one box around its tissue and drop near-empty slices at its two ends "
                             "(use another --out so the uncropped packs stay as they are)")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google"], default="openai")
    parser.add_argument("--model", choices=[*CLAUDE_PRICES, *GEMINI_PRICES],
                        help=f"Claude model for --provider anthropic (default {CLAUDE_MODEL}) or Gemini model for --provider google "
                             f"(default {GEMINI_MODEL})")
    parser.add_argument("--service-tier", choices=["flex"], help="OpenAI only: half price, slower, sometimes busy (retried)")
    parser.add_argument("--thinking-budget", type=int,
                        help="Google only: thinking tokens allowed, instead of --effort's thinking level (the API takes one of the two; "
                             "it must leave room for the answer within the 32,768 output tokens)")
    parser.add_argument("--prompt", type=Path, default=PROMPT, help="OpenAI only: instruction file (default: the twelve-finding v1)")
    parser.add_argument("--findings", nargs="+", choices=list(FINDINGS), default=list(FINDINGS),
                        help="OpenAI only: the findings the instruction file asks about")
    args = parser.parse_args()
    if args.model is None:
        args.model = GEMINI_MODEL if args.provider == "google" else CLAUDE_MODEL
    if (args.provider == "google") != (args.model in GEMINI_PRICES):
        raise SystemExit("--model must match --provider")
    if args.thinking_budget is not None and (args.provider != "google" or not 0 < args.thinking_budget <= 30000):
        raise SystemExit("--thinking-budget is for --provider google only, between 1 and 30,000 tokens")
    packs = args.out / "packs"
    keys = tuple(key for key in FINDINGS if key in args.findings)
    if args.provider == "anthropic" and (keys != tuple(FINDINGS) or args.prompt != PROMPT):
        raise SystemExit("--prompt and --findings are for --provider openai only")
    if args.step == "prepare":
        for study in args.studies:
            pack_dir = prepare_study(study, args.data_root, packs, args.max_side or None, args.crop_tissue)
            print(json.dumps({"study": study, "images": json.loads((pack_dir / "pack.json").read_text())["image_count"]}))
    elif args.step == "estimate":
        rows = [estimate(packs / study, args.effort, args.provider, args.model, args.prompt) for study in args.studies]
        for row in rows:
            print(json.dumps(row))
        print(json.dumps({"studies": len(rows), "standard_usd": round(sum(r["standard_usd"] for r in rows), 2),
                          "batch_usd": round(sum(r["batch_usd"] for r in rows), 2), "effort": args.effort,
                          "provider": args.provider, "model": args.model if args.provider == "anthropic" else MODEL,
                          "note": "Output tokens are a guess; the real count is known only after a run."}))
    elif args.step == "run":
        if not args.yes:
            raise SystemExit("Refusing to send images or spend money without --yes")
        claude, google = args.provider == "anthropic", args.provider == "google"
        key = api_key("ANTHROPIC_API_KEY" if claude else "GEMINI_API_KEY" if google else "OPENAI_API_KEY")
        replies = args.out / args.replies
        replies.mkdir(parents=True, exist_ok=True)
        settings = {"provider": args.provider, "model": args.model if claude or google else MODEL, "effort": args.effort,
                    "service_tier": args.service_tier, "prompt": str(args.prompt.resolve().relative_to(ROOT)),
                    "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(), "findings": list(keys)}
        if args.thinking_budget:
            settings["thinking_budget"] = args.thinking_budget
        settings_file = replies / "run-settings.json"
        if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
            raise SystemExit(f"{replies} holds replies made with other settings; use another --replies folder")
        settings_file.write_text(json.dumps(settings, indent=1) + "\n")
        pending = pending_studies(replies, args.studies)
        print(json.dumps({"requested": len(args.studies), "already_answered": len(args.studies) - len(pending)}), flush=True)

        def save(study: str, response: dict) -> None:
            (replies / f"{study}.json").write_text(json.dumps(response, indent=1) + "\n")

        def answer(study: str) -> dict:
            started, uploaded, left = time.time(), {}, []
            if claude:
                if payload_bytes(packs / study) > CLAUDE_REQUEST_LIMIT:
                    uploaded = upload_pack(packs / study, key)
                try:
                    response = request_json(CLAUDE_URL, {**claude_headers(key), "Content-Type": "application/json"},
                                            json.dumps(build_claude_request(packs / study, args.effort, uploaded or None,
                                                                            model=args.model)).encode())
                    save(study, response)  # before the clean-up, so a failed delete cannot lose a paid answer
                finally:
                    if uploaded:
                        left = delete_files(list(uploaded.values()), key)
                usage, status = claude_usage_cost(response.get("usage") or {}, args.model), response.get("stop_reason")
            elif google:
                if payload_bytes(packs / study) > GEMINI_REQUEST_LIMIT:
                    uploaded = upload_gemini_pack(packs / study, key)
                try:
                    uris = {image: GEMINI_API_ROOT + name for image, name in uploaded.items()} or None
                    response = request_json(GEMINI_URL.format(model=args.model), {"x-goog-api-key": key, "Content-Type": "application/json"},
                                            json.dumps(build_gemini_request(packs / study, args.effort, uris, prompt=args.prompt,
                                                                            keys=keys, thinking_budget=args.thinking_budget)).encode())
                    save(study, response)  # before the clean-up, so a failed delete cannot lose a paid answer
                finally:
                    if uploaded:
                        left = delete_gemini_files(list(uploaded.values()), key)
                candidate = (response.get("candidates") or [{}])[0]
                usage, status = gemini_usage_cost(response.get("usageMetadata") or {}, args.model), candidate.get("finishReason")
            else:
                response = send(build_request(packs / study, args.effort, service_tier=args.service_tier,
                                              prompt=args.prompt, keys=keys), key)
                save(study, response)
                usage, status = usage_cost(response.get("usage") or {}, response.get("service_tier")), response.get("status")
            return {"study": study, "seconds": round(time.time() - started, 1), "status": status,
                    "uploaded_images": len(uploaded), "files_left": len(left), **usage}

        failed = {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(answer, study): study for study in pending}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as error:  # report the study and carry on with the rest
                    failed[futures[future]] = str(error)[:1000]
                    print(json.dumps({"study": futures[future], "error": failed[futures[future]]}), flush=True)
        if failed:
            raise SystemExit(f"{len(failed)} of {len(pending)} studies failed; running the same command again "
                             "re-sends only the studies without a saved reply")
    elif args.step == "cleanup":
        key, names = api_key("ANTHROPIC_API_KEY"), {study[-12:] for study in args.studies}
        ours = [item["id"] for item in list_files(key) if item["filename"].split("-")[0] in names]
        left = delete_files(ours, key)
        print(json.dumps({"uploaded_files_found": len(ours), "deleted": len(ours) - len(left), "left": len(left)}))
    else:
        rows = [parse_reply(study, reply_text(json.loads((args.out / args.replies / f"{study}.json").read_text())), keys)
                for study in args.studies]
        pd.DataFrame(rows).to_csv(labels_path(args.out, args.replies), index=False)
        print(f"Wrote {len(rows)} studies to {labels_path(args.out, args.replies)}")


if __name__ == "__main__":
    main()
