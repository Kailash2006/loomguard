"""ONNX Runtime inference for LoomGuard models exported by the Kaggle notebook."""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Inferno-like colour ramp (dark → madder red → turmeric → near white), as a 256-entry LUT.
_ANCHORS = np.array([
    [0.00, 0, 0, 4], [0.25, 87, 16, 110], [0.50, 188, 55, 84],
    [0.75, 249, 142, 9], [1.00, 252, 255, 164],
], dtype=np.float32)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), _ANCHORS[:, 0], _ANCHORS[:, i]) for i in (1, 2, 3)], -1)
_LUT = _LUT.astype(np.uint8)


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """PIL image → float32 NCHW in [0, 1]. Normalisation happens inside the ONNX graph."""
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def heat_values(amap: np.ndarray, pixel_p50: float, threshold: float) -> np.ndarray:
    """Map raw anomaly values to [0, 1]: 0 at the typical normal level, 1 at 1.5x the decision threshold."""
    hi = 1.5 * threshold
    return np.clip((amap - pixel_p50) / max(hi - pixel_p50, 1e-12), 0.0, 1.0)


def overlay(img_rgb: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Blend the colour-mapped heat over the image; low heat stays transparent."""
    heat = _LUT[(v * 255).astype(np.uint8)].astype(np.float32)
    a = (0.85 * v)[..., None]
    return ((1 - a) * img_rgb.astype(np.float32) + a * heat).clip(0, 255).astype(np.uint8)


def to_data_url(arr: np.ndarray, quality: int = 88) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Where model files come from when they are not on disk (e.g. on Vercel, where the bundle excludes them).
MODEL_URL = os.environ.get("LOOMGUARD_MODEL_URL",
                           "https://raw.githubusercontent.com/Kailash2006/loomguard/main/models")
CACHE_DIR = os.environ.get("LOOMGUARD_CACHE", os.path.join(tempfile.gettempdir(), "loomguard-models"))


def model_file(model_dir: Path, name: str) -> Path:
    """Return a local path to a model file, downloading it from MODEL_URL once if it isn't on disk."""
    local = Path(model_dir) / name
    if local.is_file():
        return local
    cached = Path(CACHE_DIR) / Path(model_dir).name / name
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(cached.suffix + ".part")
        with urllib.request.urlopen(f"{MODEL_URL}/{Path(model_dir).name}/{name}", timeout=120) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        tmp.replace(cached)
    return cached


def read_meta(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "meta.json").read_text())


@dataclass
class Inspection:
    category: str
    score: float
    threshold: float
    severity: float          # score / threshold; above 1.0 means defect
    is_defect: bool
    affected_area_pct: float
    bbox: list[int] | None   # [x0, y0, x1, y1] in display pixels, around the hottest region
    latency_ms: float
    heatmap: str             # data URL of the overlay image

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class LoomGuardModel:
    def __init__(self, model_dir: Path, threads: int | None = None):
        self.dir = Path(model_dir)
        self.meta = read_meta(self.dir)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        onnx_path = model_file(self.dir, self.meta.get("onnx_file", "stfpm.onnx"))
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        # PatchCore models take their memory bank as a second input.
        bank_file = self.meta.get("bank_file")
        self.extra = {"bank": np.load(model_file(self.dir, bank_file)).astype(np.float32)} if bank_file else {}
        self.model_name = self.meta.get("model", "stfpm-resnet18")
        self.size = int(self.meta["img_size"])
        self.category = self.meta["category"]

    def inspect(self, img: Image.Image, display_size: int = 512) -> Inspection:
        x = preprocess(img, self.size)
        t0 = time.perf_counter()
        amap, score = self.session.run(None, {"image": x, **self.extra})
        latency = (time.perf_counter() - t0) * 1000

        amap, score = amap[0, 0], float(score[0])
        thr = float(self.meta["threshold"])
        v = heat_values(amap, float(self.meta["pixel_p50"]), thr)

        # Pixels above the normal-image 99.9th percentile count as affected.
        hot = amap > float(self.meta["pixel_max"])
        area = float(hot.mean() * 100)
        bbox = None
        if score > thr and hot.any():
            ys, xs = np.nonzero(hot)
            k = display_size / self.size
            bbox = [int(xs.min() * k), int(ys.min() * k), int((xs.max() + 1) * k), int((ys.max() + 1) * k)]

        disp = np.asarray(img.convert("RGB").resize((display_size, display_size), Image.BILINEAR))
        v_disp = np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize(
            (display_size, display_size), Image.BILINEAR), dtype=np.float32) / 255.0
        return Inspection(
            category=self.category, score=score, threshold=thr, severity=score / thr,
            is_defect=score > thr, affected_area_pct=round(area, 2), bbox=bbox,
            latency_ms=round(latency, 1), heatmap=to_data_url(overlay(disp, v_disp)),
        )


def discover_models(models_root: Path) -> dict[str, Path]:
    root = Path(models_root)
    if not root.is_dir():
        return {}
    # A category is available when its meta.json exists; weight files are fetched on demand if missing.
    return {p.name: p for p in sorted(root.iterdir()) if (p / "meta.json").is_file()}
