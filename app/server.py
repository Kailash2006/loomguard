"""LoomGuard inspection service: FastAPI + ONNX Runtime, CPU only.

Run:  uvicorn app.server:app --port 8000
"""
from __future__ import annotations

import io
import os
import time
from collections import deque
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import LoomGuardModel, discover_models

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("LOOMGUARD_MODELS", ROOT / "models"))
SAMPLES_DIR = Path(os.environ.get("LOOMGUARD_SAMPLES", ROOT / "samples"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 15 * 2**20

app = FastAPI(title="LoomGuard", description="Unsupervised fabric defect inspection", version="1.0.0")

_paths = discover_models(MODELS_DIR)
_loaded: dict[str, LoomGuardModel] = {}
_load_lock = Lock()
_log: deque[dict] = deque(maxlen=500)


def get_model(category: str) -> LoomGuardModel:
    if category not in _paths:
        raise HTTPException(404, f"No model for '{category}'. Available: {sorted(_paths) or 'none'}")
    with _load_lock:
        if category not in _loaded:
            _loaded[category] = LoomGuardModel(_paths[category])
    return _loaded[category]


@app.get("/api/models")
def list_models():
    out = []
    for name, path in _paths.items():
        meta = get_model(name).meta
        m = meta.get("metrics", {})
        out.append({"category": name, "threshold": meta["threshold"], "model": meta.get("model", "stfpm-resnet18"),
                    "image_auroc": m.get("image_auroc"), "pixel_auroc": m.get("pixel_auroc"),
                    "cpu_latency_ms": m.get("cpu_latency_ms")})
    # Open the demo on the category where the deployed model performs best (or LOOMGUARD_DEFAULT if set).
    default = os.environ.get("LOOMGUARD_DEFAULT")
    if default not in _paths:
        default = max(out, key=lambda m: get_model(m["category"]).meta.get("metrics", {}).get("f1", 0))["category"] if out else None
    return {"models": out, "default": default}


@app.get("/api/samples")
def list_samples(category: str):
    folder = SAMPLES_DIR / category
    if not folder.is_dir():
        return {"samples": []}
    files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return {"samples": [{"name": f, "url": f"/samples/{category}/{f}",
                         "kind": f.split("_")[0]} for f in files]}


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), category: str = Form("carpet"), source: str = Form("upload")):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "This file is not a readable image. Use JPG or PNG.")
    result = get_model(category).inspect(img).to_dict()
    _log.append({"time": time.time(), "category": category, "source": source,
                 "is_defect": result["is_defect"], "severity": result["severity"],
                 "latency_ms": result["latency_ms"]})
    return result


@app.get("/api/stats")
def stats():
    n = len(_log)
    defects = sum(e["is_defect"] for e in _log)
    return {"inspected": n, "defects": defects,
            "defect_rate": defects / n if n else 0.0,
            "mean_latency_ms": sum(e["latency_ms"] for e in _log) / n if n else 0.0}


@app.delete("/api/stats")
def reset_stats():
    _log.clear()
    return {"ok": True}


@app.get("/healthz")
def health():
    return {"status": "ok", "models": sorted(_paths)}


if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
