import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.inference import discover_models, heat_values, overlay, preprocess

ROOT = Path(__file__).resolve().parent.parent


def test_preprocess_shape_and_range():
    img = Image.fromarray(np.full((300, 200, 3), 255, np.uint8))
    x = preprocess(img, 256)
    assert x.shape == (1, 3, 256, 256) and x.dtype == np.float32
    assert x.min() >= 0 and x.max() <= 1


def test_heat_values_are_zero_at_normal_and_one_above_threshold():
    amap = np.array([[0.1, 0.1], [1.0, 2.0]], np.float32)
    v = heat_values(amap, pixel_p50=0.1, threshold=1.0)
    assert v[0, 0] == 0 and v[1, 1] == 1 and 0 < v[1, 0] < 1


def test_overlay_keeps_image_where_heat_is_zero():
    img = np.full((4, 4, 3), 120, np.uint8)
    out = overlay(img, np.zeros((4, 4), np.float32))
    assert (out == img).all()


@pytest.mark.skipif(not discover_models(ROOT / "models"), reason="no trained models in models/")
def test_inspect_endpoint():
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)
    category = client.get("/api/models").json()["default"]
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), np.uint8)).save(buf, format="PNG")
    r = client.post("/api/inspect", files={"file": ("x.png", buf.getvalue(), "image/png")},
                    data={"category": category})
    assert r.status_code == 200
    body = r.json()
    assert body["heatmap"].startswith("data:image/jpeg;base64,")
    assert body["is_defect"] == (body["score"] > body["threshold"])
    assert client.post("/api/inspect", files={"file": ("x.txt", b"no", "text/plain")},
                       data={"category": category}).status_code == 400
