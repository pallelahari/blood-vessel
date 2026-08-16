#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, Response, send_from_directory, abort
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline import SessionState
from cross_correlation import field_from_upload, cross_correlation_for_frontend

OUTPUT_ROOT = ROOT / "analysis_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)
STATE = SessionState(output_root=OUTPUT_ROOT)

# Shared progress log
_progress_lock = threading.Lock()
_progress_log: list[str] = []
_progress_pct = 0.0
_progress_stage = ""
_processing = False

def push_log(msg: str) -> None:
    with _progress_lock:
        _progress_log.append(msg)

def push_progress(pct: float, stage: str) -> None:
    global _progress_pct, _progress_stage
    with _progress_lock:
        _progress_pct = pct
        _progress_stage = stage

# Monkey-patch pipeline log/progress hooks to also push to frontend
import backend.pipeline as _pl
_pl._push_log = push_log
_pl._push_progress = push_progress


@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True})


@app.get("/api/progress")
def progress() -> Any:
    with _progress_lock:
        logs = list(_progress_log)
        pct = _progress_pct
        stage = _progress_stage
    return jsonify({"logs": logs, "processing": _processing, "progress": pct, "stage": stage})


@app.post("/api/process")
def process() -> Any:
    global _processing, _progress_pct, _progress_stage
    if "image_file" not in request.files or "geojson_file" not in request.files:
        return jsonify({"error": "image_file and geojson_file are required"}), 400

    image_bytes = request.files["image_file"].read()
    geojson_bytes = request.files["geojson_file"].read()
    if not image_bytes or not geojson_bytes:
        return jsonify({"error": "Empty file(s)"}), 400

    with _progress_lock:
        _progress_log.clear()
        _progress_pct = 0.0
        _progress_stage = ""
    _processing = True
    try:
        image_name = request.files["image_file"].filename or "image"
        out = STATE.process(image_bytes=image_bytes, geojson_bytes=geojson_bytes, image_name=image_name)
    except Exception as exc:
        _processing = False
        return jsonify({"error": str(exc)}), 400
    _processing = False
    return jsonify(out)


@app.post("/api/crop-process")
def crop_process() -> Any:
    """
    Region-select analysis. Re-uses the main analysis already held in
    STATE (see SessionState.region_view) instead of re-running the
    segmentation pipeline on a cropped sub-image — that would reassign
    blob IDs from scratch and could distort the shape/measurements of any
    vessel cut by the crop boundary, causing the region view to disagree
    with the main image. No image/GeoJSON re-upload is needed: the crop is
    just a window into the analysis that's already been computed.
    """
    try:
        x1 = int(request.form.get("x1", 0))
        y1 = int(request.form.get("y1", 0))
        x2 = int(request.form.get("x2", 0))
        y2 = int(request.form.get("y2", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid crop coordinates"}), 400

    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "Invalid crop region — x2 must be > x1 and y2 > y1"}), 400

    try:
        out = STATE.region_view(x1=x1, y1=y1, x2=x2, y2=y2)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(out)


@app.get("/api/blob-info")
def blob_info() -> Any:
    x = int(request.args.get("x", "-1"))
    y = int(request.args.get("y", "-1"))
    try:
        out = STATE.get_blob_info(x=x, y=y)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(out)


@app.get("/api/output-file/<path:relpath>")
def output_file(relpath: str) -> Any:
    root = OUTPUT_ROOT.resolve()
    full = (root / relpath).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        abort(403)
    if not full.is_file():
        abort(404)
    return send_from_directory(str(full.parent), full.name)


@app.post("/api/cross-correlate")
def cross_correlate() -> Any:
    """
    Cross-correlate two independently uploaded fields (image and/or GeoJSON).
    Entirely separate from /api/process — does not touch or require a main
    analysis session, and does not use the image/GeoJSON pair uploaded there.
    """
    if "file_a" not in request.files or "file_b" not in request.files:
        return jsonify({"error": "file_a and file_b are required"}), 400

    file_a = request.files["file_a"]
    file_b = request.files["file_b"]
    raw_a = file_a.read()
    raw_b = file_b.read()
    if not raw_a or not raw_b:
        return jsonify({"error": "Empty file(s)"}), 400

    try:
        field_a = field_from_upload(raw_a, file_a.filename or "")
        field_b = field_from_upload(raw_b, file_b.filename or "")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem_a = re.sub(r"[^\w\-]", "_", Path(file_a.filename or "a").stem)[:40]
        stem_b = re.sub(r"[^\w\-]", "_", Path(file_b.filename or "b").stem)[:40]
        npz_name = f"cross_correlation-{stem_a}-{stem_b}-{stamp}.npz"
        npz_rel = f"cross_correlation/{npz_name}"
        out = cross_correlation_for_frontend(field_a, field_b, npz_path=OUTPUT_ROOT / npz_rel)
        out.pop("npz_path", None)  # absolute server path — not useful to the client
        out["npz_download_path"] = npz_rel
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(out)


@app.post("/api/finalize-branches")
def finalize_branches() -> Any:
    payload = request.get_json(silent=True) or {}
    raw_points = payload.get("points", [])
    points: list[tuple[int, int]] = []
    for p in raw_points:
        if isinstance(p, dict):
            points.append((int(p.get("x", -1)), int(p.get("y", -1))))
    try:
        out = STATE.finalize_branches(points=points)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(out)


if __name__ == "__main__":
    import os
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=True, threaded=True)
