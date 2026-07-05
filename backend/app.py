#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline import SessionState

OUTPUT_ROOT = ROOT / "analysis_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)
STATE = SessionState(output_root=OUTPUT_ROOT)

# Shared progress log
_progress_lock = threading.Lock()
_progress_log: list[str] = []
_processing = False

def push_log(msg: str) -> None:
    with _progress_lock:
        _progress_log.append(msg)

# Monkey-patch pipeline log function to also push to frontend
import backend.pipeline as _pl
_pl._push_log = push_log


@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True})


@app.get("/api/progress")
def progress() -> Any:
    with _progress_lock:
        logs = list(_progress_log)
    return jsonify({"logs": logs, "processing": _processing})


@app.post("/api/process")
def process() -> Any:
    global _processing
    if "image_file" not in request.files or "geojson_file" not in request.files:
        return jsonify({"error": "image_file and geojson_file are required"}), 400

    image_bytes = request.files["image_file"].read()
    geojson_bytes = request.files["geojson_file"].read()
    if not image_bytes or not geojson_bytes:
        return jsonify({"error": "Empty file(s)"}), 400

    with _progress_lock:
        _progress_log.clear()
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
    global _processing
    if "image_file" not in request.files or "geojson_file" not in request.files:
        return jsonify({"error": "image_file and geojson_file are required"}), 400

    image_bytes  = request.files["image_file"].read()
    geojson_bytes = request.files["geojson_file"].read()
    image_name   = request.files["image_file"].filename or "image"

    try:
        x1 = int(request.form.get("x1", 0))
        y1 = int(request.form.get("y1", 0))
        x2 = int(request.form.get("x2", 0))
        y2 = int(request.form.get("y2", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid crop coordinates"}), 400

    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "Invalid crop region — x2 must be > x1 and y2 > y1"}), 400

    with _progress_lock:
        _progress_log.clear()
    _processing = True
    try:
        out = STATE.process(
            image_bytes=image_bytes,
            geojson_bytes=geojson_bytes,
            image_name=f"{image_name}_crop_{x1}_{y1}_{x2}_{y2}",
            crop=(x1, y1, x2, y2),
        )
    except Exception as exc:
        _processing = False
        return jsonify({"error": str(exc)}), 400
    _processing = False
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
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True)
