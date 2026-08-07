#!/usr/bin/env python3
"""
2D cross-correlation between two independent 2D fields.

Unlike ``autocorrelation.py`` (which correlates the vessel mask from the main
analysis pipeline with itself), this module compares two separately uploaded
inputs — each may be a regular image (.png/.jpg/.tif) or a GeoJSON annotation
file. It is intentionally decoupled from the main pipeline/session state: it
never touches the image+GeoJSON pair used for vessel analysis.

GeoJSON inputs are rasterized to a binary mask sized to their own coordinate
bounds (no reference image required). Image inputs are read as grayscale.
Since the two fields may come from very different sources/sizes, the second
field is resampled onto the first field's grid before correlating.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from geojson_to_mask import geojson_to_mask

# Working-resolution cap for this utility. This is independent of the main
# pipeline (which processes at full resolution) — cross-correlating two
# arbitrary, possibly large uploads at full size is unnecessary and slow.
MAX_DIM = 512

_GEOJSON_TYPES = {
    "FeatureCollection", "Feature", "Polygon", "MultiPolygon", "Point",
    "MultiPoint", "LineString", "MultiLineString", "GeometryCollection",
}


def _looks_like_geojson(head: bytes) -> bool:
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text.lstrip().startswith("{"):
        return False
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            # head may have been truncated mid-JSON; only a soft check.
            return '"type"' in text and any(f'"{t}"' in text for t in _GEOJSON_TYPES)
        except Exception:
            return False
    return isinstance(obj, dict) and obj.get("type") in _GEOJSON_TYPES


def _normalize_to_feature_collection(obj: dict) -> dict:
    t = obj.get("type")
    if t == "FeatureCollection":
        return obj
    if t == "Feature":
        return {"type": "FeatureCollection", "features": [obj]}
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": obj, "properties": {}}]}


def field_from_upload(raw: bytes, filename: str = "") -> np.ndarray:
    """
    Turn raw uploaded bytes (image OR GeoJSON) into a 2D float64 field.

    GeoJSON -> rasterized binary mask (1.0 inside polygons, 0.0 outside),
               canvas auto-sized to the geometry's own coordinate bounds.
    Image   -> grayscale, scaled to [0, 1].
    """
    is_geojson = filename.lower().endswith((".geojson", ".json")) or _looks_like_geojson(raw[:4096])

    if is_geojson:
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not parse '{filename}' as GeoJSON: {exc}") from exc
        fc = _normalize_to_feature_collection(obj)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False, mode="w", encoding="utf-8") as tmp:
                json.dump(fc, tmp)
                tmp_path = Path(tmp.name)
            mask, _ = geojson_to_mask(tmp_path, shape_hw=None)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            raise ValueError(
                f"'{filename}' produced a degenerate {mask.shape} canvas — "
                "check that its coordinates are in pixel-like units, not tiny lon/lat degrees."
            )
        return mask.astype(np.float64)

    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode '{filename}' as an image or GeoJSON.")
    return img.astype(np.float64) / 255.0


def _downscale_to_max_dim(field: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = field.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        field = cv2.resize(
            field, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return field


def compute_cross_correlation_2d(
    field_a: np.ndarray,
    field_b: np.ndarray,
    crop_frac: float = 0.5,
) -> dict[str, Any]:
    """
    Normalized 2D cross-correlation of two fields via FFT.

    Field B is resampled onto field A's grid first (the two uploads need not
    share a size or coordinate system). Both are mean-subtracted and
    zero-padded to 2x size to avoid circular-wraparound artifacts, matching
    the convention used for the autocorrelation panel.

    Returns a dict with the cropped, centered 2D cross-correlation map,
    its radial average, and the (dx, dy) offset + value at the correlation
    peak — the peak location indicates the best-aligning relative shift
    between the two fields.
    """
    a = _downscale_to_max_dim(np.asarray(field_a, dtype=np.float64), MAX_DIM)
    b = _downscale_to_max_dim(np.asarray(field_b, dtype=np.float64), MAX_DIM)
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    fa = a - a.mean()
    fb = b_resized - b_resized.mean()

    h, w = fa.shape
    fh, fw = 2 * h, 2 * w
    Fa = np.fft.rfft2(fa, s=(fh, fw))
    Fb = np.fft.rfft2(fb, s=(fh, fw))
    cross = Fa * np.conj(Fb)
    xcorr = np.fft.irfft2(cross, s=(fh, fw))
    xcorr = np.fft.fftshift(xcorr)

    norm = float(np.sqrt(np.sum(fa * fa) * np.sum(fb * fb)))
    if norm > 0:
        xcorr = xcorr / norm

    cy, cx = fh // 2, fw // 2
    half = max(4, int(min(h, w) * crop_frac))
    y0, y1 = max(0, cy - half), min(fh, cy + half)
    x0, x1 = max(0, cx - half), min(fw, cx + half)
    xcorr_crop = xcorr[y0:y1, x0:x1]

    cyc = xcorr_crop.shape[0] // 2
    cxc = xcorr_crop.shape[1] // 2

    peak_idx = np.unravel_index(int(np.argmax(xcorr_crop)), xcorr_crop.shape)
    peak_dy = int(peak_idx[0] - cyc)
    peak_dx = int(peak_idx[1] - cxc)
    peak_value = float(xcorr_crop[peak_idx])

    yy, xx = np.indices(xcorr_crop.shape)
    r = np.sqrt((yy - cyc) ** 2 + (xx - cxc) ** 2)
    r_int = r.astype(int)
    max_r = int(r_int.max())
    radial_sum = np.bincount(r_int.ravel(), weights=xcorr_crop.ravel(), minlength=max_r + 1)
    radial_sumsq = np.bincount(r_int.ravel(), weights=(xcorr_crop.ravel() ** 2), minlength=max_r + 1)
    radial_count = np.bincount(r_int.ravel(), minlength=max_r + 1)
    radial_profile = radial_sum / np.maximum(radial_count, 1)
    radial_var = radial_sumsq / np.maximum(radial_count, 1) - radial_profile ** 2
    radial_std = np.sqrt(np.maximum(radial_var, 0))
    radius_px = np.arange(len(radial_profile), dtype=np.float64)

    return {
        "xcorr2d": xcorr_crop,
        "half_size": half,
        "radius": radius_px,
        "radial_profile": radial_profile,
        "radial_std": radial_std,
        "peak_dx": peak_dx,
        "peak_dy": peak_dy,
        "peak_value": peak_value,
        "working_shape": (h, w),
    }


def cross_correlation_for_frontend(
    field_a: np.ndarray,
    field_b: np.ndarray,
    crop_frac: float = 0.5,
    dpi: int = 110,
    npz_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute the 2D cross-correlation and package it for inline React display.

    If ``npz_path`` is given, also writes the full-precision underlying
    arrays (2D cross-correlation map, radius, radial profile, radial std)
    plus scalar metadata to a .npz file at that path, so the raw data can
    be reloaded later (``np.load(path)``) to reproduce or restyle the plots
    without rerunning the computation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import INK_DIM, diverging_norm, style_dark
    from correlation_stats import half_max_distance

    result = compute_cross_correlation_2d(field_a, field_b, crop_frac=crop_frac)
    xc = result["xcorr2d"]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=dpi)
    extent = (-xc.shape[1] // 2, xc.shape[1] // 2, -xc.shape[0] // 2, xc.shape[0] // 2)
    # Diverging colormap centered at 0, matching the autocorrelation panel's
    # color scheme — cross-correlation is meaningfully negative (anti-correlated
    # structure), so a single-ended colormap would hide that. A plain linear
    # scale is dominated by the correlation peak and washes weaker structure
    # out near-white, so we use a symmetric log-ish norm instead.
    norm, vmax = diverging_norm(xc)
    im = ax.imshow(xc, cmap="RdBu_r", norm=norm, extent=extent, origin="lower")
    ax.axhline(0, color=INK_DIM, lw=0.5, alpha=0.5)
    ax.axvline(0, color=INK_DIM, lw=0.5, alpha=0.5)
    ax.plot(result["peak_dx"], result["peak_dy"], marker="x", color="black",
            markersize=9, markeredgewidth=2)
    ax.set_xlabel("\u0394x (px)")
    ax.set_ylabel("\u0394y (px)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cross-correlation", fontsize=9)
    style_dark(ax, cbar)
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    payload = base64.b64encode(buf.read()).decode("ascii")

    r_half, radial_peak_value = half_max_distance(result["radius"], result["radial_profile"])

    if npz_path is not None:
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            xcorr2d=result["xcorr2d"],
            radius=result["radius"],
            radial_profile=result["radial_profile"],
            radial_std=result["radial_std"],
            peak_dx=result["peak_dx"],
            peak_dy=result["peak_dy"],
            peak_value=result["peak_value"],
            half_max_distance=np.nan if r_half is None else r_half,
            radial_peak_value=radial_peak_value,
            working_shape=np.array(result["working_shape"]),
        )

    return {
        "png_data_url": f"data:image/png;base64,{payload}",
        "radius": [float(v) for v in result["radius"][1:]],
        "profile": [float(v) for v in result["radial_profile"][1:]],
        "std": [float(v) for v in result["radial_std"][1:]],
        "peak_dx": result["peak_dx"],
        "peak_dy": result["peak_dy"],
        "peak_value": result["peak_value"],
        "half_max_distance": r_half,
        "radial_peak_value": radial_peak_value,
        "working_shape": {"height": int(result["working_shape"][0]), "width": int(result["working_shape"][1])},
        "npz_path": str(npz_path) if npz_path is not None else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a", type=Path, help="First input: image or .geojson")
    parser.add_argument("file_b", type=Path, help="Second input: image or .geojson")
    parser.add_argument("-o", "--output", type=Path, default=Path("cross_correlation_2d.png"))
    args = parser.parse_args()

    field_a = field_from_upload(args.file_a.read_bytes(), args.file_a.name)
    field_b = field_from_upload(args.file_b.read_bytes(), args.file_b.name)
    out = cross_correlation_for_frontend(field_a, field_b)

    png_bytes = base64.b64decode(out["png_data_url"].split(",", 1)[1])
    args.output.write_bytes(png_bytes)
    print(f"Wrote {args.output}")
    print(f"Peak offset: dx={out['peak_dx']}, dy={out['peak_dy']}, value={out['peak_value']:.4f}")
