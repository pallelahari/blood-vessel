#!/usr/bin/env python3
"""
2D spatial autocorrelation of a binary/grayscale image (e.g. a GeoJSON-derived
vessel mask), plus a radially-averaged 1D profile — analogous to the classic
"image + 2D ACF + radial decay" panel used for granulation/texture analysis.

Produces a single self-contained, interactive HTML report (Plotly, JS from CDN)
so it can be opened directly in a browser without any server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_autocorrelation_2d(
    image: np.ndarray,
    pixel_size_um: float | None = None,
    crop_frac: float = 0.5,
) -> dict[str, Any]:
    """
    Compute the normalized 2D spatial autocorrelation function (ACF) of an
    image via FFT, plus its radially-averaged 1D profile.

    Parameters
    ----------
    image
        2D array (binary mask or grayscale). Any dtype; will be cast to float.
    pixel_size_um
        If provided, distances are reported in microns; otherwise pixels.
    crop_frac
        Fraction of min(height, width) used as the half-window kept around
        the zero-lag center of the ACF (the far field is mostly noise/padding).

    Returns
    -------
    dict with:
      acf2d            : cropped, centered 2D ACF (zero-lag = 1.0)
      half_size        : half-window size (px) of the crop, for axis extents
      radius           : 1D array of radial distances (px or µm)
      radial_profile   : mean ACF value at each radius
      radial_std       : standard deviation of ACF values at each radius
      unit_label       : "px" or "µm"
    """
    img = np.asarray(image, dtype=np.float64)
    img = img - img.mean()

    h, w = img.shape
    # Zero-pad to 2x size to avoid circular-wraparound artifacts in the FFT.
    fh, fw = 2 * h, 2 * w

    F = np.fft.rfft2(img, s=(fh, fw))
    power = F * np.conj(F)
    acf = np.fft.irfft2(power, s=(fh, fw))
    acf = np.fft.fftshift(acf)

    zero_lag = acf.max()
    if zero_lag > 0:
        acf = acf / zero_lag

    cy, cx = fh // 2, fw // 2
    half = max(4, int(min(h, w) * crop_frac))
    y0, y1 = max(0, cy - half), min(fh, cy + half)
    x0, x1 = max(0, cx - half), min(fw, cx + half)
    acf_crop = acf[y0:y1, x0:x1]

    # Radial average around the crop's own center.
    cyc = acf_crop.shape[0] // 2
    cxc = acf_crop.shape[1] // 2
    yy, xx = np.indices(acf_crop.shape)
    r = np.sqrt((yy - cyc) ** 2 + (xx - cxc) ** 2)
    r_int = r.astype(int)
    max_r = int(r_int.max())

    radial_sum = np.bincount(r_int.ravel(), weights=acf_crop.ravel(), minlength=max_r + 1)
    radial_sumsq = np.bincount(r_int.ravel(), weights=(acf_crop.ravel() ** 2), minlength=max_r + 1)
    radial_count = np.bincount(r_int.ravel(), minlength=max_r + 1)
    radial_profile = radial_sum / np.maximum(radial_count, 1)
    radial_var = radial_sumsq / np.maximum(radial_count, 1) - radial_profile ** 2
    radial_std = np.sqrt(np.maximum(radial_var, 0))
    radius_px = np.arange(len(radial_profile), dtype=np.float64)

    if pixel_size_um:
        radius = radius_px * pixel_size_um
        unit_label = "\u00b5m"
    else:
        radius = radius_px
        unit_label = "px"

    return {
        "acf2d": acf_crop,
        "half_size": half,
        "radius": radius,
        "radial_profile": radial_profile,
        "radial_std": radial_std,
        "unit_label": unit_label,
    }


def render_autocorrelation_html(
    result: dict[str, Any],
    out_path: str | Path,
    title: str = "2D Spatial Autocorrelation",
) -> Path:
    """Write a standalone, interactive HTML report (heatmap + radial profile)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    acf2d = result["acf2d"]
    half = result["half_size"]
    radius = result["radius"]
    profile = result["radial_profile"]
    unit_label = result["unit_label"]

    axis_vals = np.arange(-acf2d.shape[1] // 2, acf2d.shape[1] // 2)
    axis_vals_y = np.arange(-acf2d.shape[0] // 2, acf2d.shape[0] // 2)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("2D Autocorrelation Map", "Radial Autocorrelation Profile"),
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=acf2d,
            x=axis_vals,
            y=axis_vals_y,
            colorscale="Turbo",
            zmin=float(np.nanmin(acf2d)),
            zmax=1.0,
            colorbar=dict(title="ACF", x=0.44, len=0.9),
            hovertemplate=f"\u0394x=%{{x}} {unit_label}<br>\u0394y=%{{y}} {unit_label}<br>ACF=%{{z:.3f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=radius[1:],
            y=profile[1:],
            mode="lines+markers",
            line=dict(color="#e63946", width=2),
            marker=dict(size=4),
            name="Radial ACF",
            hovertemplate=f"r=%{{x:.2f}} {unit_label}<br>ACF=%{{y:.3f}}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text=f"\u0394x ({unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"\u0394y ({unit_label})", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text=f"Distance ({unit_label})", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=2)

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1150,
        height=540,
        showlegend=False,
        margin=dict(t=80, l=60, r=30, b=60),
    )

    out_path = Path(out_path)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path


def autocorrelation_for_frontend(
    image: np.ndarray,
    pixel_size_um: float | None = None,
    crop_frac: float = 0.5,
    dpi: int = 110,
    npz_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute the 2D ACF and package it for inline display in a web frontend:
    a base64 PNG heatmap (diverging RdBu_r colormap with a labeled colorbar,
    matching the cross-correlation panel's color scheme) plus the radial
    profile (with standard deviation) as plain lists for a JS charting
    library (e.g. recharts) to plot.

    If ``npz_path`` is given, also writes the full-precision underlying
    arrays (2D ACF map, radius, radial profile, radial std) plus scalar
    metadata to a .npz file at that path, so the raw data can be reloaded
    later (``np.load(path)``) to reproduce or restyle the plots without
    rerunning the pipeline.

    Returns
    -------
    dict with:
      png_data_url   : "data:image/png;base64,..." heatmap of the 2D ACF
      radius         : list[float] radial distances
      profile        : list[float] radially-averaged ACF at each distance
      std            : list[float] standard deviation of ACF at each distance
      unit_label     : "px" or "µm"
      npz_path       : str(npz_path) if one was written, else None
    """
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import INK_DIM, diverging_norm, style_dark
    from correlation_stats import correlation_length_1e, half_max_distance

    result = compute_autocorrelation_2d(image, pixel_size_um=pixel_size_um, crop_frac=crop_frac)
    acf2d = result["acf2d"]
    half = result["half_size"]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=dpi)
    extent = (-acf2d.shape[1] // 2, acf2d.shape[1] // 2, -acf2d.shape[0] // 2, acf2d.shape[0] // 2)
    # Diverging colormap centered at 0, matching the cross-correlation panel's
    # color scheme so the two are visually consistent. ACF is normalized so
    # zero-lag = 1.0, but off-center lags can legitimately dip negative for
    # periodic/anti-correlated structure, so a symmetric diverging scale
    # represents that correctly. A plain linear scale is dominated by that
    # single zero-lag spike and washes everything else out near-white, so we
    # use a symmetric log-ish norm instead to keep off-peak structure visible.
    norm, vmax = diverging_norm(acf2d)
    im = ax.imshow(
        acf2d,
        cmap="RdBu_r",
        norm=norm,
        extent=extent,
        origin="lower",
    )
    ax.axhline(0, color=INK_DIM, lw=0.5, alpha=0.5)
    ax.axvline(0, color=INK_DIM, lw=0.5, alpha=0.5)
    ax.set_xlabel(f"\u0394x ({result['unit_label']})")
    ax.set_ylabel(f"\u0394y ({result['unit_label']})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ACF", fontsize=9)
    style_dark(ax, cbar)
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    payload = base64.b64encode(buf.read()).decode("ascii")

    correlation_length = correlation_length_1e(result["radius"], result["radial_profile"])
    r_half, peak_value = half_max_distance(result["radius"], result["radial_profile"])

    if npz_path is not None:
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            acf2d=result["acf2d"],
            radius=result["radius"],
            radial_profile=result["radial_profile"],
            radial_std=result["radial_std"],
            unit_label=result["unit_label"],
            correlation_length=np.nan if correlation_length is None else correlation_length,
            correlation_threshold=1.0 / float(np.e),
            half_max_distance=np.nan if r_half is None else r_half,
            peak_value=peak_value,
        )

    return {
        "png_data_url": f"data:image/png;base64,{payload}",
        "radius": [float(v) for v in result["radius"][1:]],
        "profile": [float(v) for v in result["radial_profile"][1:]],
        "std": [float(v) for v in result["radial_std"][1:]],
        "unit_label": result["unit_label"],
        "correlation_length": correlation_length,
        "correlation_threshold": 1.0 / float(np.e),
        "half_max_distance": r_half,
        "peak_value": peak_value,
        "npz_path": str(npz_path) if npz_path is not None else None,
    }


def autocorrelation_report(
    image: np.ndarray,
    out_path: str | Path,
    pixel_size_um: float | None = None,
    crop_frac: float = 0.5,
    title: str = "2D Spatial Autocorrelation",
) -> Path:
    """Convenience wrapper: compute ACF + write HTML report in one call."""
    result = compute_autocorrelation_2d(image, pixel_size_um=pixel_size_um, crop_frac=crop_frac)
    return render_autocorrelation_html(result, out_path, title=title)


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a binary mask / grayscale image")
    parser.add_argument("-o", "--output", type=Path, default=Path("autocorrelation_2d.html"))
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--crop-frac", type=float, default=0.5)
    args = parser.parse_args()

    img = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    out = autocorrelation_report(
        img, args.output, pixel_size_um=args.pixel_size_um, crop_frac=args.crop_frac
    )
    print(f"Wrote {out}")
