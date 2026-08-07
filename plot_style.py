"""Shared matplotlib styling for the inline correlation heatmaps.

Both autocorrelation.py and cross_correlation.py render their 2D maps as
base64 PNGs embedded directly in a dark-themed React page. Two things needed
fixing here:

1. A plain linear, symmetric-about-zero normalization is dominated by the
   single sharp zero-lag peak (value ~1.0), which compresses every other
   pixel toward the colormap's white center — the map looks almost blank.
   `diverging_norm()` uses a symmetric log-ish scale so small off-peak
   structure still gets real color contrast while the peak stays saturated.

2. The PNGs are saved with a transparent figure background so they blend
   into the page, but matplotlib's default text color is black — invisible
   against the app's dark background outside the (light) heatmap panel.
   `style_dark()` recolors labels/ticks/colorbar text to a light ink color.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import SymLogNorm

INK = "#E7EAF2"
INK_DIM = "#8A93AA"


def diverging_norm(arr: np.ndarray) -> tuple[SymLogNorm, float]:
    vmax = float(np.nanmax(np.abs(arr))) or 1.0
    linthresh = max(vmax * 0.02, 1e-4)
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    return norm, vmax


def style_dark(ax, cbar=None) -> None:
    for spine in ax.spines.values():
        spine.set_color(INK_DIM)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if cbar is not None:
        cbar.ax.tick_params(colors=INK, labelsize=8)
        cbar.ax.yaxis.label.set_color(INK)
        cbar.outline.set_edgecolor(INK_DIM)
