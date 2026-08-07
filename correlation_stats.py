"""Characteristic-distance metrics for radial correlation profiles.

Shared by autocorrelation.py and cross_correlation.py so both radial
profiles (ACF and cross-correlation) report these consistently.
"""
from __future__ import annotations

import numpy as np


def correlation_length_1e(radius, profile, threshold: float = 1.0 / np.e) -> float | None:
    """
    Distance from r=0 where the profile first decays to ``threshold``
    (default 1/e) of its zero-lag value. Assumes ``profile[0]`` is the peak,
    which holds for a normalized autocorrelation function.
    """
    for i in range(1, len(profile)):
        if profile[i] <= threshold:
            r0, r1 = float(radius[i - 1]), float(radius[i])
            p0, p1 = float(profile[i - 1]), float(profile[i])
            if p0 == p1:
                return r0
            frac = (p0 - threshold) / (p0 - p1)
            return r0 + frac * (r1 - r0)
    return None


def half_max_distance(radius, profile) -> tuple[float | None, float]:
    """
    The distance, measured out from wherever the profile's peak actually
    falls, at which the curve first decays to half the peak value — i.e.
    "where does the y-axis become half of the peak, on the x-axis."

    Locating the peak by argmax (rather than assuming it's at r=0) keeps
    this correct even for a cross-correlation radial profile, whose peak
    need not sit at zero lag if the two fields are spatially offset.

    Returns (distance_or_None, peak_value).
    """
    profile = np.asarray(profile, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    if len(profile) == 0:
        return None, 0.0

    peak_idx = int(np.argmax(profile))
    peak = float(profile[peak_idx])
    if peak <= 0:
        return None, peak

    threshold = peak / 2.0
    for i in range(peak_idx + 1, len(profile)):
        if profile[i] <= threshold:
            r0, r1 = float(radius[i - 1]), float(radius[i])
            p0, p1 = float(profile[i - 1]), float(profile[i])
            if p0 == p1:
                return r0, peak
            frac = (p0 - threshold) / (p0 - p1)
            return r0 + frac * (r1 - r0), peak
    return None, peak
