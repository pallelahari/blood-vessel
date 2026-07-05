#!/usr/bin/env python3
"""
Build a binary vessel mask from a QuPath-exported GeoJSON (FeatureCollection).

Supports Polygon and MultiPolygon. Rasterizes exterior rings and subtracts
interior rings (holes). Mask axes are image-style: column = x, row = y.

Example:
  python geojson_to_mask.py "annotation.geojson" -o mask.png
  python geojson_to_mask.py "annotation.geojson" --reference-image slide.tif -o mask.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple

import cv2
import numpy as np


def _iter_xy_pairs(ring: list) -> Iterator[Tuple[float, float]]:
    for pt in ring:
        if len(pt) < 2:
            continue
        yield float(pt[0]), float(pt[1])


def _bounds_from_geojson(data: dict) -> Tuple[int, int, int, int]:
    """Return integer inclusive bounds (min_x, min_y, max_x, max_y)."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    def visit_coords(obj: Any) -> None:
        nonlocal min_x, min_y, max_x, max_y
        if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[0], (int, float)):
            x, y = float(obj[0]), float(obj[1])
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit_coords(item)

    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        visit_coords(geom.get("coordinates"))

    if min_x == float("inf"):
        raise ValueError("No coordinates found in GeoJSON features.")

    return int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x)), int(np.ceil(max_y))


def _ring_to_int32(ring: list) -> np.ndarray:
    pts = np.array([[int(round(x)), int(round(y))] for x, y in _iter_xy_pairs(ring)], dtype=np.int32)
    if pts.shape[0] < 3:
        return pts
    # Drop closing duplicate vertex if present (fillPoly does not require it).
    if np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def _ring_bbox_area(ring: list) -> Tuple[float, float, float, float, float]:
    xs = [float(p[0]) for p in ring if len(p) >= 2]
    ys = [float(p[1]) for p in ring if len(p) >= 2]
    if not xs:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    return min_x, max_x, min_y, max_y, (max_x - min_x) * (max_y - min_y)


def _is_full_slide_exterior(
    rings: list,
    global_min_x: int,
    global_min_y: int,
    global_max_x: int,
    global_max_y: int,
    area_fraction: float,
) -> bool:
    """
    QuPath sometimes adds a MultiPolygon member that is the whole image rectangle.
    Union semantics would paint the entire mask; skip that exterior-only polygon.
    """
    if len(rings) != 1:
        return False
    ring = rings[0]
    min_x, max_x, min_y, max_y, a = _ring_bbox_area(ring)
    gw = float(global_max_x - global_min_x + 1)
    gh = float(global_max_y - global_min_y + 1)
    if gw * gh <= 0:
        return False
    if a < area_fraction * gw * gh:
        return False
    tol = 2.0
    return (
        min_x <= global_min_x + tol
        and min_y <= global_min_y + tol
        and max_x >= global_max_x - tol
        and max_y >= global_max_y - tol
    )


def _fill_polygon_rings(
    mask: np.ndarray,
    rings: list,
    origin_x: int,
    origin_y: int,
    *,
    global_bounds: Tuple[int, int, int, int],
    skip_full_slide_exterior: bool,
    full_slide_area_fraction: float,
) -> None:
    """Fill first ring as foreground; subsequent rings as background (holes)."""
    if not rings:
        return
    if skip_full_slide_exterior and _is_full_slide_exterior(
        rings, *global_bounds, full_slide_area_fraction
    ):
        return
    shifted: list[np.ndarray] = []
    for ring in rings:
        p = _ring_to_int32(ring)
        if p.shape[0] < 3:
            continue
        p = p.copy()
        p[:, 0] -= origin_x
        p[:, 1] -= origin_y
        shifted.append(p)
    if not shifted:
        return
    cv2.fillPoly(mask, [shifted[0]], 1)
    for hole in shifted[1:]:
        cv2.fillPoly(mask, [hole], 0)


def _rasterize_geometry(
    mask: np.ndarray,
    geometry: dict,
    origin_x: int,
    origin_y: int,
    *,
    global_bounds: Tuple[int, int, int, int],
    skip_full_slide_exterior: bool,
    full_slide_area_fraction: float,
) -> None:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return

    if gtype == "Polygon":
        _fill_polygon_rings(
            mask,
            coords,
            origin_x,
            origin_y,
            global_bounds=global_bounds,
            skip_full_slide_exterior=skip_full_slide_exterior,
            full_slide_area_fraction=full_slide_area_fraction,
        )
    elif gtype == "MultiPolygon":
        for polygon in coords:
            _fill_polygon_rings(
                mask,
                polygon,
                origin_x,
                origin_y,
                global_bounds=global_bounds,
                skip_full_slide_exterior=skip_full_slide_exterior,
                full_slide_area_fraction=full_slide_area_fraction,
            )
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")


def mask_shape_from_reference_image(path: Path) -> Tuple[int, int]:
    """Return (height, width) for a 2D image."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    return h, w


def geojson_to_mask(
    geojson_path: Path,
    shape_hw: Tuple[int, int] | None = None,
    origin_xy: Tuple[int, int] | None = None,
    *,
    skip_full_slide_exterior: bool = True,
    full_slide_area_fraction: float = 0.95,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load GeoJSON and produce a binary mask (uint8, values 0 and 1).

    Parameters
    ----------
    geojson_path
        Path to a FeatureCollection GeoJSON file.
    shape_hw
        Optional (height, width). If omitted, tight bounds from all coordinates.
    origin_xy
        Optional top-left (x, y) in the same coordinate system as the GeoJSON.
        When ``shape_hw`` is set, defaults to the inclusive min corner of all
        geometry so exported coordinates still align. Pass (0, 0) to place the
        mask in a fixed canvas starting at the image origin.

    Returns
    -------
    mask, (origin_x, origin_y)
        ``mask`` has shape (H, W). ``origin_x, origin_y`` map mask indices back
        to GeoJSON / image pixels: image_x = col + origin_x, image_y = row + origin_y.
    """
    with geojson_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection at the top level.")

    min_x, min_y, max_x, max_y = _bounds_from_geojson(data)

    if shape_hw is None:
        origin_x, origin_y = min_x, min_y
        height = max_y - min_y + 1
        width = max_x - min_x + 1
    else:
        height, width = shape_hw[0], shape_hw[1]
        if origin_xy is None:
            origin_x, origin_y = min_x, min_y
        else:
            origin_x, origin_y = origin_xy[0], origin_xy[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    global_bounds = (min_x, min_y, max_x, max_y)

    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        _rasterize_geometry(
            mask,
            geom,
            origin_x,
            origin_y,
            global_bounds=global_bounds,
            skip_full_slide_exterior=skip_full_slide_exterior,
            full_slide_area_fraction=full_slide_area_fraction,
        )

    return mask, (origin_x, origin_y)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geojson", type=Path, help="Input GeoJSON path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output mask path (.png or .npy). Default: <geojson_stem>_mask.png",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Use this image's height and width as the mask canvas (align with fluorescence image).",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Force mask height (pixels). Use with --width.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Force mask width (pixels). Use with --height.",
    )
    parser.add_argument(
        "--origin",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Top-left of the mask in GeoJSON coordinates (default: min x,y of all geometry).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also write a float32 0/1 .npy next to the PNG (or alone if output ends with .npy).",
    )
    parser.add_argument(
        "--keep-full-frame",
        action="store_true",
        help="Do not skip a lone exterior ring that covers the whole GeoJSON bounds (QuPath sometimes adds this).",
    )
    parser.add_argument(
        "--full-frame-area-fraction",
        type=float,
        default=0.95,
        help="Minimum bbox-area fraction of the GeoJSON canvas to treat as full-slide exterior when skipping (default: 0.95).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.geojson.is_file():
        print(f"Error: GeoJSON not found: {args.geojson}", file=sys.stderr)
        return 1

    shape_hw: Tuple[int, int] | None = None
    if args.reference_image is not None:
        shape_hw = mask_shape_from_reference_image(args.reference_image)
    elif args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            print("Error: both --height and --width are required when not using --reference-image.", file=sys.stderr)
            return 1
        shape_hw = (args.height, args.width)

    origin = tuple(args.origin) if args.origin is not None else None
    mask, (ox, oy) = geojson_to_mask(
        args.geojson,
        shape_hw=shape_hw,
        origin_xy=origin,
        skip_full_slide_exterior=not args.keep_full_frame,
        full_slide_area_fraction=args.full_frame_area_fraction,
    )

    out = args.output
    if out is None:
        out = args.geojson.with_name(f"{args.geojson.stem}_mask.png")

    if out.suffix.lower() == ".npy":
        np.save(out, mask.astype(np.float32))
        print(f"Wrote {out} shape={mask.shape} dtype=float32 origin_xy=({ox}, {oy})")
        return 0

    png = (mask * 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), png):
        print(f"Error: failed to write {out}", file=sys.stderr)
        return 1
    print(f"Wrote {out} shape={mask.shape} (PNG 0/255) origin_xy=({ox}, {oy})")

    if args.save_npy:
        npy = out.with_suffix(".npy")
        np.save(npy, mask.astype(np.float32))
        print(f"Wrote {npy} dtype=float32 values 0/1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
