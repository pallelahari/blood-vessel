"""
Quantitative analysis of blood vessels from a binary mask.

Uses a 1-pixel skeleton (medial axis topology) and decomposes the network into
segments between graph nodes (endpoints and junctions). Each segment has no
branches along its interior; bifurcations appear only at segment ends.

Radius/width is estimated from the Euclidean distance transform of the
foreground mask, sampled along the skeleton path.

Helpers :func:`label_foreground_blobs`, :func:`blob_statistics_records`, and
:func:`render_blob_overlay` aggregate segment metrics per connected foreground
blob (mask region). Per-blob rows include 2D shape descriptors (eccentricity,
axis ratio, circularity, ``morphology_2d``) to separate compact vs elongated
footprints in the slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, skeletonize

# 8-neighbor offsets (row, col), excluding (0,0)
_NEIGH8: tuple[tuple[int, int], ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if dr != 0 or dc != 0
)


class SegmentFilter(str, Enum):
    """Which skeleton segments to keep for reporting."""

    ALL = "all"
    """Every segment between topological nodes (junctions and free tips)."""

    TERMINAL_ONLY = "terminal_only"
    """Segments with at least one free tip (degree-1 skeleton endpoint)."""

    INTERNAL_ONLY = "internal_only"
    """Segments whose both ends are junctions (degree >= 3)."""


@dataclass
class VesselSegment:
    """One vessel entity: a maximal straight run between skeleton graph nodes."""

    segment_id: int
    path_rc: np.ndarray
    """(N, 2) int32 array of (row, col) skeleton pixels along the segment."""

    length_px: float
    """Geodesic length along the 8-connected skeleton path in pixels."""

    radius_mean_px: float
    radius_median_px: float
    radius_std_px: float
    width_mean_px: float
    width_median_px: float

    degree_end_a: int
    degree_end_b: int
    """Skeleton neighbor count at the two ends (1 = tip, >=3 = junction)."""

    is_terminal: bool
    """True if at least one end is a tip (degree 1)."""

    is_internal: bool
    """True if both ends are junctions (degree >= 3)."""

    radius_computed_px: int
    """Number of skeleton samples used for radius (after optional junction mask)."""


def load_binary_mask(
    path: str | Path,
    *,
    fg_threshold: int | None = None,
) -> np.ndarray:
    """
    Load a binary vessel mask from PNG/TIF/etc.

    Values are normalized to bool foreground. If ``fg_threshold`` is None,
    any nonzero pixel is foreground; otherwise pixels ``>= fg_threshold``.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if fg_threshold is None:
        fg = img > 0
    else:
        fg = img >= fg_threshold
    return fg


def preprocess_mask(
    mask: np.ndarray,
    *,
    min_object_area_px: int = 0,
) -> np.ndarray:
    """Optional cleanup: remove tiny foreground blobs."""
    fg = mask.astype(bool)
    if min_object_area_px > 0:
        fg = remove_small_objects(fg, min_size=int(min_object_area_px))
    return fg


def _neighbor_skeleton_degrees(skel: np.ndarray) -> np.ndarray:
    """For each pixel, count 8-neighbors that are skeleton (0 if not on skeleton)."""
    s = skel.astype(np.uint8)
    acc = np.zeros_like(s, dtype=np.int32)
    for dr, dc in _NEIGH8:
        acc += np.roll(np.roll(s, dr, axis=0), dc, axis=1)
    out = np.zeros_like(s, dtype=np.int32)
    out[skel] = acc[skel]
    return out


def _iter_neighbors8(r: int, c: int, h: int, w: int) -> Iterator[tuple[int, int]]:
    for dr, dc in _NEIGH8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _path_length_px(path_rc: np.ndarray) -> float:
    if path_rc.shape[0] < 2:
        return 0.0
    d = np.diff(path_rc.astype(np.float64), axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def _junction_mask(skel: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """Skeleton pixels that are junctions (degree >= 3)."""
    j = np.zeros_like(skel, dtype=bool)
    j[skel] = degrees[skel] >= 3
    return j


def _radius_stats_along_path(
    dt: np.ndarray,
    path_rc: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, int]:
    """
    Sample distance transform along path. DT is in pixels (radius-like for tubes).

    Returns mean, median, std, width_mean, width_median, count_used.
    """
    if path_rc.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    r, c = path_rc[:, 0], path_rc[:, 1]
    if valid_mask is not None:
        ok = valid_mask[r, c]
        r, c = r[ok], c[ok]
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    rad = dt[r, c].astype(np.float64)
    w = 2.0 * rad
    return (
        float(np.mean(rad)),
        float(np.median(rad)),
        float(np.std(rad)),
        float(np.mean(w)),
        float(np.median(w)),
        int(rad.size),
    )


def _walk_segment(
    skel: np.ndarray,
    node_mask: np.ndarray,
    start_r: int,
    start_c: int,
    first_r: int,
    first_c: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [(start_r, start_c), (first_r, first_c)]
    prev = (start_r, start_c)
    cur = (first_r, first_c)
    seen: set[tuple[int, int]] = {path[0], path[1]}
    while True:
        r, c = cur
        if node_mask[r, c] and (r, c) != (start_r, start_c):
            break
        nxt = None
        for rr, cc in _iter_neighbors8(r, c, h, w):
            if not skel[rr, cc] or (rr, cc) == prev:
                continue
            if nxt is not None:
                # branch in supposedly degree-2 region — stop at ambiguity
                nxt = None
                break
            nxt = (rr, cc)
        if nxt is None:
            break
        if nxt in seen:
            break
        seen.add(nxt)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def prune_skeleton_spurs(skel: np.ndarray, max_spur_length_px: float) -> np.ndarray:
    """
    Remove short dead-end branches on the skeleton (degree-1 tips).

    Only spurs whose first topological node is a junction (degree >= 3) are
    removed; segments between two free tips are left unchanged. This reduces
    spurious short edges at thick or noisy bifurcations.
    """
    if max_spur_length_px <= 0:
        return skel.astype(bool)

    s = skel.astype(bool).copy()
    h, w = s.shape
    changed = True
    while changed:
        changed = False
        deg = _neighbor_skeleton_degrees(s)
        tips = np.argwhere(s & (deg == 1))
        for tr, tc in tips:
            tr, tc = int(tr), int(tc)
            path: list[tuple[int, int]] = [(tr, tc)]
            prev: tuple[int, int] | None = None
            cur = (tr, tc)
            while True:
                nbrs = [(rr, cc) for rr, cc in _iter_neighbors8(*cur, h, w) if s[rr, cc]]
                if prev is not None:
                    nbrs = [n for n in nbrs if n != prev]
                if len(nbrs) != 1:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
                d = int(_neighbor_skeleton_degrees(s)[cur[0], cur[1]])
                if d != 2:
                    break
            if len(path) < 2:
                continue
            end = path[-1]
            end_deg = int(_neighbor_skeleton_degrees(s)[end[0], end[1]])
            if end_deg < 3:
                continue
            length = _path_length_px(np.array(path, dtype=np.int32))
            if length < max_spur_length_px:
                for rr, cc in path[:-1]:
                    s[rr, cc] = False
                changed = True
    return s


def extract_skeleton_segments(
    skel: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    """
    Decompose skeleton into undirected paths between nodes (deg != 2).

    Returns list of paths (each path is list of (r,c)) and ``node_mask``.
    """
    skel = skel.astype(bool)
    h, w = skel.shape
    deg = _neighbor_skeleton_degrees(skel)
    node_mask = skel & (deg != 2)

    # If no nodes (e.g. simple loop or single chain wrongly all deg2 — rare),
    # treat entire connected component as one segment by arbitrary break.
    if not np.any(node_mask) and np.any(skel):
        ys, xs = np.where(skel)
        # start at top-left-most skeleton pixel and walk until repeat
        start = (int(ys[0]), int(xs[0]))
        path = [start]
        prev: tuple[int, int] | None = None
        cur = start
        seen: set[tuple[int, int]] = {start}
        while True:
            nbrs = [(rr, cc) for rr, cc in _iter_neighbors8(*cur, h, w) if skel[rr, cc]]
            nxt = None
            for nb in nbrs:
                if prev is not None and nb == prev:
                    continue
                if nb not in seen:
                    nxt = nb
                    break
            if nxt is None:
                break
            seen.add(nxt)
            path.append(nxt)
            prev, cur = cur, nxt
        return [path], np.zeros_like(skel, dtype=bool)

    visited_edge: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[list[tuple[int, int]]] = []

    nodes_rc = np.argwhere(node_mask)
    for start_r, start_c in nodes_rc:
        for fr, fc in _iter_neighbors8(int(start_r), int(start_c), h, w):
            if not skel[fr, fc]:
                continue
            a, b = (int(start_r), int(start_c)), (int(fr), int(fc))
            key = (a, b) if a < b else (b, a)
            if key in visited_edge:
                continue
            visited_edge.add(key)
            raw = _walk_segment(skel, node_mask, a[0], a[1], b[0], b[1], h, w)
            if len(raw) < 2:
                continue
            end = raw[-1]
            if not node_mask[end[0], end[1]]:
                continue
            # mark all undirected edges along path as visited
            for i in range(len(raw) - 1):
                p, q = raw[i], raw[i + 1]
                ek = (p, q) if p < q else (q, p)
                visited_edge.add(ek)
            segments.append(raw)

    return segments, node_mask


def _segment_filter_keep(
    deg_a: int,
    deg_b: int,
    filt: SegmentFilter,
) -> bool:
    is_terminal = (deg_a == 1) or (deg_b == 1)
    is_internal = (deg_a >= 3) and (deg_b >= 3)
    if filt == SegmentFilter.ALL:
        return True
    if filt == SegmentFilter.TERMINAL_ONLY:
        return is_terminal
    if filt == SegmentFilter.INTERNAL_ONLY:
        return is_internal
    return True


def analyze_vessel_mask(
    mask: np.ndarray,
    *,
    pixel_size_um: float | None = None,
    min_segment_length_px: float = 0.0,
    segment_filter: SegmentFilter | str = SegmentFilter.ALL,
    radius_exclude_near_junction_px: float = 0.0,
    min_object_area_px: int = 0,
    prune_spur_length_px: float = 0.0,
) -> tuple[list[VesselSegment], np.ndarray, dict]:
    """
    Full pipeline: mask → skeleton → segments → lengths and radii.

    Parameters
    ----------
    mask
        Boolean or 0/1 foreground mask.
    pixel_size_um
        If set, ``length_um`` / ``radius_um`` style scaling could be added later;
        currently all outputs are in pixels unless you multiply externally.
    min_segment_length_px
        Drop segments shorter than this (geodesic length along skeleton).
    segment_filter
        ``all`` | ``terminal_only`` | ``internal_only`` (see :class:`SegmentFilter`).
    radius_exclude_near_junction_px
        If > 0, skeleton samples within this distance (pixels) of any junction
        pixel are excluded from mean/median radius (width) statistics.
    min_object_area_px
        Passed to :func:`preprocess_mask`.
    prune_spur_length_px
        If > 0, short dead-end skeleton branches shorter than this length (in
        pixels, along the path before the first junction) are removed before
        graph extraction. Helps suppress spurious tips from rasterization noise.

    Returns
    -------
    segments
        List of :class:`VesselSegment`.
    label_image
        int32 array, shape like ``mask``; ``label_image[r,c] == segment_id`` on
        skeleton pixels (0 = background / unlabeled).
    meta
        Dict with keys ``skeleton``, ``node_mask``, ``distance_transform``,
        ``junction_distance`` (if junction exclusion used), ``pixel_size_um``.
    """
    _ = pixel_size_um  # reserved for future physical units
    if isinstance(segment_filter, str):
        segment_filter = SegmentFilter(segment_filter)

    fg = preprocess_mask(mask.astype(bool), min_object_area_px=min_object_area_px)
    if not np.any(fg):
        return [], np.zeros_like(fg, dtype=np.int32), {
            "skeleton": np.zeros_like(fg, dtype=bool),
            "node_mask": np.zeros_like(fg, dtype=bool),
            "distance_transform": np.zeros_like(fg, dtype=np.float32),
            "junction_distance": None,
            "pixel_size_um": pixel_size_um,
        }

    skel = skeletonize(fg)
    skel = prune_skeleton_spurs(skel, prune_spur_length_px)
    dt = distance_transform_edt(fg)
    deg = _neighbor_skeleton_degrees(skel)
    node_mask = skel & (deg != 2)
    raw_paths, _ = extract_skeleton_segments(skel)

    jmask = _junction_mask(skel, deg)
    junction_dist: np.ndarray | None = None
    valid_for_radius: np.ndarray | None = None
    if radius_exclude_near_junction_px > 0 and np.any(jmask):
        junction_dist = distance_transform_edt(~jmask)
        valid_for_radius = junction_dist > float(radius_exclude_near_junction_px)
        valid_for_radius &= skel

    label_img = np.zeros(fg.shape, dtype=np.int32)
    segments: list[VesselSegment] = []
    seg_id = 0
    for path in raw_paths:
        path_rc = np.array(path, dtype=np.int32)
        length_px = _path_length_px(path_rc)
        if length_px < min_segment_length_px:
            continue
        ra, ca = path[0]
        rb, cb = path[-1]
        da, db = int(deg[ra, ca]), int(deg[rb, cb])
        if not _segment_filter_keep(da, db, segment_filter):
            continue
        seg_id += 1
        r, c = path_rc[:, 0], path_rc[:, 1]
        vm = valid_for_radius
        rm, rmed, rs, wm, wmed, n_used = _radius_stats_along_path(dt, path_rc, valid_mask=vm)
        is_terminal = (da == 1) or (db == 1)
        is_internal = (da >= 3) and (db >= 3)
        segments.append(
            VesselSegment(
                segment_id=seg_id,
                path_rc=path_rc,
                length_px=length_px,
                radius_mean_px=rm,
                radius_median_px=rmed,
                radius_std_px=rs,
                width_mean_px=wm,
                width_median_px=wmed,
                degree_end_a=da,
                degree_end_b=db,
                is_terminal=is_terminal,
                is_internal=is_internal,
                radius_computed_px=n_used,
            )
        )
        label_img[r, c] = seg_id

    meta = {
        "skeleton": skel,
        "node_mask": node_mask,
        "distance_transform": dt.astype(np.float32),
        "junction_distance": junction_dist,
        "pixel_size_um": pixel_size_um,
    }
    return segments, label_img, meta


def label_foreground_blobs(
    mask: np.ndarray,
    *,
    connectivity: int = 2,
) -> tuple[np.ndarray, int]:
    """
    Label connected components of the 2D foreground mask.

    Parameters
    ----------
    connectivity
        ``1`` = 4-neighborhood, ``2`` = 8-neighborhood (``scipy.ndimage`` convention).

    Returns
    -------
    cc_labels
        int32 array; 0 = background, 1..N = blob ids.
    n_blobs
        Number of foreground components (N).
    """
    fg = mask.astype(bool)
    struct = generate_binary_structure(2, int(connectivity))
    lab, nf = label(fg, structure=struct)
    return lab.astype(np.int32), int(nf)


def assign_segments_to_blobs(
    segments: Sequence[VesselSegment],
    cc_labels: np.ndarray,
) -> np.ndarray:
    """
    Map each segment to the blob id (1..N) that contains a plurality of its
    skeleton pixels. Returns int32 shape ``(len(segments),)``; ``0`` if no
    labeled foreground is sampled on the path.
    """
    out = np.zeros(len(segments), dtype=np.int32)
    for i, s in enumerate(segments):
        r = s.path_rc[:, 0].astype(np.intp, copy=False)
        c = s.path_rc[:, 1].astype(np.intp, copy=False)
        labs = cc_labels[r, c]
        pos = labs > 0
        if not np.any(pos):
            continue
        labs = labs[pos]
        uniq, counts = np.unique(labs, return_counts=True)
        out[i] = int(uniq[int(np.argmax(counts))])
    return out


def _blob_longest_lengths_px(
    subset: Sequence[VesselSegment],
    *,
    bridge_disconnected: bool = False,
) -> tuple[float, float, float]:
    """
    For segments assigned to one blob:

    Returns
    -------
    max_segment_length_px
        Longest contiguous path across connected segments in the blob
        (maximum weighted trail without reusing any segment/edge).
    centerline_network_diameter_px
        Metric diameter of the segment graph: maximum over all node pairs of
        the shortest-path distance, where edge weights are segment lengths.
        For a branched tree this is the longest tip-to-tip path along the
        skeleton; disconnected subgraphs yield the max over components.
        If ``bridge_disconnected=True``, disconnected subgraphs are joined by
        shortest endpoint-to-endpoint virtual bridges inside each blob graph.
    width_along_longest_path_px
        Length-weighted mean segment width along the longest contiguous path.
        Virtual bridge links (if enabled) do not contribute width.
    """
    if not subset:
        return 0.0, 0.0, 0.0
    nodes_set: set[tuple[int, int]] = set()
    edges: list[tuple[tuple[int, int], tuple[int, int], float, float]] = []
    for s in subset:
        p = s.path_rc
        a = (int(p[0, 0]), int(p[0, 1]))
        b = (int(p[-1, 0]), int(p[-1, 1]))
        w = float(s.length_px)
        sw = float(s.width_mean_px)
        nodes_set.add(a)
        nodes_set.add(b)
        edges.append((a, b, w, sw))

    nodes = list(nodes_set)
    n = len(nodes)
    if n < 2:
        single = max(float(s.length_px) for s in subset)
        i = int(np.argmax([float(s.length_px) for s in subset]))
        return single, 0.0, float(subset[i].width_mean_px)
    idx = {node: i for i, node in enumerate(nodes)}
    edge_idx: list[tuple[int, int, float, float, bool]] = [
        (idx[a], idx[b], w, sw, False) for a, b, w, sw in edges
    ]

    def _longest_edge_simple_path(
        edges_local: list[tuple[int, int, float, float, bool]],
    ) -> tuple[float, float]:
        m_local = len(edges_local)
        if m_local == 0:
            return 0.0, 0.0
        if m_local > 22:
            return -1.0, 0.0
        adj_local: list[list[tuple[int, int, float, float, bool]]] = [[] for _ in range(n)]
        for ei, (ia, ib, w, sw, is_virtual) in enumerate(edges_local):
            adj_local[ia].append((ib, ei, w, sw, is_virtual))
            adj_local[ib].append((ia, ei, w, sw, is_virtual))
        best_len = 0.0
        best_w = 0.0

        def _dfs_edge_simple(
            u: int,
            used_mask: int,
            acc_len: float,
            acc_real_len: float,
            acc_real_w_num: float,
        ) -> None:
            nonlocal best_len, best_w
            if acc_len > best_len:
                best_len = acc_len
                best_w = (acc_real_w_num / acc_real_len) if acc_real_len > 0 else 0.0
            for v, ei, w, sw, is_virtual in adj_local[u]:
                bit = 1 << ei
                if used_mask & bit:
                    continue
                if is_virtual:
                    _dfs_edge_simple(v, used_mask | bit, acc_len + w, acc_real_len, acc_real_w_num)
                else:
                    _dfs_edge_simple(
                        v,
                        used_mask | bit,
                        acc_len + w,
                        acc_real_len + w,
                        acc_real_w_num + (w * sw),
                    )

        for start in range(n):
            _dfs_edge_simple(start, 0, 0.0, 0.0, 0.0)
        return best_len, best_w

    if bridge_disconnected:
        # Greedily connect disconnected endpoint-graphs with shortest virtual links.
        # This allows per-blob "treat everything together" length summaries.
        parent = list(range(n))

        def _find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        for ia, ib, _, _, _ in edge_idx:
            _union(ia, ib)

        while True:
            reps = {_find(i) for i in range(n)}
            if len(reps) <= 1:
                break
            best_pair: tuple[int, int, float] | None = None
            for i in range(n):
                ri = _find(i)
                ai = nodes[i]
                for j in range(i + 1, n):
                    rj = _find(j)
                    if ri == rj:
                        continue
                    aj = nodes[j]
                    d = float(np.hypot(ai[0] - aj[0], ai[1] - aj[1]))
                    if best_pair is None or d < best_pair[2]:
                        best_pair = (i, j, d)
            if best_pair is None:
                break
            i, j, d = best_pair
            edge_idx.append((i, j, d, 0.0, True))
            _union(i, j)

    # Longest contiguous path (edge-simple trail).
    max_contig, w_on_max = _longest_edge_simple_path(edge_idx)

    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for ia, ib, w, _, _ in edge_idx:
        if w < dist[ia, ib]:
            dist[ia, ib] = w
            dist[ib, ia] = w
    for k in range(n):
        for i in range(n):
            dik = dist[i, k]
            if dik == np.inf:
                continue
            row_k = dist[k, :]
            dist[i, :] = np.minimum(dist[i, :], dik + row_k)

    dmax = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dij = dist[i, j]
            if np.isfinite(dij) and dij > dmax:
                dmax = float(dij)
    if max_contig < 0:
        # Fallback for large graphs where exact edge-simple search is expensive.
        max_contig = dmax
        real_len = sum(w for _, _, w, _, is_virtual in edge_idx if not is_virtual)
        w_num = sum(w * sw for _, _, w, sw, is_virtual in edge_idx if not is_virtual)
        w_on_max = (w_num / real_len) if real_len > 0 else 0.0
    return max_contig, dmax, w_on_max


def _blob_2d_shape_metrics(
    blob_mask: np.ndarray,
    *,
    compact_max_eccentricity: float,
    elongated_min_eccentricity: float,
    elongated_min_axis_ratio: float,
    min_area_for_morphology_label: int = 4,
) -> dict[str, float | str]:
    """
    Describe each foreground blob's 2D footprint: eccentricity / axis ratio
    (from the equivalent ellipse), isoperimetric circularity, and a coarse
    ``morphology_2d`` label.

    Low eccentricity (~0) corresponds to a compact, near-circular footprint
    (typical of vessels cutting through the slice). High eccentricity and/or
    a large major/minor axis ratio indicate an elongated in-plane footprint.
    """
    nan = float("nan")
    out: dict[str, float | str] = {
        "eccentricity_2d": nan,
        "ellipse_axis_major_px": nan,
        "ellipse_axis_minor_px": nan,
        "ellipse_axis_ratio": nan,
        "circularity_2d": nan,
        "morphology_2d": "degenerate",
    }
    if blob_mask.ndim != 2:
        return out
    area = int(np.count_nonzero(blob_mask))
    if area < 1:
        return out

    labeled = blob_mask.astype(np.uint8)
    rp = regionprops(labeled)[0]
    ecc = float(rp.eccentricity)
    maj = float(rp.axis_major_length)
    min_ax = float(rp.axis_minor_length)
    axis_ratio = maj / min_ax if min_ax > 1e-9 else float("inf")

    perim = getattr(rp, "perimeter_crofton", None)
    if perim is None or perim <= 0:
        perim = float(getattr(rp, "perimeter", 0.0) or 0.0)
    circ = (4.0 * np.pi * float(rp.area)) / (perim**2) if perim > 0 else nan

    out["eccentricity_2d"] = ecc
    out["ellipse_axis_major_px"] = maj
    out["ellipse_axis_minor_px"] = min_ax
    out["ellipse_axis_ratio"] = float(axis_ratio) if np.isfinite(axis_ratio) else nan
    out["circularity_2d"] = float(circ) if np.isfinite(circ) else nan

    if area < min_area_for_morphology_label or not np.isfinite(ecc):
        out["morphology_2d"] = "degenerate"
        return out

    if ecc <= compact_max_eccentricity:
        out["morphology_2d"] = "compact"
    elif ecc >= elongated_min_eccentricity or (
        np.isfinite(axis_ratio) and axis_ratio >= elongated_min_axis_ratio
    ):
        out["morphology_2d"] = "elongated"
    else:
        out["morphology_2d"] = "intermediate"
    return out


def blob_statistics_records(
    segments: Sequence[VesselSegment],
    mask: np.ndarray,
    *,
    connectivity: int = 2,
    pixel_size_um: float | None = None,
    shape_compact_max_eccentricity: float = 0.62,
    shape_elongated_min_eccentricity: float = 0.82,
    shape_elongated_min_axis_ratio: float = 4.0,
    shape_min_area_for_morphology_label: int = 4,
    merge_disconnected_segments_in_blob: bool = True,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Aggregate segment metrics per foreground connected component (blob).

    Blobs are defined by ``label_foreground_blobs`` on ``mask``. Each segment
    is assigned to exactly one blob via :func:`assign_segments_to_blobs`.
    Blobs with no segments (e.g. too small to survive skeletonization) still
    appear with ``n_segments=0`` and skeleton length / width fields at zero,
    but ``area_px`` is filled. Rows also include ``max_segment_length_px`` and
    ``centerline_network_diameter_px`` as blob-level path lengths on the
    junction/tip segment graph, plus width-on-path summaries.

    If ``merge_disconnected_segments_in_blob=True`` (default), disconnected
    segment pieces within the same foreground blob are linked using shortest
    endpoint-to-endpoint virtual bridges before these path metrics are
    computed. This provides a "treat all skeleton pieces in a blob together"
    interpretation for downstream per-blob analysis.

    Each row also includes 2D footprint shape metrics from the equivalent
    ellipse (``eccentricity_2d``, ``ellipse_axis_*``, ``ellipse_axis_ratio``),
    ``circularity_2d`` (``4π·area / perimeter²``), and ``morphology_2d``:
    ``compact`` / ``elongated`` / ``intermediate`` / ``degenerate``, using
    ``shape_*`` thresholds below.

    Parameters
    ----------
    shape_compact_max_eccentricity
        Blobs with ``eccentricity_2d`` at or below this value are labeled
        ``compact`` (near-circular cross-section in the slice).
    shape_elongated_min_eccentricity
        Blobs at or above this eccentricity are ``elongated`` (unless already
        ``compact`` from the first rule; the compact rule is checked first).
    shape_elongated_min_axis_ratio
        Alternatively, ``elongated`` if major/minor ellipse axis ratio reaches
        this value even when eccentricity is in the middle range.
    shape_min_area_for_morphology_label
        Blobs with fewer foreground pixels are ``degenerate`` for
        ``morphology_2d`` (ellipse / classification unstable).

    Returns
    -------
    rows
        One dict per blob id 1..N.
    cc_labels
        Label image, same shape as ``mask``.
    segment_blob_id
        Shape ``(len(segments),)``; ``segment_blob_id[i]`` is the blob for
        ``segments[i]``.
    """
    cc_labels, n_blobs = label_foreground_blobs(mask, connectivity=connectivity)
    seg_blob = assign_segments_to_blobs(segments, cc_labels)

    areas = np.bincount(
        cc_labels.ravel().astype(np.int64, copy=False),
        minlength=n_blobs + 1,
    )
    ps = float(pixel_size_um) if pixel_size_um is not None else None

    rows: list[dict] = []
    for bid in range(1, n_blobs + 1):
        idx = np.flatnonzero(seg_blob == bid)
        area_px = int(areas[bid]) if bid < areas.size else 0

        length_sum = 0.0
        w_num = 0.0
        r_num = 0.0
        seg_widths: list[float] = []
        for i in idx:
            s = segments[int(i)]
            L = float(s.length_px)
            W = float(s.width_mean_px)
            R = float(s.radius_mean_px)
            length_sum += L
            w_num += L * W
            r_num += L * R
            seg_widths.append(W)

        w_lw = (w_num / length_sum) if length_sum > 0 else 0.0
        r_lw = (r_num / length_sum) if length_sum > 0 else 0.0
        w_simple = float(np.mean(seg_widths)) if seg_widths else 0.0

        subsegs = [segments[int(i)] for i in idx]
        max_len_unmerged, diam_unmerged, w_on_max_unmerged = _blob_longest_lengths_px(
            subsegs,
            bridge_disconnected=False,
        )
        max_len_merged, diam_merged, w_on_max_merged = _blob_longest_lengths_px(
            subsegs,
            bridge_disconnected=True,
        )
        if merge_disconnected_segments_in_blob:
            max_seg_len = max_len_merged
            net_diam = diam_merged
            w_on_max = w_on_max_merged
        else:
            max_seg_len = max_len_unmerged
            net_diam = diam_unmerged
            w_on_max = w_on_max_unmerged

        blob_mask = cc_labels == bid
        shape_row = _blob_2d_shape_metrics(
            blob_mask,
            compact_max_eccentricity=shape_compact_max_eccentricity,
            elongated_min_eccentricity=shape_elongated_min_eccentricity,
            elongated_min_axis_ratio=shape_elongated_min_axis_ratio,
            min_area_for_morphology_label=shape_min_area_for_morphology_label,
        )

        row: dict = {
            "blob_id": int(bid),
            "area_px": area_px,
            "n_segments": int(idx.size),
            "total_skeleton_length_px": length_sum,
            "max_segment_length_px": max_seg_len,
            "centerline_network_diameter_px": net_diam,
            "centerline_network_diameter_unmerged_px": diam_unmerged,
            "centerline_network_diameter_merged_px": diam_merged,
            "width_mean_length_weighted_px": w_lw,
            "radius_mean_length_weighted_px": r_lw,
            "width_mean_over_segments_px": w_simple,
            "width_along_longest_path_px": w_on_max,
            "width_along_longest_unmerged_path_px": w_on_max_unmerged,
            "width_along_longest_merged_path_px": w_on_max_merged,
            **shape_row,
        }
        if ps is not None:
            row["area_um2"] = area_px * ps * ps
            row["total_skeleton_length_um"] = length_sum * ps
            row["max_segment_length_um"] = max_seg_len * ps
            row["centerline_network_diameter_um"] = net_diam * ps
            row["centerline_network_diameter_unmerged_um"] = diam_unmerged * ps
            row["centerline_network_diameter_merged_um"] = diam_merged * ps
            row["width_mean_length_weighted_um"] = w_lw * ps
            row["radius_mean_length_weighted_um"] = r_lw * ps
            row["width_mean_over_segments_um"] = w_simple * ps
            row["width_along_longest_path_um"] = w_on_max * ps
            row["width_along_longest_unmerged_path_um"] = w_on_max_unmerged * ps
            row["width_along_longest_merged_path_um"] = w_on_max_merged * ps
        rows.append(row)

    return rows, cc_labels, seg_blob


def render_blob_overlay(
    background: np.ndarray,
    cc_labels: np.ndarray,
    *,
    alpha: float = 0.55,
    cmap_seed: int = 0,
) -> np.ndarray:
    """
    Blend a random color per connected component (``cc_labels > 0``) on gray BGR.
    """
    rng = np.random.default_rng(cmap_seed)
    if background.dtype != np.uint8:
        bg = background.astype(np.float64)
        if bg.max() > 1.5:
            bg = np.clip(bg, 0, 255) / 255.0
        else:
            bg = np.clip(bg, 0, 1)
        bg_u8 = (bg * 255).astype(np.uint8)
    else:
        bg_u8 = background
    if bg_u8.ndim == 2:
        base = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
    else:
        base = bg_u8.copy()
    h, w = cc_labels.shape
    color = np.zeros((h, w, 3), dtype=np.float32)
    n = int(cc_labels.max())
    for k in range(1, n + 1):
        bgr = rng.random(3).astype(np.float32)
        color[cc_labels == k] = bgr
    color_u8 = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    out = (alpha * color_u8 + (1 - alpha) * base.astype(np.float32)).astype(np.uint8)
    return out


def annotate_blob_ids(
    image_bgr: np.ndarray,
    cc_labels: np.ndarray,
    *,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> np.ndarray:
    """Draw ``blob_id`` at the centroid of each labeled component."""
    out = image_bgr.copy()
    n = int(cc_labels.max())
    for k in range(1, n + 1):
        ys, xs = np.where(cc_labels == k)
        if ys.size == 0:
            continue
        r, c = int(np.mean(ys)), int(np.mean(xs))
        txt = str(k)
        cv2.putText(
            out,
            txt,
            (c, r),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            txt,
            (c, r),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def segments_to_records(
    segments: Sequence[VesselSegment],
    *,
    pixel_size_um: float | None = None,
) -> list[dict]:
    """Flatten segments to JSON-friendly dict rows. Optional ``pixel_size_um`` adds µm fields."""
    rows: list[dict] = []
    ps = float(pixel_size_um) if pixel_size_um is not None else None
    for s in segments:
        row: dict = {
            "segment_id": s.segment_id,
            "length_px": s.length_px,
            "radius_mean_px": s.radius_mean_px,
            "radius_median_px": s.radius_median_px,
            "radius_std_px": s.radius_std_px,
            "width_mean_px": s.width_mean_px,
            "width_median_px": s.width_median_px,
            "degree_end_a": s.degree_end_a,
            "degree_end_b": s.degree_end_b,
            "is_terminal": s.is_terminal,
            "is_internal": s.is_internal,
            "radius_computed_px": s.radius_computed_px,
        }
        if ps is not None:
            row["length_um"] = s.length_px * ps
            row["radius_mean_um"] = s.radius_mean_px * ps
            row["radius_median_um"] = s.radius_median_px * ps
            row["width_mean_um"] = s.width_mean_px * ps
            row["width_median_um"] = s.width_median_px * ps
        rows.append(row)
    return rows


def render_segment_overlay(
    background: np.ndarray,
    label_image: np.ndarray,
    *,
    alpha: float = 0.55,
    cmap_seed: int = 0,
) -> np.ndarray:
    """
    Blend a random-color per-segment overlay on a grayscale background.

    ``background`` is promoted to uint8 gray if needed.
    """
    rng = np.random.default_rng(cmap_seed)
    if background.dtype != np.uint8:
        bg = background.astype(np.float64)
        if bg.max() > 1.5:
            bg = np.clip(bg, 0, 255) / 255.0
        else:
            bg = np.clip(bg, 0, 1)
        bg_u8 = (bg * 255).astype(np.uint8)
    else:
        bg_u8 = background
    if bg_u8.ndim == 2:
        base = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
    else:
        base = bg_u8.copy()
    h, w = label_image.shape
    color = np.zeros((h, w, 3), dtype=np.float32)
    n_seg = int(label_image.max())
    for k in range(1, n_seg + 1):
        bgr = rng.random(3).astype(np.float32)
        color[label_image == k] = bgr
    color_u8 = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    out = (alpha * color_u8 + (1 - alpha) * base.astype(np.float32)).astype(np.uint8)
    return out


def annotate_segment_ids(
    image_bgr: np.ndarray,
    segments: Sequence[VesselSegment],
    *,
    font_scale: float = 0.35,
    thickness: int = 1,
) -> np.ndarray:
    """Draw segment_id text near the midpoint of each segment path."""
    out = image_bgr.copy()
    for s in segments:
        mid = s.path_rc[len(s.path_rc) // 2]
        r, c = int(mid[0]), int(mid[1])
        txt = str(s.segment_id)
        cv2.putText(
            out,
            txt,
            (c, r),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            txt,
            (c, r),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out
