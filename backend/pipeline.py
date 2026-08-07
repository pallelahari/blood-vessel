from __future__ import annotations

import base64
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from geojson_to_mask import geojson_to_mask
from vessel_quantification import (
    SegmentFilter,
    analyze_vessel_mask,
    annotate_blob_ids,
    blob_statistics_records,
    render_blob_overlay,
)
from autocorrelation import autocorrelation_report, autocorrelation_for_frontend


PIXEL_SIZE_UM: float | None = None
MIN_SEGMENT_LENGTH_PX = 3.0
SEGMENT_FILTER: SegmentFilter | str = SegmentFilter.ALL
PRUNE_SPUR_LENGTH_PX = 15.0
RADIUS_EXCLUDE_NEAR_JUNCTION_PX = 8.0
MIN_OBJECT_AREA_PX = 20
MAX_DIM = None   # No resizing — process at full resolution


_push_log = None  # injected by app.py

def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)
    if _push_log:
        _push_log(msg)


def clean_length_width_for_plot(df, area_col="area_px", length_col="max_segment_length_px",
    width_col="width_along_longest_merged_path_px", out_length_col="length_clean", out_width_col="width_clean"):
    out = df.copy()
    out[out_length_col] = out[length_col].astype(float)
    out[out_width_col] = out[width_col].astype(float)
    both_zero = (out[out_length_col] == 0) & (out[out_width_col] == 0) & (out[area_col] > 0)
    eq = np.sqrt(out.loc[both_zero, area_col] / np.pi)
    out.loc[both_zero, out_length_col] = eq
    out.loc[both_zero, out_width_col] = eq
    width_zero_len_nonzero = (out[out_length_col] > 0) & (out[out_width_col] == 0)
    out.loc[width_zero_len_nonzero, out_width_col] = out.loc[width_zero_len_nonzero, out_length_col]
    return out


def to_png_data_url(img: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode PNG image.")
    payload = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        if x.max() > 1.5:
            x = np.clip(x, 0, 255)
        else:
            x = np.clip(x, 0, 1) * 255.0
        img = x.astype(np.uint8)
    return img


def skeleton_overlay(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    base = ensure_gray_u8(mask.astype(np.uint8) * 255)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    bgr[skeleton.astype(bool)] = (0, 255, 0)
    return bgr


def compact_views(mask: np.ndarray, cc_labels: np.ndarray, df_blobs: pd.DataFrame) -> np.ndarray:
    compact = df_blobs[(df_blobs["morphology_2d"] != "elongated") & (df_blobs["circularity_2d"] > 0.7)]["blob_id"].tolist()
    not_compact = df_blobs[((df_blobs["morphology_2d"] == "elongated") | (df_blobs["morphology_2d"] == "intermediate")) & (df_blobs["area_px"] > 100)]["blob_id"].tolist()
    compact_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    compact_img[:, :, 2] = (np.isin(cc_labels, compact) * 255).astype(np.uint8)
    compact_img[:, :, 0] = (np.isin(cc_labels, not_compact) * 255).astype(np.uint8)
    return compact_img


def blob_summary_row(df: pd.DataFrame, blob_id: int) -> dict[str, Any] | None:
    row = df[df["blob_id"] == blob_id]
    if row.empty:
        return None
    rec = row.iloc[0].to_dict()
    clean = clean_length_width_for_plot(row).iloc[0]
    rec["length_clean"] = float(clean["length_clean"])
    rec["width_clean"] = float(clean["width_clean"])
    return rec


@dataclass
class SessionState:
    output_root: Path
    cc_labels: np.ndarray | None = None
    df_blobs: pd.DataFrame | None = None
    output_dir: Path | None = None
    branch_points: list[tuple[int, int]] = field(default_factory=list)

    def process(self, image_bytes: bytes, geojson_bytes: bytes, image_name: str = "image", crop: tuple | None = None) -> dict[str, Any]:
        import re
        t0 = time.time()

        log("Decoding image...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Unable to decode image.")

        h0, w0 = image_bgr.shape[:2]
        log(f"Image size: {w0}x{h0}")

        scale = 1.0  # No resizing — full resolution

        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Apply crop if requested
        if crop is not None:
            cx1, cy1, cx2, cy2 = crop
            cx1 = max(0, cx1); cy1 = max(0, cy1)
            cx2 = min(w0, cx2); cy2 = min(h0, cy2)
            image_bgr  = image_bgr[cy1:cy2, cx1:cx2]
            image_gray = image_gray[cy1:cy2, cx1:cx2]
            log(f"Cropped to ({cx1},{cy1})→({cx2},{cy2}) = {cx2-cx1}×{cy2-cy1} px")

        log(f"Image decoded in {time.time()-t0:.1f}s")

        base_name = Path(image_name).stem
        safe_name = re.sub(r"[^\w\-]", "_", base_name)[:60]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_root / f"{safe_name}__{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"Output folder: {out_dir.name}")

        geojson_path = out_dir / "input.geojson"
        geojson_path.write_bytes(geojson_bytes)

        log("Building mask from GeoJSON...")
        t1 = time.time()
        mask_u8, _ = geojson_to_mask(geojson_path, shape_hw=image_gray.shape[:2], origin_xy=(0, 0))
        mask_bool = mask_u8.astype(bool)
        log(f"Mask built in {time.time()-t1:.1f}s — foreground px: {mask_bool.sum():,}")

        log("Running vessel analysis (skeletonization)...")
        t2 = time.time()
        segments, seg_label_image, meta = analyze_vessel_mask(
            mask_bool,
            pixel_size_um=PIXEL_SIZE_UM,
            min_segment_length_px=MIN_SEGMENT_LENGTH_PX,
            segment_filter=SEGMENT_FILTER,
            radius_exclude_near_junction_px=RADIUS_EXCLUDE_NEAR_JUNCTION_PX,
            min_object_area_px=MIN_OBJECT_AREA_PX,
            prune_spur_length_px=PRUNE_SPUR_LENGTH_PX,
        )
        log(f"Analysis done in {time.time()-t2:.1f}s — {len(segments)} segments found")

        log("Computing blob statistics...")
        t3 = time.time()
        rows, cc_labels, _ = blob_statistics_records(segments, mask_bool, pixel_size_um=PIXEL_SIZE_UM)
        df_blobs = pd.DataFrame(rows)
        df_blobs_clean = clean_length_width_for_plot(df_blobs)
        df_blobs_clean.to_csv(out_dir / "blob_metrics.csv", index=False)
        log(f"Blob stats done in {time.time()-t3:.1f}s — {len(df_blobs)} blobs")

        log("Saving intermediates...")
        np.savez_compressed(
            out_dir / "intermediates.npz",
            original_image=image_gray.astype(np.uint8),
            binary_mask=mask_u8.astype(np.uint8),
            blob_labels=cc_labels.astype(np.int32),
            skeleton=meta["skeleton"].astype(np.uint8),
            segment_label_image=seg_label_image.astype(np.int32),
        )

        log("Computing 2D spatial autocorrelation...")
        t3b = time.time()
        acf_path = out_dir / "autocorrelation_2d.html"
        autocorrelation_report(
            mask_u8,
            acf_path,
            pixel_size_um=PIXEL_SIZE_UM,
            title=f"2D Spatial Autocorrelation \u2014 {safe_name}",
        )
        log(f"Autocorrelation report done in {time.time()-t3b:.1f}s")

        log("Rendering 2D autocorrelation for frontend...")
        t3c = time.time()
        acf_npz_name = f"autocorrelation-{safe_name}.npz"
        acf_frontend = autocorrelation_for_frontend(
            mask_u8, pixel_size_um=PIXEL_SIZE_UM,
            npz_path=out_dir / acf_npz_name,
        )
        log(f"Frontend ACF rendered in {time.time()-t3c:.1f}s")

        log("Rendering overlays...")
        t4 = time.time()
        blob_overlay = annotate_blob_ids(render_blob_overlay(image_gray, cc_labels, alpha=0.6), cc_labels)
        skel_overlay = skeleton_overlay(mask_u8, meta["skeleton"])
        compact_overlay = compact_views(mask_u8, cc_labels, df_blobs_clean)

        cv2.imwrite(str(out_dir / "original.png"), image_gray)
        cv2.imwrite(str(out_dir / "binary_mask.png"), (mask_u8 * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / "blob_labeled.png"), blob_overlay)
        cv2.imwrite(str(out_dir / "skeleton_overlay.png"), skel_overlay)
        cv2.imwrite(str(out_dir / "compact_overlay.png"), compact_overlay)
        log(f"Overlays rendered in {time.time()-t4:.1f}s")

        self.cc_labels = cc_labels
        self.df_blobs = df_blobs_clean
        self.output_dir = out_dir
        self.branch_points = []

        log(f"TOTAL processing time: {time.time()-t0:.1f}s")

        # Build blob records for dashboard (safe JSON serializable)
        def safe(v):
            import math
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v

        blob_records = []
        for _, row in df_blobs_clean.iterrows():
            blob_records.append({k: safe(row.get(k)) for k in [
                "blob_id","morphology_2d","circularity_2d","eccentricity_2d",
                "area_px","n_segments","total_skeleton_length_px","max_segment_length_px",
                "width_mean_length_weighted_px","length_clean","width_clean",
                "centroid_x","centroid_y",
            ]})

        return {
            "width": int(image_gray.shape[1]),
            "height": int(image_gray.shape[0]),
            "original_width": w0,
            "original_height": h0,
            "scale": round(scale, 4),
            "output_dir": str(out_dir),
            "reports": {
                "autocorrelation_html": f"{out_dir.name}/autocorrelation_2d.html",
                "autocorrelation_npz": f"{out_dir.name}/{acf_npz_name}",
            },
            "blobs": blob_records,
            "images": {
                "original": to_png_data_url(image_gray),
                "binary_mask": to_png_data_url((mask_u8 * 255).astype(np.uint8)),
                "blob_labeled": to_png_data_url(blob_overlay),
                "skeleton_overlay": to_png_data_url(skel_overlay),
                "compact_overlay": to_png_data_url(compact_overlay),
                "autocorrelation_2d": acf_frontend["png_data_url"],
            },
            "acf_radial": {
                "radius": acf_frontend["radius"],
                "profile": acf_frontend["profile"],
                "std": acf_frontend["std"],
                "unit_label": acf_frontend["unit_label"],
                "correlation_length": acf_frontend["correlation_length"],
                "correlation_threshold": acf_frontend["correlation_threshold"],
                "half_max_distance": acf_frontend["half_max_distance"],
                "peak_value": acf_frontend["peak_value"],
            },
        }

    def get_blob_info(self, x: int, y: int) -> dict[str, Any]:
        if self.cc_labels is None or self.df_blobs is None:
            raise ValueError("No active session")
        if y < 0 or x < 0 or y >= self.cc_labels.shape[0] or x >= self.cc_labels.shape[1]:
            return {"blob_id": 0}
        blob_id = int(self.cc_labels[y, x])
        if blob_id <= 0:
            return {"blob_id": 0}
        row = blob_summary_row(self.df_blobs, blob_id)
        return row if row is not None else {"blob_id": blob_id}

    def finalize_branches(self, points: list[tuple[int, int]]) -> dict[str, Any]:
        if self.cc_labels is None or self.df_blobs is None or self.output_dir is None:
            raise ValueError("No active session")
        self.branch_points = points
        df = self.df_blobs.copy()
        branch_blob_ids = []
        for x, y in self.branch_points:
            if 0 <= y < self.cc_labels.shape[0] and 0 <= x < self.cc_labels.shape[1]:
                bid = int(self.cc_labels[y, x])
                if bid > 0:
                    branch_blob_ids.append(bid)
        branch_blob_ids = sorted(set(branch_blob_ids))
        df["user_marked_branching"] = df["blob_id"].isin(branch_blob_ids)
        counts = {bid: 0 for bid in branch_blob_ids}
        for x, y in self.branch_points:
            if 0 <= y < self.cc_labels.shape[0] and 0 <= x < self.cc_labels.shape[1]:
                bid = int(self.cc_labels[y, x])
                if bid in counts:
                    counts[bid] += 1
        df["user_marked_branch_points"] = df["blob_id"].map(lambda b: counts.get(int(b), 0))
        csv_path = self.output_dir / "blob_metrics_with_branching.csv"
        df.to_csv(csv_path, index=False)
        np.savez_compressed(
            self.output_dir / "branch_points.npz",
            branch_points=np.array(self.branch_points, dtype=np.int32),
            branch_blob_ids=np.array(branch_blob_ids, dtype=np.int32),
        )
        (self.output_dir / "branch_points.json").write_text(
            json.dumps({"points": [{"x": x, "y": y} for x, y in self.branch_points]}),
            encoding="utf-8",
        )
        self.df_blobs = df
        return {"csv_path": str(csv_path), "marked_blobs": branch_blob_ids}
