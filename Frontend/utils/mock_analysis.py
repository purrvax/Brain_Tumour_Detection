"""Simulated analysis pipeline and future backend integration hook.

This module intentionally isolates demonstration segmentation logic from the
Streamlit UI components. When a trained ML model is ready inside `Backend/`,
replace the body of `run_analysis(...)` to call the model inference pipeline
while preserving the returned schema.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from utils.image_processing import (
    build_comparison_export_bytes,
    preprocess_mri_image,
    render_segmentation_overlay,
)

# Sequential simulated progress stages shown in the analysis column
ANALYSIS_STAGES: List[Tuple[float, str, str]] = [
    (0.24, "Loading MRI image", "Reading uploaded scan matrix and spatial metadata"),
    (0.52, "Preprocessing scan", "Normalising pixel intensities and cranial framing"),
    (0.78, "Analysing image regions", "Running simulated spatial region inspection pass"),
    (1.00, "Generating segmentation visualisation", "Compositing illustrative overlay and metrics"),
]


def _compute_deterministic_seed(image_arr: np.ndarray) -> int:
    """Create a stable seed from image shape and intensity statistics."""
    h, w = image_arr.shape
    center_crop = image_arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    mean_val = float(np.mean(center_crop)) if center_crop.size else 0.5
    digest = hashlib.sha256(f"{h}x{w}:{mean_val:.5f}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _locate_cranial_focus(image_arr: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float]:
    """Locate an anatomically plausible anchor point inside the bright tissue area."""
    h, w = image_arr.shape
    # Identify foreground tissue above dark MRI background
    low, high = np.percentile(image_arr, [15, 92])
    threshold = max(0.14, float(low + 0.22 * (high - low)))
    fg = image_arr > threshold

    ys, xs = np.where(fg)
    # Fallback cleanly if image is largely uniform or extremely bright/dark
    if ys.size < (h * w * 0.08) or ys.size > (h * w * 0.94):
        cx = w * float(rng.uniform(0.36, 0.64))
        cy = h * float(rng.uniform(0.34, 0.56))
        scale = min(h, w) * float(rng.uniform(0.085, 0.115))
        return cx, cy, scale

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    span_x = max(24, x_max - x_min)
    span_y = max(24, y_max - y_min)

    # Choose an upper/mid cerebral hemisphere offset within tissue bounds
    side = -1.0 if rng.random() < 0.5 else 1.0
    cx = float(np.mean(xs)) + side * span_x * float(rng.uniform(0.14, 0.24))
    cy = y_min + span_y * float(rng.uniform(0.30, 0.48))
    scale = min(span_x, span_y) * float(rng.uniform(0.095, 0.135))
    return cx, cy, max(12.0, scale)


def generate_illustrative_mask(
    preprocessed_image: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a smooth, organic, multi-lobed illustrative region mask in [0, 1]."""
    h, w = preprocessed_image.shape
    resolved_seed = seed if seed is not None else _compute_deterministic_seed(preprocessed_image)
    rng = np.random.default_rng(resolved_seed)

    cx, cy, base_radius = _locate_cranial_focus(preprocessed_image, rng)

    yy, xx = np.ogrid[:h, :w]
    dx = xx - cx
    dy = yy - cy

    # Slight elliptical tilt for organic irregular morphology
    tilt = float(rng.uniform(-0.6, 0.6))
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t

    aspect = float(rng.uniform(1.12, 1.38))
    dist = np.sqrt((rx / aspect) ** 2 + ry ** 2)
    theta = np.arctan2(ry, rx)

    # Radial harmonic modulation creates natural non-circular lobes
    p1, p2, p3 = rng.uniform(0, 2 * np.pi, size=3)
    a1 = float(rng.uniform(0.12, 0.20))
    a2 = float(rng.uniform(0.08, 0.14))
    a3 = float(rng.uniform(0.04, 0.08))
    radial_profile = base_radius * (
        1.0
        + a1 * np.cos(2.0 * theta + p1)
        + a2 * np.sin(3.0 * theta + p2)
        + a3 * np.cos(5.0 * theta + p3)
    )

    raw_mask = (dist <= radial_profile).astype(np.uint8) * 255
    feather_radius = max(1.4, min(4.2, min(h, w) / 140.0))
    soft_pil = Image.fromarray(raw_mask, mode="L").filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )
    soft_mask = np.asarray(soft_pil, dtype=np.float32) / 255.0
    return np.clip(soft_mask, 0.0, 1.0)


def run_mock_inference(
    preprocessed_image: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Produce simulated segmentation outputs for demonstration purposes only."""
    mask = generate_illustrative_mask(preprocessed_image, seed=seed)
    region_area_px = int(np.sum(mask >= 0.45))

    return {
        "mask": mask,
        "region_area_px": region_area_px,
        "analysis_status": "Demo Complete",
        "is_demo": True,
        "disclaimer": "Illustrative Region — Not a Model Prediction",
    }


def run_analysis(
    image: Image.Image,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run brain MRI analysis and return structured visualisation data.

    Future Backend Integration:
        Replace `run_mock_inference(...)` below with a call to your trained
        segmentation model in `Backend/` (returning a spatial probability/binary
        mask aligned to `image`), while keeping the returned dictionary schema intact.
    """
    meta = dict(metadata or {})
    if "dimensions_label" not in meta:
        meta["dimensions_label"] = f"{image.width} × {image.height} px"
    if "format" not in meta:
        meta["format"] = "PNG"

    preprocessed = preprocess_mri_image(image)
    inference_output = run_mock_inference(preprocessed)

    overlay_image = render_segmentation_overlay(
        base_image=image,
        mask=inference_output["mask"],
    )

    export_bytes = build_comparison_export_bytes(
        original_image=image,
        overlay_image=overlay_image,
        metadata=meta,
        area_px=inference_output["region_area_px"],
    )

    return {
        "original_image": image,
        "overlay_image": overlay_image,
        "mask": inference_output["mask"],
        "region_area_px": inference_output["region_area_px"],
        "image_dimensions": meta["dimensions_label"],
        "image_format": meta["format"],
        "analysis_status": inference_output["analysis_status"],
        "is_demo": inference_output["is_demo"],
        "disclaimer": inference_output["disclaimer"],
        "export_bytes": export_bytes,
    }
