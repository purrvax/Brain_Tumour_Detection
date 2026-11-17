"""High-level orchestration pipeline for Brain Tumour Detection and Segmentation.

Provides unified entry points for both local Python integration (Streamlit)
and remote API execution (FastAPI).
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image

from Backend.segmentation import (
    SegmentationMetrics,
    SegmentationResult,
    run_tumour_segmentation,
)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an image array (RGB, grayscale, or float) to a PIL Image."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr)


def array_to_base64(arr: np.ndarray, fmt: str = "PNG") -> str:
    """Encode an image NumPy array to a base64 string."""
    pil_img = array_to_pil(arr)
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_blended_overlay(
    original_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (204, 125, 114),  # Coral-rose
    contour_rgb: Tuple[int, int, int] = (168, 86, 78),  # Terracotta
    fill_alpha: float = 0.45,
) -> np.ndarray:
    """Generate a clean blended overlay with filled region and crisp perimeter."""
    orig = original_rgb.astype(np.float32)
    h, w, _ = orig.shape

    bin_mask = (mask > 0).astype(np.float32)

    if bin_mask.shape != (h, w):
        bin_mask = cv.resize(bin_mask, (w, h), interpolation=cv.INTER_NEAREST)

    # Compute contour via morphological gradient / erosion
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eroded = cv.erode((bin_mask * 255).astype(np.uint8), kernel, iterations=1)
    contour = ((bin_mask * 255).astype(np.uint8) - eroded) > 0

    # Fill layer
    fill_layer = np.full_like(orig, color_rgb, dtype=np.float32)
    alpha = (bin_mask * fill_alpha)[..., None]
    blended = orig * (1.0 - alpha) + fill_layer * alpha

    # Contour outline
    contour_mask = contour[..., None]
    contour_layer = np.full_like(orig, contour_rgb, dtype=np.float32)
    blended = np.where(contour_mask, contour_layer, blended)

    return np.clip(blended, 0, 255).astype(np.uint8)


class BrainTumourPipeline:
    """Configurable pipeline for detecting and segmenting brain tumours in MRI scans."""

    def __init__(
        self,
        roi_margin: float = 0.15,
        percentile: float = 85.0,
        min_tumor_area_px: int = 30,
        pixel_spacing_mm: Optional[float] = None,
    ):
        self.roi_margin = roi_margin
        self.percentile = percentile
        self.min_tumor_area_px = min_tumor_area_px
        self.pixel_spacing_mm = pixel_spacing_mm

    def analyze(
        self,
        image_input: Union[Image.Image, np.ndarray, bytes, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute complete analysis and return rich structured output.

        Compatible with both Streamlit frontend session state and REST API schemas.
        """
        meta = dict(metadata or {})
        raw_res: SegmentationResult = run_tumour_segmentation(
            image_input=image_input,
            roi_margin=self.roi_margin,
            percentile=self.percentile,
            min_tumor_area_px=self.min_tumor_area_px,
        )

        orig_pil = array_to_pil(raw_res.original_rgb)
        mask_u8 = raw_res.tumour_mask
        mask_norm = (mask_u8.astype(np.float32) / 255.0)

        # Generate blended soft overlay
        blended_arr = create_blended_overlay(raw_res.original_rgb, mask_u8)
        overlay_pil = array_to_pil(blended_arr)

        area_px = raw_res.metrics.estimated_area_px
        area_mm2 = (
            float(area_px * (self.pixel_spacing_mm ** 2))
            if self.pixel_spacing_mm is not None
            else None
        )

        stages_pil: Dict[str, Image.Image] = {
            name: array_to_pil(arr) for name, arr in raw_res.stages.items()
        }

        stages_b64: Dict[str, str] = {
            name: array_to_base64(arr) for name, arr in raw_res.stages.items()
        }

        return {
            "tumor_detected": raw_res.tumor_detected,
            "status_message": raw_res.status_message,
            "original_image": orig_pil,
            "overlay_image": overlay_pil,
            "overlay_contour": array_to_pil(raw_res.overlay_contour),
            "mask": mask_norm,
            "mask_u8": mask_u8,
            "region_area_px": area_px,
            "perimeter_px": round(raw_res.metrics.perimeter_px, 2),
            "area_mm2": area_mm2,
            "bounding_box": {
                "x": raw_res.metrics.bounding_box[0],
                "y": raw_res.metrics.bounding_box[1],
                "width": raw_res.metrics.bounding_box[2],
                "height": raw_res.metrics.bounding_box[3],
            },
            "centroid": {
                "x": raw_res.metrics.centroid[0],
                "y": raw_res.metrics.centroid[1],
            },
            "circularity": round(raw_res.metrics.circularity, 3),
            "brain_roi_area_px": raw_res.metrics.brain_roi_area_px,
            "thresholds": {
                "otsu": round(raw_res.metrics.otsu_threshold, 2),
                "percentile_85": round(raw_res.metrics.percentile_threshold, 2),
                "applied": round(raw_res.metrics.applied_threshold, 2),
            },
            "stages_pil": stages_pil,
            "stages_b64": stages_b64,
            "image_dimensions": f"{orig_pil.width} × {orig_pil.height} px",
            "image_format": meta.get("format", "PNG"),
            "analysis_status": "Analysis Complete" if raw_res.tumor_detected else "Scan Clear",
            "is_demo": False,
            "disclaimer": "Classical CV Watershed Segmentation",
        }


# Global default instance for convenience
_default_pipeline = BrainTumourPipeline()


def run_pipeline(
    image_input: Union[Image.Image, np.ndarray, bytes, str],
    metadata: Optional[Dict[str, Any]] = None,
    roi_margin: float = 0.15,
    percentile: float = 85.0,
) -> Dict[str, Any]:
    """Convenience helper to run the pipeline with standard default settings."""
    pipeline = BrainTumourPipeline(roi_margin=roi_margin, percentile=percentile)
    return pipeline.analyze(image_input=image_input, metadata=metadata)
