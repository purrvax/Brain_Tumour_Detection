"""Backend analysis bridge for NeuroScan Brain Tumour Detection.

Directly connects the Streamlit frontend to the classical computer vision
and marker-based watershed pipeline implemented in `Backend/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure project root is in sys.path so Backend is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.pipeline import run_pipeline
from utils.image_processing import (
    build_comparison_export_bytes,
    render_segmentation_overlay,
)

# Sequential real pipeline stages shown in the interactive UI progress bar
ANALYSIS_STAGES: List[Tuple[float, str, str]] = [
    (0.25, "Gaussian Smoothing & Contrast Enhancement", "Applying 5×5 Gaussian blur, intensity shift, and median filter"),
    (0.50, "Intracranial ROI Extraction", "Detecting cranial envelope and computing outer-skull distance transform exclusion"),
    (0.75, "Adaptive Thresholding & Markers", "Computing Otsu floor + 85th percentile inside ROI and distance transform"),
    (1.00, "Watershed Segmentation & Quantification", "Flooding relief surface from markers, tracing boundary, and calculating metrics"),
]


def run_analysis(
    image: Image.Image,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run real brain MRI analysis using the Backend computer vision pipeline.

    Takes a single in-memory PIL Image (no zip file required!) and returns
    structured segmentation results, metrics, and visualization overlays.
    """
    meta = dict(metadata or {})
    if "dimensions_label" not in meta:
        meta["dimensions_label"] = f"{image.width} × {image.height} px"
    if "format" not in meta:
        meta["format"] = "PNG"

    # Execute backend computer vision pipeline
    result = run_pipeline(image_input=image, metadata=meta)

    # Build comparison report export bytes
    export_bytes = build_comparison_export_bytes(
        original_image=result["original_image"],
        overlay_image=result["overlay_image"],
        metadata=meta,
        area_px=result["region_area_px"],
        tumor_detected=result["tumor_detected"],
        perimeter_px=float(result.get("perimeter_px", 0.0)),
    )
    result["export_bytes"] = export_bytes

    return result
