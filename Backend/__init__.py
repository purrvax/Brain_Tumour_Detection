"""NeuroScan Brain Tumour Detection & Segmentation Backend.

Classical computer vision and marker-based watershed pipeline adapted from
research image processing notebook. Processes single in-memory MRI scans.
"""

from Backend.pipeline import BrainTumourPipeline, run_pipeline
from Backend.segmentation import (
    SegmentationMetrics,
    SegmentationResult,
    extract_brain_roi,
    run_tumour_segmentation,
)

__all__ = [
    "BrainTumourPipeline",
    "SegmentationMetrics",
    "SegmentationResult",
    "extract_brain_roi",
    "run_pipeline",
    "run_tumour_segmentation",
]
