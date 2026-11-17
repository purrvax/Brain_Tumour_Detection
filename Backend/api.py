"""FastAPI REST Service for Brain Tumour Detection and Segmentation.

Exposes endpoints for single-scan MRI analysis using the classical computer
vision watershed pipeline.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Backend.pipeline import BrainTumourPipeline, array_to_base64
from Backend.segmentation import _to_rgb_array

app = FastAPI(
    title="NeuroScan Brain Tumour Detection API",
    description="Classical Computer Vision & Watershed Segmentation Engine for Brain MRI Analysis",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Centroid(BaseModel):
    x: int
    y: int


class ThresholdMetrics(BaseModel):
    otsu: float
    percentile_85: float
    applied: float


class AnalysisMetrics(BaseModel):
    tumor_detected: bool
    estimated_area_px: int
    perimeter_px: float
    area_mm2: Optional[float] = None
    bounding_box: BoundingBox
    centroid: Centroid
    circularity: float
    brain_roi_area_px: int
    thresholds: ThresholdMetrics
    status_message: str


class FullAnalysisResponse(BaseModel):
    success: bool
    metrics: AnalysisMetrics
    overlay_base64: str = Field(description="Base64 encoded PNG of the tumour overlay")
    mask_base64: str = Field(description="Base64 encoded PNG of the binary tumour mask")
    stages_base64: Dict[str, str] = Field(
        default_factory=dict,
        description="Base64 encoded images of the 9 pipeline stages",
    )


@app.get("/")
def root():
    return {
        "service": "NeuroScan Brain Tumour Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/analyze", response_model=FullAnalysisResponse)
def analyze_mri_scan(
    file: UploadFile = File(..., description="Brain MRI image file (PNG, JPG, JPEG)"),
    roi_margin: float = Query(0.15, ge=0.05, le=0.35, description="Cranial margin fraction"),
    percentile: float = Query(85.0, ge=50.0, le=98.0, description="Intensity percentile floor"),
    min_area_px: int = Query(30, ge=5, le=5000, description="Minimum area in pixels to qualify as tumour"),
    spacing_mm: Optional[float] = Query(None, description="In-plane pixel spacing in mm"),
):
    """Analyze a single uploaded MRI scan and return full segmentation masks and metrics."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing filename")

    content_type = file.content_type or ""
    allowed_types = ("image/png", "image/jpeg", "image/jpg", "application/octet-stream")
    if content_type and not any(t in content_type for t in allowed_types):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {content_type}. Please upload PNG or JPEG.",
        )

    try:
        # Normalize Query params if function was invoked directly
        roi_val = float(roi_margin.default if hasattr(roi_margin, "default") else roi_margin)
        pct_val = float(percentile.default if hasattr(percentile, "default") else percentile)
        min_area_val = int(min_area_px.default if hasattr(min_area_px, "default") else min_area_px)
        spacing_val = (
            float(spacing_mm.default if hasattr(spacing_mm, "default") else spacing_mm)
            if spacing_mm is not None and getattr(spacing_mm, "default", None) is not None
            else None
        )

        file_bytes = file.file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        pipeline = BrainTumourPipeline(
            roi_margin=roi_val,
            percentile=pct_val,
            min_tumor_area_px=min_area_val,
            pixel_spacing_mm=spacing_val,
        )

        res = pipeline.analyze(
            image_input=file_bytes,
            metadata={"filename": file.filename},
        )

        mask_u8 = res["mask_u8"]
        mask_b64 = array_to_base64(mask_u8)

        overlay_buf = BytesIO()
        res["overlay_image"].save(overlay_buf, format="PNG")
        overlay_b64 = base64.b64encode(overlay_buf.getvalue()).decode("ascii")

        metrics_obj = AnalysisMetrics(
            tumor_detected=res["tumor_detected"],
            estimated_area_px=res["region_area_px"],
            perimeter_px=res["perimeter_px"],
            area_mm2=res["area_mm2"],
            bounding_box=BoundingBox(**res["bounding_box"]),
            centroid=Centroid(**res["centroid"]),
            circularity=res["circularity"],
            brain_roi_area_px=res["brain_roi_area_px"],
            thresholds=ThresholdMetrics(**res["thresholds"]),
            status_message=res["status_message"],
        )

        return FullAnalysisResponse(
            success=True,
            metrics=metrics_obj,
            overlay_base64=overlay_b64,
            mask_base64=mask_b64,
            stages_base64=res["stages_b64"],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(exc)}") from exc
