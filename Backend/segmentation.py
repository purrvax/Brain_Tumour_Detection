"""Brain Tumour Segmentation and Analysis Engine using Classical Computer Vision.

Adapted from the research pipeline in Brain_Tumour_Segmentation_(1).ipynb.
Executes on an in-memory single MRI image (PIL Image, NumPy array, or bytes)
without requiring dataset ZIP files or disk extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image


@dataclass
class SegmentationMetrics:
    """Quantitative measurements for the detected tumour candidate region."""

    tumor_detected: bool
    estimated_area_px: int = 0
    perimeter_px: float = 0.0
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    centroid: Tuple[int, int] = (0, 0)  # cx, cy
    circularity: float = 0.0  # 4 * pi * area / perimeter^2
    brain_roi_area_px: int = 0
    otsu_threshold: float = 0.0
    percentile_threshold: float = 0.0
    applied_threshold: float = 0.0


@dataclass
class SegmentationResult:
    """Full structured output of the brain tumour segmentation pipeline."""

    tumor_detected: bool
    metrics: SegmentationMetrics
    original_rgb: np.ndarray
    tumour_mask: np.ndarray  # Binary uint8 mask (0 or 255)
    overlay_contour: np.ndarray  # RGB image with green contour drawn
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    status_message: str = ""


def _to_rgb_array(image_input: Union[Image.Image, np.ndarray, bytes, Path, str]) -> np.ndarray:
    """Normalize any image input representation into an RGB uint8 NumPy array."""
    if isinstance(image_input, (str, Path)):
        img_bgr = cv.imread(str(image_input), cv.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not load image from path: {image_input}")
        return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    if isinstance(image_input, bytes):
        arr = np.frombuffer(image_input, dtype=np.uint8)
        img_bgr = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback via PIL
            with Image.open(BytesIO(image_input)) as pil_img:
                return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    if isinstance(image_input, Image.Image):
        return np.asarray(image_input.convert("RGB"), dtype=np.uint8)

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return cv.cvtColor(arr, cv.COLOR_GRAY2RGB)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                return cv.cvtColor(arr, cv.COLOR_RGBA2RGB)
            if arr.shape[2] == 3:
                return arr.copy()
            if arr.shape[2] == 1:
                return cv.cvtColor(arr, cv.COLOR_GRAY2RGB)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def extract_brain_roi(
    blur: np.ndarray,
    roi_margin: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract intracranial brain tissue region of interest, excluding outer skull.

    Corresponds to Cell 14 in the research notebook.
    """
    X = blur.shape[0]
    blur_gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY) if blur.ndim == 3 else blur

    # Separate the head from black background
    _, head = cv.threshold(blur_gray, 20, 255, cv.THRESH_BINARY)

    # Close small gaps in the head region
    k7 = max(3, int(7 * X / 100) | 1)
    kernel7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k7, k7))
    head_closed = cv.morphologyEx(head, cv.MORPH_CLOSE, kernel7)

    # Keep the largest external contour (the skull/head)
    contours, _ = cv.findContours(head_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    head_filled = np.zeros_like(blur_gray)
    if contours:
        largest_head = max(contours, key=cv.contourArea)
        cv.drawContours(head_filled, [largest_head], -1, 255, cv.FILLED)
    else:
        # Fallback if threshold didn't capture head: use central ellipse
        h, w = blur_gray.shape
        cv.ellipse(head_filled, (w // 2, h // 2), (int(w * 0.42), int(h * 0.45)), 0, 0, 360, 255, -1)

    # Exclude pixels close to the outer head boundary via distance transform
    dist_head = cv.distanceTransform(head_filled, cv.DIST_L2, 5)
    margin_threshold = roi_margin * X

    # Guard against overly aggressive margin if head is small
    if dist_head.max() <= margin_threshold:
        margin_threshold = dist_head.max() * 0.5

    roi = (dist_head > margin_threshold).astype(np.uint8) * 255
    if cv.countNonZero(roi) == 0:
        roi = head_filled.copy()

    return roi, head_filled


def run_tumour_segmentation(
    image_input: Union[Image.Image, np.ndarray, bytes, Path, str],
    roi_margin: float = 0.15,
    percentile: float = 85.0,
    min_tumor_area_px: int = 30,
) -> SegmentationResult:
    """Run full brain tumour detection and segmentation on a single image.

    This implements the complete 10-step OpenCV image processing and
    marker-based watershed pipeline:
      1. Image ingestion & RGB normalization
      2. Gaussian smoothing (5x5, sigma=2)
      3. First intensity enhancement (orig + (blur - 100))
      4. Median filtering (5x5)
      5. Morphological gradient & 2nd enhancement (median + gradient)
      6. Intracranial Brain ROI extraction via skull distance transform
      7. Otsu + 85th percentile thresholding inside Brain ROI + morphological opening/closing
      8. Euclidean distance transform & watershed marker generation (sure background/foreground/unknown)
      9. Marker-based watershed segmentation
      10. Candidate tumour extraction (largest label inside ROI) & contour quantification
    """
    orig_rgb = _to_rgb_array(image_input)
    H, W, _ = orig_rgb.shape
    X = H

    stages: Dict[str, np.ndarray] = {
        "input_rgb": orig_rgb.copy(),
    }

    # Step 2: Gaussian filtering
    blur = cv.GaussianBlur(orig_rgb, (5, 5), 2)
    stages["gaussian_blur"] = blur.copy()

    # Step 3: First image enhancement
    enh = cv.add(orig_rgb, cv.add(blur, -100))
    stages["first_enhancement"] = enh.copy()

    # Step 4: Median filtering
    median = cv.medianBlur(enh, 5)
    stages["median_filtering"] = median.copy()

    # Step 5: Morphological gradient and second enhancement
    kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    gradient = cv.morphologyEx(median, cv.MORPH_GRADIENT, kernel_cross)
    enh2 = cv.add(median, gradient)
    stages["morphological_gradient"] = gradient.copy()
    stages["second_enhancement"] = enh2.copy()

    # Step 6: Brain region of interest (ROI) extraction
    roi, head_filled = extract_brain_roi(blur, roi_margin=roi_margin)
    stages["brain_roi"] = roi.copy()

    # Step 7: Thresholding and morphological cleanup
    enh2_gray = cv.cvtColor(enh2, cv.COLOR_RGB2GRAY)
    brain_values = enh2_gray[roi > 0]

    if brain_values.size == 0:
        brain_values = enh2_gray[enh2_gray > 15]
    if brain_values.size == 0:
        brain_values = enh2_gray.flatten()

    otsu_threshold, _ = cv.threshold(
        brain_values.reshape(-1, 1),
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU,
    )
    pct_val = float(np.percentile(brain_values, percentile))
    threshold_value = float(max(otsu_threshold, pct_val))

    threshold_roi = ((enh2_gray >= threshold_value) & (roi > 0)).astype(np.uint8) * 255
    stages["threshold_roi"] = threshold_roi.copy()

    k3 = max(3, int(3 * X / 100) | 1)
    k5 = max(5, int(5 * X / 100) | 1)
    kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k3, k3))
    kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k5, k5))

    mask = cv.morphologyEx(threshold_roi, cv.MORPH_OPEN, kernel3)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel5)
    stages["cleaned_mask"] = mask.copy()

    # Step 8: Distance transform and watershed marker generation
    non_zero_mask = int(cv.countNonZero(mask))
    brain_roi_area = int(cv.countNonZero(roi))

    overlay_contour = orig_rgb.copy()

    if non_zero_mask == 0:
        # Healthy scan / no high-intensity abnormal tissue inside ROI
        metrics = SegmentationMetrics(
            tumor_detected=False,
            estimated_area_px=0,
            perimeter_px=0.0,
            bounding_box=(0, 0, 0, 0),
            centroid=(0, 0),
            circularity=0.0,
            brain_roi_area_px=brain_roi_area,
            otsu_threshold=float(otsu_threshold),
            percentile_threshold=pct_val,
            applied_threshold=threshold_value,
        )
        return SegmentationResult(
            tumor_detected=False,
            metrics=metrics,
            original_rgb=orig_rgb,
            tumour_mask=np.zeros((H, W), dtype=np.uint8),
            overlay_contour=overlay_contour,
            stages=stages,
            status_message="No tumour detected inside intracranial brain ROI.",
        )

    # Estimate sure background
    sure_background = cv.dilate(mask, kernel3, iterations=3)
    stages["sure_background"] = sure_background.copy()

    # Distance transform identifies the interior of foreground regions
    distance = cv.distanceTransform(mask, cv.DIST_L2, 5)
    max_dist = float(distance.max())

    if max_dist > 0:
        dist_display = np.uint8(np.clip(distance / max_dist * 255.0, 0, 255))
    else:
        dist_display = np.zeros((H, W), dtype=np.uint8)
    stages["distance_transform"] = dist_display

    if max_dist == 0:
        metrics = SegmentationMetrics(
            tumor_detected=False,
            estimated_area_px=0,
            perimeter_px=0.0,
            brain_roi_area_px=brain_roi_area,
            otsu_threshold=float(otsu_threshold),
            percentile_threshold=pct_val,
            applied_threshold=threshold_value,
        )
        return SegmentationResult(
            tumor_detected=False,
            metrics=metrics,
            original_rgb=orig_rgb,
            tumour_mask=np.zeros((H, W), dtype=np.uint8),
            overlay_contour=overlay_contour,
            stages=stages,
            status_message="No tumour detected (distance transform empty).",
        )

    # Identify sure foreground
    _, sure_foreground = cv.threshold(distance, 0.5 * max_dist, 255, cv.THRESH_BINARY)
    sure_foreground = np.uint8(sure_foreground)
    stages["sure_foreground"] = sure_foreground.copy()

    # Unknown region lies between sure foreground and sure background
    unknown = cv.subtract(sure_background, sure_foreground)
    stages["unknown_region"] = unknown.copy()

    # Connected components provide watershed markers
    num_markers, markers = cv.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    marker_display = np.uint8(np.clip(markers / max(1, markers.max()) * 255.0, 0, 255))
    stages["markers"] = cv.applyColorMap(marker_display, cv.COLORMAP_JET)

    # Step 9: Watershed segmentation
    watershed_image = cv.cvtColor(orig_rgb, cv.COLOR_RGB2BGR)
    markers = cv.watershed(watershed_image, markers)

    watershed_display = watershed_image.copy()
    watershed_display[markers == -1] = [0, 0, 255]
    stages["watershed_boundaries"] = cv.cvtColor(watershed_display, cv.COLOR_BGR2RGB)

    # Step 10: Candidate region extraction and contour measurement
    labels = [label for label in np.unique(markers) if label > 1]

    if not labels:
        metrics = SegmentationMetrics(
            tumor_detected=False,
            estimated_area_px=0,
            perimeter_px=0.0,
            brain_roi_area_px=brain_roi_area,
            otsu_threshold=float(otsu_threshold),
            percentile_threshold=pct_val,
            applied_threshold=threshold_value,
        )
        return SegmentationResult(
            tumor_detected=False,
            metrics=metrics,
            original_rgb=orig_rgb,
            tumour_mask=np.zeros((H, W), dtype=np.uint8),
            overlay_contour=overlay_contour,
            stages=stages,
            status_message="Watershed did not produce candidate foreground regions.",
        )

    # Select largest watershed region
    largest_label = max(labels, key=lambda label: int(np.count_nonzero(markers == label)))
    tumour_mask = (markers == largest_label).astype(np.uint8) * 255
    tumour_mask = cv.bitwise_and(tumour_mask, roi)
    stages["candidate_tumour_mask"] = tumour_mask.copy()

    # Find the outer contour of the candidate mask
    contours, _ = cv.findContours(tumour_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        metrics = SegmentationMetrics(
            tumor_detected=False,
            estimated_area_px=0,
            perimeter_px=0.0,
            brain_roi_area_px=brain_roi_area,
            otsu_threshold=float(otsu_threshold),
            percentile_threshold=pct_val,
            applied_threshold=threshold_value,
        )
        return SegmentationResult(
            tumor_detected=False,
            metrics=metrics,
            original_rgb=orig_rgb,
            tumour_mask=tumour_mask,
            overlay_contour=overlay_contour,
            stages=stages,
            status_message="No valid contour found in candidate region.",
        )

    largest_contour = max(contours, key=cv.contourArea)
    estimated_area_px = int(np.count_nonzero(tumour_mask))
    perimeter_px = float(cv.arcLength(largest_contour, True))

    # Physiological & contrast validation:
    # 1. Tumour must meet minimum size
    # 2. Tumour is a focal lesion and should not encompass the vast majority (> 65%) of the brain cavity
    # 3. Candidate region must exhibit elevated hyper-intensity relative to average brain tissue
    mean_candidate = float(np.mean(enh2_gray[tumour_mask > 0])) if estimated_area_px > 0 else 0.0
    mean_brain = float(np.mean(brain_values)) if brain_values.size > 0 else 0.0
    contrast_diff = mean_candidate - mean_brain

    is_too_small = estimated_area_px < min_tumor_area_px
    is_entire_brain = brain_roi_area > 0 and (estimated_area_px > 0.65 * brain_roi_area)
    is_insufficient_contrast = contrast_diff < 12.0 and brain_values.std() < 18.0

    if is_too_small or is_entire_brain or is_insufficient_contrast:
        reason = (
            f"Candidate area ({estimated_area_px} px) is below detection threshold"
            if is_too_small
            else "Candidate spans majority of brain parenchyma without focal lesion"
            if is_entire_brain
            else "Tissue intensity inside candidate is not significantly hyper-intense"
        )
        metrics = SegmentationMetrics(
            tumor_detected=False,
            estimated_area_px=estimated_area_px,
            perimeter_px=perimeter_px,
            brain_roi_area_px=brain_roi_area,
            otsu_threshold=float(otsu_threshold),
            percentile_threshold=pct_val,
            applied_threshold=threshold_value,
        )
        return SegmentationResult(
            tumor_detected=False,
            metrics=metrics,
            original_rgb=orig_rgb,
            tumour_mask=np.zeros((H, W), dtype=np.uint8),
            overlay_contour=orig_rgb.copy(),
            stages=stages,
            status_message=f"No tumour detected: {reason}.",
        )

    # Draw the candidate boundary on the original MRI (in vibrant green)
    cv.drawContours(overlay_contour, [largest_contour], -1, (0, 255, 0), 2)
    stages["candidate_boundary"] = overlay_contour.copy()

    # Calculate bounding box & centroid
    bx, by, bw, bh = cv.boundingRect(largest_contour)
    M = cv.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = bx + bw // 2, by + bh // 2

    circularity = float(4.0 * np.pi * estimated_area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0

    metrics = SegmentationMetrics(
        tumor_detected=True,
        estimated_area_px=estimated_area_px,
        perimeter_px=perimeter_px,
        bounding_box=(int(bx), int(by), int(bw), int(bh)),
        centroid=(int(cx), int(cy)),
        circularity=circularity,
        brain_roi_area_px=brain_roi_area,
        otsu_threshold=float(otsu_threshold),
        percentile_threshold=pct_val,
        applied_threshold=threshold_value,
    )

    return SegmentationResult(
        tumor_detected=True,
        metrics=metrics,
        original_rgb=orig_rgb,
        tumour_mask=tumour_mask,
        overlay_contour=overlay_contour,
        stages=stages,
        status_message="Tumour detected via watershed segmentation.",
    )
