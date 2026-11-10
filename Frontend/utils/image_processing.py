"""Image validation, preprocessing, and segmentation overlay rendering utilities.

Keeps image handling and visual rendering strictly separated from model inference
and UI layout components.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

MAX_FILE_SIZE_MB: float = 10.0
MAX_FILE_SIZE_BYTES: int = int(MAX_FILE_SIZE_MB * 1024 * 1024)
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("png", "jpg", "jpeg")
SUPPORTED_FORMATS_PIL: Tuple[str, ...] = ("PNG", "JPEG")

# Palette constants for segmentation rendering (zero blue shades)
OVERLAY_FILL_RGB: Tuple[int, int, int] = (204, 125, 114)      # Muted coral-rose
OVERLAY_CONTOUR_RGB: Tuple[int, int, int] = (168, 86, 78)    # Deep terracotta-rose
EXPORT_BG_RGB: Tuple[int, int, int] = (248, 247, 242)        # Warm ivory #F8F7F2
EXPORT_CARD_RGB: Tuple[int, int, int] = (242, 240, 233)      # Soft cream #F2F0E9
EXPORT_BORDER_RGB: Tuple[int, int, int] = (218, 212, 198)    # Thin beige border
EXPORT_TEXT_RGB: Tuple[int, int, int] = (41, 43, 39)         # Dark charcoal #292B27
EXPORT_MUTED_RGB: Tuple[int, int, int] = (119, 120, 111)     # Muted grey #77786F
EXPORT_SAGE_RGB: Tuple[int, int, int] = (89, 102, 79)        # Deep olive #59664F


def format_file_size(num_bytes: int) -> str:
    """Format byte count into a human-readable KB or MB string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kb = num_bytes / 1024.0
    if kb < 1024.0:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    return f"{mb:.2f} MB"


def validate_uploaded_file(
    file_bytes: bytes,
    filename: str,
) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file extension, byte size, and image integrity."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        return False, "Unsupported file format. Please upload a PNG, JPG, or JPEG image."

    if not file_bytes:
        return False, "The uploaded file appears to be empty."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File size ({format_file_size(len(file_bytes))}) exceeds the "
            f"{MAX_FILE_SIZE_MB:.0f} MB maximum limit.",
        )

    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img.verify()
        with Image.open(BytesIO(file_bytes)) as img:
            if img.format not in SUPPORTED_FORMATS_PIL:
                return (
                    False,
                    "File contents do not match a valid PNG or JPEG image.",
                )
            width, height = img.size
            if width < 32 or height < 32:
                return (
                    False,
                    "Image resolution is too small (minimum 32 × 32 pixels).",
                )
    except Exception:
        return False, "Unable to decode the uploaded file as a valid MRI image."

    return True, None


def load_mri_image(file_bytes: bytes, filename: str) -> Tuple[Image.Image, Dict[str, Any]]:
    """Load an uploaded MRI image, normalize orientation, and extract metadata."""
    with Image.open(BytesIO(file_bytes)) as raw_img:
        detected_format = raw_img.format or Path(filename).suffix.upper().lstrip(".")
        source_mode = raw_img.mode
        transposed = ImageOps.exif_transpose(raw_img)
        rgb_image = transposed.convert("RGB")

    width, height = rgb_image.size
    metadata: Dict[str, Any] = {
        "filename": filename,
        "width": width,
        "height": height,
        "dimensions_label": f"{width} × {height} px",
        "format": "JPEG" if detected_format in ("JPG", "JPEG") else detected_format,
        "source_mode": source_mode,
        "size_bytes": len(file_bytes),
        "size_label": format_file_size(len(file_bytes)),
    }
    return rgb_image, metadata


def preprocess_mri_image(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image into a normalized float32 grayscale array [0, 1].

    Placeholder normalization hook that can later be adapted to the trained
    model's exact spatial resolution and intensity standardization.
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr


def render_segmentation_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    fill_rgb: Tuple[int, int, int] = OVERLAY_FILL_RGB,
    contour_rgb: Tuple[int, int, int] = OVERLAY_CONTOUR_RGB,
    fill_opacity: float = 0.42,
) -> Image.Image:
    """Composite a translucent muted coral-rose segmentation region onto the MRI."""
    base_rgb = base_image.convert("RGB")
    base_arr = np.asarray(base_rgb, dtype=np.float32)
    h, w, _ = base_arr.shape

    soft_mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if soft_mask.shape != (h, w):
        mask_pil = Image.fromarray((soft_mask * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((w, h), Image.Resampling.BILINEAR)
        soft_mask = np.asarray(mask_pil, dtype=np.float32) / 255.0

    # Derive resolution-aware crisp perimeter ring via Pillow MinFilter
    binary_u8 = (soft_mask >= 0.45).astype(np.uint8) * 255
    binary_pil = Image.fromarray(binary_u8, mode="L")
    stroke_radius = max(1, min(3, round(min(h, w) / 260)))
    kernel_size = 2 * stroke_radius + 1
    eroded_pil = binary_pil.filter(ImageFilter.MinFilter(size=kernel_size))

    contour_raw = np.clip(
        (np.asarray(binary_pil, dtype=np.float32) - np.asarray(eroded_pil, dtype=np.float32))
        / 255.0,
        0.0,
        1.0,
    )
    contour_smooth = np.asarray(
        Image.fromarray((contour_raw * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=0.6)
        ),
        dtype=np.float32,
    ) / 255.0

    # Blend soft coral-rose interior tint over base scan
    alpha_fill = (soft_mask * fill_opacity)[..., None]
    fill_layer = np.full_like(base_arr, fill_rgb, dtype=np.float32)
    blended = base_arr * (1.0 - alpha_fill) + fill_layer * alpha_fill

    # Lift subtle warm highlights slightly inside dark tissue regions so overlay stays legible
    luminance = (
        0.299 * base_arr[:, :, 0] + 0.587 * base_arr[:, :, 1] + 0.114 * base_arr[:, :, 2]
    ) / 255.0
    warm_lift = (soft_mask * (1.0 - luminance) * 24.0)[..., None]
    blended = np.clip(blended + warm_lift * np.array([1.0, 0.72, 0.68]), 0.0, 255.0)

    # Composite crisp coral-terracotta perimeter ring
    alpha_contour = np.clip(contour_smooth * 0.88, 0.0, 0.92)[..., None]
    contour_layer = np.full_like(base_arr, contour_rgb, dtype=np.float32)
    final_arr = blended * (1.0 - alpha_contour) + contour_layer * alpha_contour

    return Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8), mode="RGB")


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string for clean inline HTML embedding."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def file_to_base64(path: Path) -> str:
    """Read a binary asset file and encode it as base64."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fit_panel_image(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize image preserving aspect ratio inside a fixed viewport box."""
    return ImageOps.contain(image.convert("RGB"), (box_w, box_h), Image.Resampling.LANCZOS)


def _load_export_font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a clean TrueType font if available on host OS, else fallback safely."""
    candidates = (
        ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            font_path = base / name
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size=size)
                except Exception:
                    continue
    return ImageFont.load_default(size=size)


def build_comparison_export_bytes(
    original_image: Image.Image,
    overlay_image: Image.Image,
    metadata: Dict[str, Any],
    area_px: int,
) -> bytes:
    """Build a polished side-by-side illustrative report PNG for download."""
    canvas_w, canvas_h = 1240, 740
    canvas = Image.new("RGB", (canvas_w, canvas_h), EXPORT_BG_RGB)
    draw = ImageDraw.Draw(canvas)

    font_title = _load_export_font(26, bold=True)
    font_sub = _load_export_font(14)
    font_label = _load_export_font(16, bold=True)
    font_small = _load_export_font(13)

    # Header strip
    draw.text((48, 32), "NeuroScan  |  MRI Visualisation Summary", fill=EXPORT_TEXT_RGB, font=font_title)
    draw.text(
        (48, 68),
        "DEMO MODE - SIMULATED RESULTS   |   Illustrative Region - Not a Model Prediction",
        fill=EXPORT_SAGE_RGB,
        font=font_sub,
    )

    # Side-by-side panels
    panel_w, panel_h = 548, 500
    left_x, right_x, panel_y = 48, 644, 112
    for px in (left_x, right_x):
        draw.rounded_rectangle(
            [px, panel_y, px + panel_w, panel_y + panel_h],
            radius=16,
            fill=EXPORT_CARD_RGB,
            outline=EXPORT_BORDER_RGB,
            width=2,
        )

    orig_fit = _fit_panel_image(original_image, panel_w - 40, panel_h - 76)
    over_fit = _fit_panel_image(overlay_image, panel_w - 40, panel_h - 76)

    canvas.paste(
        orig_fit,
        (left_x + (panel_w - orig_fit.width) // 2, panel_y + 20 + (panel_h - 76 - orig_fit.height) // 2),
    )
    canvas.paste(
        over_fit,
        (right_x + (panel_w - over_fit.width) // 2, panel_y + 20 + (panel_h - 76 - over_fit.height) // 2),
    )

    draw.text((left_x + 22, panel_y + panel_h - 44), "Original MRI Scan", fill=EXPORT_TEXT_RGB, font=font_label)
    draw.text(
        (right_x + 22, panel_y + panel_h - 44),
        "Illustrative Segmentation Overlay (Not a Model Prediction)",
        fill=OVERLAY_CONTOUR_RGB,
        font=font_label,
    )

    dims_ascii = f"{metadata.get('width', '-')} x {metadata.get('height', '-')} px"
    footer_text = (
        f"Source: {metadata.get('filename', 'MRI Scan')}   |   "
        f"Dimensions: {dims_ascii}   |   "
        f"Format: {metadata.get('format', '-')}   |   "
        f"Illustrative Area: {area_px:,} px   |   "
        "Research Prototype - For Visualisation Demonstration Only"
    )
    draw.text((48, 668), footer_text, fill=EXPORT_MUTED_RGB, font=font_small)

    out = BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()
