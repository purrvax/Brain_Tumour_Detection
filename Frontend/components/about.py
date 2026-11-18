"""Case study research, methodology workflow, technical highlights, and PDF section."""

from __future__ import annotations

import html
from typing import Dict, List, Tuple

from components import render_html
from components.case_study_pdf import render_case_study_pdf

# ---------------------------------------------------------------------------
# Minimal Editorial SVG Icons (Strictly Warm Ivory / Sage / Lavender / Peach)
# ---------------------------------------------------------------------------
_PROBLEM_ICON_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<circle cx="12" cy="12" r="8.2" stroke="#465343" stroke-width="1.5"/>'
    '<circle cx="12" cy="12" r="4.6" stroke="#87977B" stroke-width="1.4" stroke-dasharray="2.4 2.2"/>'
    '<path d="M12 2.4V5M12 19V21.6M2.4 12H5M19 12H21.6" stroke="#59664F" stroke-width="1.4" stroke-linecap="round"/>'
    '</svg>'
)

_MOTIVATION_ICON_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="3.5" y="4.5" width="17" height="15" rx="3.5" stroke="#59664F" stroke-width="1.5"/>'
    '<path d="M10.4 9.2C12 8.2 14.4 8.6 15.5 10.2C16.4 11.5 16.1 13.7 14.8 14.8'
    'C13.4 15.9 10.9 15.8 9.8 14.4C8.7 13 8.9 10.2 10.4 9.2Z" '
    'fill="rgba(199, 187, 211, 0.45)" stroke="#756389" stroke-width="1.5"/>'
    '</svg>'
)

_APPROACH_ICON_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M4.5 8.5H9.5M4.5 12H13.5M4.5 15.5H10.5" stroke="#59664F" stroke-width="1.5" stroke-linecap="round"/>'
    '<circle cx="16.5" cy="9.5" r="2.2" stroke="#A8564E" stroke-width="1.5"/>'
    '<circle cx="17.5" cy="15.5" r="1.8" stroke="#87977B" stroke-width="1.5"/>'
    '<path d="M16.5 11.7L17.1 13.7" stroke="#77786F" stroke-width="1.3" stroke-linecap="round"/>'
    '</svg>'
)

_STEP_ICONS: Tuple[str, ...] = (
    # 1. Input MRI Image
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#59664F" stroke-width="1.4"/>'
    '<ellipse cx="14" cy="14" rx="5.8" ry="7.2" fill="rgba(135,151,123,0.18)" stroke="#465343" stroke-width="1.3"/>'
    '<path d="M14 7.2V20.8" stroke="#87977B" stroke-width="1.1" stroke-dasharray="2 2"/>'
    '<circle cx="16.3" cy="12" r="1.7" fill="rgba(198,124,114,0.45)" stroke="#A8564E" stroke-width="1.1"/>'
    '</svg>',
    # 2. Gaussian Blur - Noise reduction
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#756389" stroke-width="1.4"/>'
    '<path d="M6.5 19C9 19 10.5 9.5 14 9.5C17.5 9.5 19 19 21.5 19" stroke="#756389" stroke-width="1.6" stroke-linecap="round"/>'
    '<circle cx="14" cy="14.5" r="2.2" fill="rgba(199,187,211,0.5)"/>'
    '</svg>',
    # 3. CLAHE - Contrast enhancement
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#59664F" stroke-width="1.4"/>'
    '<path d="M7 20C10.5 19.5 12 14.5 14.2 12C16.4 9.5 18 8.2 21 8" stroke="#465343" stroke-width="1.5" stroke-linecap="round"/>'
    '<path d="M8 20V16.5M12 20V14M16 20V11.5M20 20V14.5" stroke="#87977B" stroke-width="1.4" stroke-linecap="round"/>'
    '</svg>',
    # 4. Otsu Thresholding
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#59664F" stroke-width="1.4"/>'
    '<path d="M6.5 19.5C8 19.5 8.8 13.5 10.3 13.5C11.8 13.5 12.2 18.5 13.8 18.5C15.5 18.5 16.4 10.5 18.2 10.5C20 10.5 20.4 19.5 21.5 19.5" stroke="#465343" stroke-width="1.35" stroke-linecap="round"/>'
    '<path d="M14.2 7.5V20.5" stroke="#A8564E" stroke-width="1.4" stroke-dasharray="2.2 2"/>'
    '</svg>',
    # 5. Morphological Operations
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#A8564E" stroke-width="1.35"/>'
    '<rect x="8.5" y="8.5" width="4.5" height="4.5" rx="1" fill="rgba(135,151,123,0.45)" stroke="#59664F" stroke-width="1.1"/>'
    '<rect x="15" y="8.5" width="4.5" height="4.5" rx="1" fill="rgba(233,198,175,0.55)" stroke="#A8564E" stroke-width="1.1"/>'
    '<rect x="8.5" y="15" width="4.5" height="4.5" rx="1" fill="rgba(233,198,175,0.55)" stroke="#A8564E" stroke-width="1.1"/>'
    '<rect x="15" y="15" width="4.5" height="4.5" rx="1" fill="rgba(135,151,123,0.45)" stroke="#59664F" stroke-width="1.1"/>'
    '</svg>',
    # 6. Distance Transform
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#59664F" stroke-width="1.4"/>'
    '<circle cx="14" cy="14" r="7" stroke="#87977B" stroke-width="1.1" stroke-dasharray="2 2"/>'
    '<circle cx="14" cy="14" r="4.4" fill="rgba(199,187,211,0.38)" stroke="#756389" stroke-width="1.2"/>'
    '<circle cx="14" cy="14" r="1.8" fill="#465343"/>'
    '</svg>',
    # 7. Marker-Based Watershed
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#756389" stroke-width="1.4"/>'
    '<path d="M6.5 18.5C8.5 14 10.5 13 12.2 15.8C13.8 18.2 15.2 11 17.5 11C19.5 11 20.5 15.5 21.5 18.5" stroke="#59664F" stroke-width="1.45" stroke-linecap="round"/>'
    '<path d="M13.8 8V20" stroke="#A8564E" stroke-width="1.35" stroke-dasharray="2 2"/>'
    '<circle cx="17.5" cy="14.5" r="1.7" fill="#A8564E"/>'
    '</svg>',
    # 8. Contour Extraction
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#A8564E" stroke-width="1.35"/>'
    '<path d="M11 9.5C13.5 8 17.5 8.8 18.8 11.4C20 13.8 19 17.8 16.4 18.8C13.6 19.9 9.8 18.6 9 15.8C8.3 13.2 9.2 10.6 11 9.5Z" fill="rgba(233,198,175,0.52)" stroke="#A8564E" stroke-width="1.5"/>'
    '</svg>',
    # 9. Area Estimation
    '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="4" y="4" width="20" height="20" rx="5" fill="#F2F0E9" stroke="#465343" stroke-width="1.4"/>'
    '<path d="M8.5 9.5H19.5M8.5 14H19.5M8.5 18.5H19.5M9.5 8.5V19.5M14 8.5V19.5M18.5 8.5V19.5" stroke="rgba(135,151,123,0.45)" stroke-width="1"/>'
    '<rect x="10.2" y="10.2" width="7.6" height="7.6" rx="1.8" fill="rgba(198,124,114,0.42)" stroke="#A8564E" stroke-width="1.35"/>'
    '</svg>',
)

_HIGHLIGHT_ICONS: Dict[str, str] = {
    "enhancement": _STEP_ICONS[2],
    "watershed": _STEP_ICONS[6],
    "morphology": _STEP_ICONS[4],
    "area": _STEP_ICONS[8],
}

# ---------------------------------------------------------------------------
# Case Study Structured Content (Extracted from IP-Case Study.pdf)
# ---------------------------------------------------------------------------
_DATASET_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    ("3,064", "Cleaned MRI Slices", "512×512 NumPy grayscale slices with paired masks & class labels (1–3)", "sage"),
    ("6,892", "Categorised MRI Images", "158.77 MB dataset across Glioma, Meningioma, Pituitary & No-Tumor", "lavender"),
    ("CPU · Secs", "Transparent Execution", "Deterministic classical math with clinician-tunable parameters", "peach"),
    ("523.6 mm²", "Area Estimation Output", "Pixel summation A_pixel = ΣΣ M(x, y) & physical mm² calibration", "cream"),
)

_OVERVIEW_CARDS: Tuple[Dict[str, str], ...] = (
    {
        "tag": "01 · Problem Statement",
        "title": "Problem Statement",
        "tone": "sage",
        "icon_tone": "sage",
        "icon": _PROBLEM_ICON_SVG,
        "body": (
            "Hospital grayscale brain MRI scans frequently exhibit acquisition noise, uneven intensity "
            "distributions, and irregular tumour boundaries across active tumour, oedema, and necrosis "
            "components. Manual delineation is slow and subjective, making automated image segmentation "
            "and cranial region-of-interest (ROI) extraction vital for isolating suspected abnormalities "
            "from surrounding healthy tissue and skull structures."
        ),
    },
    {
        "tag": "02 · Project Motivation",
        "title": "Project Motivation",
        "tone": "lavender",
        "icon_tone": "lavender",
        "icon": _MOTIVATION_ICON_SVG,
        "body": (
            "Analysing MRI scans reliably is critical because brain tumours appear as subtle variations "
            "in local intensity or texture. While deep networks such as U-Net offer strong accuracy, they "
            "require large labelled cohorts, GPUs, and opaque feature representations. Exploring classical "
            "image processing delivers complete mathematical interpretability, parameter tunability, and "
            "fast CPU execution within seconds."
        ),
    },
    {
        "tag": "03 · Proposed Approach",
        "title": "Proposed Approach",
        "tone": "peach",
        "icon_tone": "peach",
        "icon": _APPROACH_ICON_SVG,
        "body": (
            "This project implements a structured classical computer vision pipeline in Python and OpenCV "
            "to enhance MRI contrast, suppress high-frequency noise while preserving tumour margins, "
            "extract the inner brain ROI, refine binary masks through morphological operations, separate "
            "connected structures via marker-based watershed, and compute the suspected tumour's pixel area."
        ),
    },
)

_PIPELINE_STEPS: Tuple[Dict[str, str], ...] = (
    {
        "num": "01",
        "phase": "Acquisition",
        "title": "Input MRI Image",
        "desc": "Load grayscale cranial MRI slice (single-channel 512×512 matrix) and inspect intensity histogram.",
        "tone": "sage",
    },
    {
        "num": "02",
        "phase": "Denoising",
        "title": "Gaussian Blur",
        "desc": "Noise reduction — smooths high-frequency scan grain prior to segmentation alongside edge-preserving median filtering.",
        "tone": "lavender",
    },
    {
        "num": "03",
        "phase": "Enhancement",
        "title": "CLAHE",
        "desc": "Contrast enhancement — boosts local contrast to highlight bright suspected tumour tissue against parenchyma.",
        "tone": "sage",
    },
    {
        "num": "04",
        "phase": "Binarisation",
        "title": "Otsu Thresholding",
        "desc": "Initial foreground/background separation — automatically derives optimal threshold inside the extracted brain ROI.",
        "tone": "sage",
    },
    {
        "num": "05",
        "phase": "Refinement",
        "title": "Morphological Operations",
        "desc": "Refinement of the binary image — opening removes isolated specks while closing seals internal gaps and holes.",
        "tone": "peach",
    },
    {
        "num": "06",
        "phase": "Region Discovery",
        "title": "Distance Transform",
        "desc": "Identification of foreground regions — computes pixel distance to background to define sure foreground vs unknown ring.",
        "tone": "lavender",
    },
    {
        "num": "07",
        "phase": "Segmentation",
        "title": "Marker-Based Watershed",
        "desc": "Region segmentation — floods the topographic intensity surface from connected-component markers without over-segmentation.",
        "tone": "lavender",
    },
    {
        "num": "08",
        "phase": "Delineation",
        "title": "Contour Extraction",
        "desc": "Extraction of the segmented region — selects the largest candidate region inside the brain ROI and traces its outer boundary.",
        "tone": "peach",
    },
    {
        "num": "09",
        "phase": "Quantification",
        "title": "Area Estimation",
        "desc": "Calculation of the suspected region's pixel area (A_pixel = ΣΣ M(x, y) px²), perimeter, and mm² when spacing is known.",
        "tone": "cream",
    },
)

_TECHNICAL_HIGHLIGHTS: Tuple[Dict[str, str], ...] = (
    {
        "tag": "01 · Preprocessing",
        "title": "Image Enhancement",
        "tone": "sage",
        "icon": _HIGHLIGHT_ICONS["enhancement"],
        "body": (
            "Combines single-channel grayscale normalization, histogram analysis, CLAHE contrast enhancement, "
            "Gaussian smoothing, median filtering, and morphological gradient sharpening to suppress isolated "
            "noise while preserving sharp tumour boundaries."
        ),
    },
    {
        "tag": "02 · Topographic Splitting",
        "title": "Watershed Segmentation",
        "tone": "lavender",
        "icon": _HIGHLIGHT_ICONS["watershed"],
        "body": (
            "Treats pixel intensity as a topographic relief map. Pairing Euclidean distance transforms with "
            "sure-foreground cores, dilated sure-background boundaries, and connected-component markers "
            "prevents the over-segmentation typical of unguided watershed."
        ),
    },
    {
        "tag": "03 · Binary Mask Cleanup",
        "title": "Morphological Processing",
        "tone": "peach",
        "icon": _HIGHLIGHT_ICONS["morphology"],
        "body": (
            "Applies structuring-element opening to discard spurious artefacts, closing to bridge small mask "
            "discontinuities, and head-contour distance masking to strip scalp and skull borders before "
            "isolating the inner brain ROI."
        ),
    },
    {
        "tag": "04 · Spatial Measurement",
        "title": "Tumour Area Estimation",
        "tone": "cream",
        "icon": _HIGHLIGHT_ICONS["area"],
        "body": (
            "Retains the largest connected component inside the brain ROI, overlays its outer contour onto "
            "the source MRI, and sums foreground mask pixels (A_pixel = Σ<sub>x=1..W</sub> Σ<sub>y=1..H</sub> "
            "M(x, y) px²) with optional physical conversion (e.g., 523.6 mm²)."
        ),
    },
)

def _build_dataset_strip_html() -> str:
    """Build the quantitative metrics strip extracted from the case study report."""
    cards: List[str] = []
    for value, label, sub, tone in _DATASET_METRICS:
        cards.append(
            f'<div class="ns-cs-stat ns-metric-{tone}">'
            f'<span class="ns-cs-stat-val">{html.escape(value)}</span>'
            f'<span class="ns-cs-stat-lbl">{html.escape(label)}</span>'
            f'<span class="ns-cs-stat-sub">{html.escape(sub)}</span>'
            f'</div>'
        )
    return '<div class="ns-cs-stats-grid">' + "".join(cards) + "</div>"


def _build_overview_cards_html() -> str:
    """Build the three editorial narrative cards for Problem, Motivation, and Approach."""
    cards: List[str] = []
    for item in _OVERVIEW_CARDS:
        cards.append(
            f'<article class="ns-info-card ns-card-{item["tone"]}">'
            f'<div class="ns-card-icon ns-icon-{item["icon_tone"]}">{item["icon"]}</div>'
            f'<span class="ns-card-tag">{html.escape(item["tag"])}</span>'
            f'<h3 class="ns-card-title">{html.escape(item["title"])}</h3>'
            f'<p class="ns-card-text">{html.escape(item["body"])}</p>'
            f'</article>'
        )
    return '<div class="ns-about-grid">' + "".join(cards) + "</div>"


def _build_methodology_pipeline_html() -> str:
    """Build the 9-step connected classical image-processing workflow."""
    step_cards: List[str] = []
    for idx, step in enumerate(_PIPELINE_STEPS):
        icon_svg = _STEP_ICONS[idx]
        step_cards.append(
            f'<article class="ns-step-card ns-step-{step["tone"]}">'
            f'<div class="ns-step-top">'
            f'<span class="ns-step-num">{html.escape(step["num"])}</span>'
            f'<span class="ns-step-phase">{html.escape(step["phase"])}</span>'
            f'<span class="ns-step-icon">{icon_svg}</span>'
            f'</div>'
            f'<h4 class="ns-step-title">{html.escape(step["title"])}</h4>'
            f'<p class="ns-step-desc">{html.escape(step["desc"])}</p>'
            f'</article>'
        )
    return (
        '<div class="ns-cs-subsection">'
        '<div class="ns-cs-subhead">'
        '<span class="ns-eyebrow">Methodology Workflow</span>'
        '<h3 class="ns-cs-subtitle">Pipeline</h3>'
        '</div>'
        f'<div class="ns-pipeline-grid">{"".join(step_cards)}</div>'
        '</div>'
        '</div>'
    )


def _build_technical_highlights_html() -> str:
    """Build the four minimal technical highlight cards plus key report findings."""
    cards: List[str] = []
    for item in _TECHNICAL_HIGHLIGHTS:
        cards.append(
            f'<article class="ns-highlight-card ns-card-{item["tone"]}">'
            f'<div class="ns-highlight-head">'
            f'<div class="ns-card-icon ns-icon-{item["tone"] if item["tone"] != "cream" else "sage"}">{item["icon"]}</div>'
            f'<div>'
            f'<span class="ns-card-tag">{html.escape(item["tag"])}</span>'
            f'<h4 class="ns-card-title">{html.escape(item["title"])}</h4>'
            f'</div>'
            f'</div>'
            f'<p class="ns-card-text">{item["body"]}</p>'
            f'</article>'
        )

    findings_html = (
        '<div class="ns-findings-grid">'
        '<div class="ns-finding-box">'
        '<span class="ns-card-tag">Report Findings &amp; Limitations</span>'
        '<h4 class="ns-finding-title">Filter Sensitivity &amp; Tumour Heterogeneity</h4>'
        '<p class="ns-card-text">'
        'Comparative studies in the report (Pooja et al.) confirm that pre-processing filter choice strongly '
        'governs final mask fidelity: median filtering suppresses isolated noise while preserving tumour margins, '
        'whereas excessive Gaussian blurring softens boundaries. Because gliomas, meningiomas, and pituitary '
        'tumours vary widely in size, shape, and intensity, classical thresholds serve as a computer-aided '
        'analysis baseline rather than a clinical diagnostic tool.'
        '</p>'
        '</div>'
        '<div class="ns-finding-box">'
        '<span class="ns-card-tag">Future Scope &amp; Benchmarking</span>'
        '<h4 class="ns-finding-title">Quantitative Validation &amp; Deep Learning Extension</h4>'
        '<p class="ns-card-text">'
        'Future iterations documented in the study extend this classical baseline with CNN and U-Net models to '
        'learn multi-scale tumour representations automatically, evaluated against expert-annotated masks using '
        '<strong>Dice Coefficient</strong>, <strong>Intersection over Union (IoU)</strong>, '
        '<strong>Precision</strong>, and <strong>Recall</strong>.'
        '</p>'
        '</div>'
        '</div>'
    )

    return (
        '<div class="ns-cs-subsection">'
        '<div class="ns-cs-subhead">'
        '<span class="ns-eyebrow">Core Techniques</span>'
        '<h3 class="ns-cs-subtitle">Technical Highlights</h3>'
        '<p class="ns-cs-subdesc">'
        'Four foundational image-processing pillars powering contrast conditioning, mask refinement, '
        'topographic boundary separation, and region measurement.'
        '</p>'
        '</div>'
        f'<div class="ns-highlights-grid">{"".join(cards)}</div>'
        f'{findings_html}'
        '</div>'
    )


def render_about() -> None:
    """Render the upgraded About / Case Study section and embedded case study PDF."""
    overview_cards = _build_overview_cards_html()
    methodology_html = _build_methodology_pipeline_html()

    render_html(
        f"""
        <div class="ns-divider"></div>
        <span id="about" class="ns-anchor"></span>
        <section class="ns-section ns-case-study-section" aria-label="The Research Behind the Project">
            <div class="ns-section-header ns-cs-section-header">
                <div class="ns-cs-label-row">
                    <span class="ns-cs-pill-badge">PROJECT CASE STUDY</span>
                    <span class="ns-eyebrow ns-cs-eyebrow">THE RESEARCH BEHIND THE PROJECT</span>
                </div>
                <h2 class="ns-section-title">Understanding Brain Tumour Segmentation</h2>
            </div>
            {overview_cards}
            {methodology_html}
        </section>
        """
    )

    render_case_study_pdf()
