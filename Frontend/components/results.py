"""Results dashboard component displaying original MRI, watershed segmentation overlay, and metrics."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable

import streamlit as st

from components import render_html
from utils.image_processing import image_to_base64


def render_results(on_reset: Callable[[], None]) -> None:
    """Render the full results dashboard when analysis_result is populated."""
    result = st.session_state.get("analysis_result")
    if not result:
        return

    meta = st.session_state.get("image_metadata") or {}
    orig_b64 = image_to_base64(result["original_image"])
    overlay_b64 = image_to_base64(result["overlay_image"])
    filename_stem = Path(str(meta.get("filename", "mri_scan"))).stem

    tumor_detected = bool(result.get("tumor_detected", False))
    area_px = int(result.get("region_area_px", 0))
    perimeter_px = float(result.get("perimeter_px", 0.0))
    status_msg = str(result.get("status_message", "Analysis Complete"))

    status_badge_class = "ns-format-pill"
    if tumor_detected:
        status_pill = (
            f'<span class="{status_badge_class}" style="background:rgba(204,125,114,0.18);'
            f'color:#A8564E;border-color:rgba(168,86,78,0.42);font-weight:600;">'
            f'Suspected Tumour Detected · Watershed Localised</span>'
        )
    else:
        status_pill = (
            f'<span class="{status_badge_class}" style="background:rgba(135,151,123,0.18);'
            f'color:#59664F;border-color:rgba(89,102,79,0.36);font-weight:600;">'
            f'Scan Clear · No Tumour Region Detected</span>'
        )

    render_html(
        f"""
        <div class="ns-divider"></div>
        <span id="results" class="ns-anchor"></span>
        <section class="ns-results-wrap" aria-label="Analysis Results">
            <div class="ns-results-topbar" style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:0.75rem;">
                <div>
                    <span class="ns-eyebrow">Classical CV & Watershed Output</span>
                    <h2 class="ns-section-title" style="margin-bottom:0.25rem;">Diagnostic Results</h2>
                </div>
                <div>{status_pill}</div>
            </div>
        </section>
        """
    )

    # Side-by-Side Comparison (Original MRI & Segmentation Overlay)
    col_orig, col_overlay = st.columns([1.0, 1.0], gap="large")

    with col_orig:
        with st.container(key="mri_orig_card"):
            render_html(
                f"""
                <div class="ns-mri-card-head">
                    <h3 class="ns-mri-card-title">Input MRI</h3>
                </div>
                <div class="ns-mri-viewport">
                    <img src="data:image/png;base64,{orig_b64}" alt="Original uploaded MRI scan" />
                </div>
                """
            )

    with col_overlay:
        with st.container(key="mri_overlay_card"):
            overlay_title = "Tumour Segmentation Overlay" if tumor_detected else "Intracranial Tissue (Clear Scan)"
            render_html(
                f"""
                <div class="ns-mri-card-head">
                    <h3 class="ns-mri-card-title">{overlay_title}</h3>
                </div>
                <div class="ns-mri-viewport">
                    <img src="data:image/png;base64,{overlay_b64}" alt="Tumour segmentation overlay on MRI scan" />
                </div>
                """
            )

    # Full-Width Horizontal Estimated Area Readout
    perimeter_display = f"Perimeter: {perimeter_px:.1f} px" if tumor_detected else "Normal Parenchyma"
    render_html(
        f"""
        <div class="ns-area-bar">
            <div class="ns-area-bar-left">
                <span class="ns-area-bar-label">Estimated Region Area</span>
                <span class="ns-area-bar-sep" aria-hidden="true">&middot;</span>
                <span class="ns-area-bar-sub">{perimeter_display} &middot; {html.escape(status_msg)}</span>
            </div>
            <div class="ns-area-bar-right">
                <span class="ns-area-bar-value">{area_px:,} px</span>
            </div>
        </div>
        """
    )

    # Detailed 9-Stage CV Pipeline Breakdown (from Research Notebook)
    stages = result.get("stages_pil", {})
    if stages:
        with st.expander("Explore 9-Stage Computer Vision Pipeline Breakdown", expanded=False):
            st.markdown(
                "Inspect the intermediate representations generated at each step of the classical "
                "computer vision and marker-based watershed pipeline:"
            )

            tab1, tab2, tab3 = st.tabs([
                "1 · Denoising & Enhancement",
                "2 · Cranial ROI & Thresholding",
                "3 · Watershed & Delineation",
            ])

            with tab1:
                col_t1_a, col_t1_b, col_t1_c = st.columns(3)
                with col_t1_a:
                    if "gaussian_blur" in stages:
                        st.image(stages["gaussian_blur"], caption="Gaussian Filtering (5×5, σ=2)", use_container_width=True)
                with col_t1_b:
                    if "first_enhancement" in stages:
                        st.image(stages["first_enhancement"], caption="1st Enhancement (Intensity Shift)", use_container_width=True)
                with col_t1_c:
                    if "median_filtering" in stages:
                        st.image(stages["median_filtering"], caption="Median Filtering (5×5 Denoise)", use_container_width=True)

                col_t1_d, col_t1_e = st.columns(2)
                with col_t1_d:
                    if "morphological_gradient" in stages:
                        st.image(stages["morphological_gradient"], caption="Morphological Gradient (Cross 3×3)", use_container_width=True)
                with col_t1_e:
                    if "second_enhancement" in stages:
                        st.image(stages["second_enhancement"], caption="2nd Enhancement (Combined Relief)", use_container_width=True)

            with tab2:
                col_t2_a, col_t2_b, col_t2_c = st.columns(3)
                with col_t2_a:
                    if "brain_roi" in stages:
                        st.image(stages["brain_roi"], caption="Brain ROI (Skull Excluded via Distance Map)", use_container_width=True)
                with col_t2_b:
                    if "threshold_roi" in stages:
                        st.image(stages["threshold_roi"], caption="Thresholded ROI (Otsu + 85th Percentile)", use_container_width=True)
                with col_t2_c:
                    if "cleaned_mask" in stages:
                        st.image(stages["cleaned_mask"], caption="Cleaned Mask (Morphological Open & Close)", use_container_width=True)

            with tab3:
                col_t3_a, col_t3_b, col_t3_c = st.columns(3)
                with col_t3_a:
                    if "distance_transform" in stages:
                        st.image(stages["distance_transform"], caption="Euclidean Distance Transform", use_container_width=True)
                with col_t3_b:
                    if "markers" in stages:
                        st.image(stages["markers"], caption="Watershed Markers (Sure Foreground)", use_container_width=True)
                with col_t3_c:
                    if "watershed_boundaries" in stages:
                        st.image(stages["watershed_boundaries"], caption="Watershed Boundaries (Topographic Splitting)", use_container_width=True)

                if "candidate_boundary" in stages and tumor_detected:
                    st.image(stages["candidate_boundary"], caption="Candidate Tumour Boundary Traced on MRI", use_container_width=True)

    # Action Buttons (Download Diagnostic Summary + Analyse Another Scan)
    action_col_dl, action_col_reset, _ = st.columns([0.34, 0.30, 0.36], gap="medium")
    with action_col_dl:
        st.download_button(
            label="Download Diagnostic Summary",
            data=result["export_bytes"],
            file_name=f"{filename_stem}_neuroscan_analysis.png",
            mime="image/png",
            type="primary",
            width="stretch",
            key="btn_download_analysis",
        )
    with action_col_reset:
        if st.button(
            "Analyse Another Scan",
            type="secondary",
            width="stretch",
            key="btn_analyse_another",
        ):
            on_reset()
            st.rerun()
