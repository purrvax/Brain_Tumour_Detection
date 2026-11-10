"""Results dashboard component displaying original MRI, illustrative overlay, and metrics."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable

import streamlit as st

from components import render_html
from utils.image_processing import image_to_base64


def render_results(on_reset: Callable[[], None]) -> None:
    """Render the simulated results dashboard when analysis_result is populated."""
    result = st.session_state.get("analysis_result")
    if not result:
        return

    meta = st.session_state.get("image_metadata") or {}
    orig_b64 = image_to_base64(result["original_image"])
    overlay_b64 = image_to_base64(result["overlay_image"])
    filename_stem = Path(str(meta.get("filename", "mri_scan"))).stem

    render_html(
        """
        <div class="ns-divider"></div>
        <span id="results" class="ns-anchor"></span>
        <section class="ns-results-wrap" aria-label="Simulation Results">
            <div class="ns-results-topbar">
                <div>
                    <span class="ns-eyebrow">Visualisation Output</span>
                    <h2 class="ns-section-title" style="margin-bottom:0.25rem;">Results</h2>
                </div>
            </div>
        </section>
        """
    )

    # Side-by-Side Comparison (Original MRI & Illustrative Segmentation Overlay)
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
            render_html(
                f"""
                <div class="ns-mri-card-head">
                    <h3 class="ns-mri-card-title">Tumour Segmentation</h3>
                </div>
                <div class="ns-mri-viewport">
                    <img src="data:image/png;base64,{overlay_b64}" alt="Tumour segmentation overlay on MRI scan" />
                </div>
                """
            )

    # Full-Width Horizontal Estimated Area Readout
    render_html(
        f"""
        <div class="ns-area-bar">
            <div class="ns-area-bar-left">
                <span class="ns-area-bar-label">Estimated Region Area (pixels)</span>
                <span class="ns-area-bar-sep" aria-hidden="true">&middot;</span>
            </div>
            <div class="ns-area-bar-right">
                <span class="ns-area-bar-value">{result['region_area_px']:,} px</span>
            </div>
        </div>
        """
    )

    # Action Buttons (Download Illustrative Result + Analyse Another Scan)
    action_col_dl, action_col_reset, _ = st.columns([0.34, 0.30, 0.36], gap="medium")
    with action_col_dl:
        st.download_button(
            label="Download Illustrative Result",
            data=result["export_bytes"],
            file_name=f"{filename_stem}_neuroscan_demo.png",
            mime="image/png",
            type="primary",
            width="stretch",
            key="btn_download_demo",
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
