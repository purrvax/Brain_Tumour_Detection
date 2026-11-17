"""MRI upload and analysis trigger workflow component."""

from __future__ import annotations

import html
import time
from typing import Callable

import streamlit as st

from components import render_html
from utils.image_processing import (
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    image_to_base64,
    load_mri_image,
    validate_uploaded_file,
)
from utils.mock_analysis import ANALYSIS_STAGES, run_analysis

_UPLOAD_HEADER_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M12 15V4M12 4L7.5 8.5M12 4L16.5 8.5" stroke="#465343" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M4.5 15.5V17.5C4.5 18.9 5.6 20 7 20H17C18.4 20 19.5 18.9 19.5 17.5V15.5" stroke="#59664F" stroke-width="1.6" stroke-linecap="round"/>'
    '</svg>'
)


def render_upload_and_analysis(on_reset: Callable[[], None]) -> None:
    """Render the MRI upload section and inline analysis execution trigger."""
    render_html(
        """
        <div class="ns-divider"></div>
        <span id="detection" class="ns-anchor"></span>
        <section class="ns-section" aria-label="Analyse Your MRI Scan">
            <div class="ns-section-header">
                <span class="ns-eyebrow">Interactive Workflow</span>
                <h2 class="ns-section-title">Analyse Your MRI Scan</h2>
                <p class="ns-section-desc">
                    Upload a brain MRI image to explore the tumour-region visualisation workflow.
                </p>
            </div>
        </section>
        """
    )

    analyse_clicked = False

    with st.container(key="upload_panel"):
        render_html(
            f"""
            <div class="ns-panel-head">
                <div style="display:flex;align-items:center;gap:0.55rem;">
                    {_UPLOAD_HEADER_SVG}
                    <h3 class="ns-panel-title">Upload MRI Scan</h3>
                </div>
                <span class="ns-format-pill">PNG · JPG · JPEG · Max {MAX_FILE_SIZE_MB:.0f} MB</span>
            </div>
            """
        )

        uploader_widget_key = f"mri_uploader_{st.session_state.uploader_key}"
        uploaded_file = st.file_uploader(
            "Drop your MRI image here",
            type=list(SUPPORTED_EXTENSIONS),
            accept_multiple_files=False,
            key=uploader_widget_key,
            label_visibility="collapsed",
        )

        if st.session_state.uploaded_image is None:
            render_html(
                """
                <div class="ns-upload-hints">
                    <span>Valid PNG, JPG, or JPEG MRI slices only</span>
                </div>
                """
            )

        # Synchronize uploaded file with session state without losing preview on reruns
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_sig = f"{uploaded_file.name}:{len(file_bytes)}"

            if st.session_state.uploaded_file_sig != file_sig:
                is_valid, err_msg = validate_uploaded_file(file_bytes, uploaded_file.name)
                if is_valid:
                    loaded_img, meta = load_mri_image(file_bytes, uploaded_file.name)
                    st.session_state.uploaded_file_sig = file_sig
                    st.session_state.uploaded_image = loaded_img
                    st.session_state.uploaded_preview_b64 = image_to_base64(loaded_img)
                    st.session_state.image_metadata = meta
                    st.session_state.upload_error = None
                    st.session_state.analysis_result = None
                else:
                    st.session_state.uploaded_file_sig = file_sig
                    st.session_state.uploaded_image = None
                    st.session_state.uploaded_preview_b64 = None
                    st.session_state.image_metadata = None
                    st.session_state.upload_error = err_msg
                    st.session_state.analysis_result = None
        else:
            # Only clear if user cleared via native 'x' button on the uploader
            if st.session_state.uploaded_file_sig is not None:
                st.session_state.uploaded_file_sig = None
                st.session_state.uploaded_image = None
                st.session_state.uploaded_preview_b64 = None
                st.session_state.image_metadata = None
                st.session_state.upload_error = None
                st.session_state.analysis_result = None

        if st.session_state.upload_error:
            safe_err = html.escape(st.session_state.upload_error)
            render_html(f'<div class="ns-alert-error">{safe_err}</div>')

        if st.session_state.uploaded_image is not None and st.session_state.image_metadata is not None:
            preview_b64 = st.session_state.uploaded_preview_b64

            render_html(
                f"""
                <div class="ns-preview-frame">
                    <img src="data:image/png;base64,{preview_b64}" alt="Uploaded MRI scan preview" />
                </div>
                """
            )

            btn_col_remove, btn_col_analyse = st.columns([0.42, 0.58], gap="small")
            with btn_col_remove:
                if st.button("Remove Image", key="btn_remove_image", type="secondary", width="stretch"):
                    on_reset()
                    st.rerun()
            with btn_col_analyse:
                analyse_clicked = st.button(
                    "Analyse MRI Scan",
                    key="btn_analyse_mri",
                    type="primary",
                    width="stretch",
                )

            if analyse_clicked:
                status_slot = st.empty()
                progress_bar = st.progress(0)

                for idx, (fraction, stage_title, stage_desc) in enumerate(ANALYSIS_STAGES, start=1):
                    status_slot.markdown(
                        f"**Step {idx} of {len(ANALYSIS_STAGES)}:** {stage_title} — _{stage_desc}_"
                    )
                    progress_bar.progress(int(fraction * 100))
                    time.sleep(0.18)

                result = run_analysis(
                    st.session_state.uploaded_image,
                    metadata=st.session_state.image_metadata,
                )
                status_slot.markdown("**Processing complete:** Visualising segmentation and quantification...")
                progress_bar.progress(100)
                st.session_state.analysis_result = result
                st.rerun()

