"""MRI upload and sequential simulated analysis workflow component."""

from __future__ import annotations

import html
import time
from typing import Callable, Optional

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

_READY_ILLUSTRATION_SVG = (
    '<svg width="104" height="104" viewBox="0 0 110 110" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<circle cx="55" cy="55" r="44" stroke="#C8CEBF" stroke-width="1.2" stroke-dasharray="3 4"/>'
    '<circle cx="55" cy="55" r="32" stroke="#C7BBD3" stroke-width="1.3"/>'
    '<path d="M55 24C41 24 31 35 31 50C31 64 38 77 50 81C52.4 81.8 54 79.8 54 77.4V27'
    'C54 25.2 54.4 24 55 24Z" fill="rgba(135, 151, 123, 0.14)" stroke="#59664F" stroke-width="1.4"/>'
    '<path d="M55 24C69 24 79 35 79 50C79 64 72 77 60 81C57.6 81.8 56 79.8 56 77.4V27'
    'C56 25.2 55.6 24 55 24Z" fill="rgba(199, 187, 211, 0.18)" stroke="#756389" stroke-width="1.4"/>'
    '<circle cx="66" cy="45" r="6.5" fill="rgba(233, 198, 175, 0.55)" stroke="#A8564E" stroke-width="1.3" stroke-dasharray="2 2"/>'
    '<path d="M18 55H27M83 55H92M55 12V19M55 91V98" stroke="#87977B" stroke-width="1.3" stroke-linecap="round"/>'
    '</svg>'
)

_UPLOAD_HEADER_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M12 15V4M12 4L7.5 8.5M12 4L16.5 8.5" stroke="#465343" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M4.5 15.5V17.5C4.5 18.9 5.6 20 7 20H17C18.4 20 19.5 18.9 19.5 17.5V15.5" stroke="#59664F" stroke-width="1.6" stroke-linecap="round"/>'
    '</svg>'
)


def _render_stage_list(active_index: int = -1, all_done: bool = False) -> str:
    """Build HTML markup for the four sequential analysis stages."""
    items = []
    for idx, (_, title, subtitle) in enumerate(ANALYSIS_STAGES, start=1):
        if all_done or (active_index > 0 and idx < active_index):
            state_cls = "is-done"
            marker = "✓"
        elif idx == active_index:
            state_cls = "is-active"
            marker = str(idx)
        else:
            state_cls = ""
            marker = str(idx)

        items.append(
            f'<div class="ns-stage-item {state_cls}">'
            f'<span class="ns-stage-num">{marker}</span>'
            f'<div>'
            f'<span class="ns-stage-label">{html.escape(title)}</span>'
            f'<span class="ns-stage-sub">{html.escape(subtitle)}</span>'
            f'</div>'
            f'</div>'
        )
    return '<div class="ns-stage-list">' + "".join(items) + "</div>"


def render_upload_and_analysis(on_reset: Callable[[], None]) -> None:
    """Render the two-column MRI upload and simulated analysis section."""
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

    col_upload, col_analysis = st.columns([1.0, 1.0], gap="large")
    analyse_clicked = False

    # ------------------------------------------------------------------
    # LEFT COLUMN — MRI IMAGE UPLOAD
    # ------------------------------------------------------------------
    with col_upload:
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
                        <span>Processed locally in session</span>
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
                meta = st.session_state.image_metadata
                preview_b64 = st.session_state.uploaded_preview_b64
                safe_name = html.escape(str(meta["filename"]))
                safe_dims = html.escape(str(meta["dimensions_label"]))
                safe_size = html.escape(str(meta["size_label"]))

                render_html(
                    f"""
                    <div class="ns-preview-frame">
                        <img src="data:image/png;base64,{preview_b64}" alt="Uploaded MRI preview: {safe_name}" />
                    </div>
                    <div class="ns-file-meta-grid">
                        <div class="ns-file-meta-pill">
                            <span class="ns-file-meta-k">Filename</span>
                            <span class="ns-file-meta-v" title="{safe_name}">{safe_name}</span>
                        </div>
                        <div class="ns-file-meta-pill">
                            <span class="ns-file-meta-k">Dimensions</span>
                            <span class="ns-file-meta-v">{safe_dims}</span>
                        </div>
                        <div class="ns-file-meta-pill">
                            <span class="ns-file-meta-k">File Size</span>
                            <span class="ns-file-meta-v">{safe_size}</span>
                        </div>
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

    # ------------------------------------------------------------------
    # RIGHT COLUMN — ANALYSIS INFORMATION & PROGRESS SIMULATION
    # ------------------------------------------------------------------
    with col_analysis:
        with st.container(key="analysis_panel"):
            render_html(
                """
                <div class="ns-panel-head">
                    <h3 class="ns-panel-title">Analysis Overview</h3>
                    <span class="ns-format-pill">Demo Pipeline</span>
                </div>
                """
            )

            if analyse_clicked and st.session_state.uploaded_image is not None:
                status_slot = st.empty()
                progress_bar = st.progress(0)
                stages_slot = st.empty()

                for idx, (fraction, stage_title, stage_desc) in enumerate(ANALYSIS_STAGES, start=1):
                    status_slot.markdown(
                        f"**Step {idx} of {len(ANALYSIS_STAGES)}:** {stage_title} — _{stage_desc}_"
                    )
                    progress_bar.progress(int(fraction * 100))
                    stages_slot.markdown(_render_stage_list(active_index=idx), unsafe_allow_html=True)
                    time.sleep(0.55)

                result = run_analysis(
                    st.session_state.uploaded_image,
                    metadata=st.session_state.image_metadata,
                )
                st.session_state.analysis_result = result
                st.rerun()

            elif st.session_state.analysis_result is not None:
                res = st.session_state.analysis_result
                render_html(
                    f"""
                    <div class="ns-empty-state" style="padding-top:0.8rem;padding-bottom:0.8rem;">
                        <h4 class="ns-empty-title">Simulation Complete</h4>
                        <p class="ns-empty-desc" style="margin-bottom:1rem;">
                            Illustrative segmentation overlay generated for
                            <strong>{html.escape(str(res['image_dimensions']))}</strong>.
                            Inspect the side-by-side comparison below.
                        </p>
                        {_render_stage_list(all_done=True)}
                    </div>
                    """
                )

            elif st.session_state.uploaded_image is not None:
                meta = st.session_state.image_metadata or {}
                render_html(
                    f"""
                    <div class="ns-empty-state" style="padding-top:0.8rem;padding-bottom:0.8rem;">
                        <div class="ns-empty-visual" style="width:116px;height:116px;margin-bottom:0.9rem;">
                            {_READY_ILLUSTRATION_SVG}
                        </div>
                        <h4 class="ns-empty-title">Scan Loaded — Ready to Analyse</h4>
                        <p class="ns-empty-desc" style="margin-bottom:1rem;">
                            <strong>{html.escape(str(meta.get('filename', 'MRI Scan')))}</strong> is validated
                            and queued. Click <strong>Analyse MRI Scan</strong> to run the simulated
                            visualisation stages.
                        </p>
                        {_render_stage_list(active_index=0)}
                    </div>
                    """
                )

            else:
                render_html(
                    f"""
                    <div class="ns-empty-state">
                        <div class="ns-empty-visual">
                            {_READY_ILLUSTRATION_SVG}
                        </div>
                        <h4 class="ns-empty-title">Ready for Analysis</h4>
                        <p class="ns-empty-desc">
                            Your scan analysis and visualisation results will appear here once an image is uploaded.
                        </p>
                        {_render_stage_list(active_index=0)}
                    </div>
                    """
                )
