"""Main Streamlit entry point for the NeuroScan Brain Tumour Detection frontend."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
ASSETS_DIR = FRONTEND_DIR / "assets"

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from components.about import render_about
from components.flux_background import render_flux_background
from components.footer import render_footer
from components.hero import render_hero
from components.navbar import render_navbar
from components.results import render_results
from components.upload import render_upload_and_analysis
from utils.image_processing import file_to_base64


def _configure_page() -> None:
    """Configure Streamlit page metadata and initial viewport behavior."""
    icon_path = ASSETS_DIR / "brain_illustration.png"
    st.set_page_config(
        page_title="NeuroScan — Brain Tumour Detection",
        page_icon=str(icon_path) if icon_path.exists() else None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


@st.cache_data(show_spinner=False)
def _load_brain_asset_b64() -> str:
    """Load and cache the base64-encoded 3D translucent glass brain asset."""
    return file_to_base64(ASSETS_DIR / "brain_illustration.png")


def _init_session_state() -> None:
    """Initialize required Streamlit session state keys safely."""
    defaults = {
        "uploader_key": 0,
        "uploaded_file_sig": None,
        "uploaded_image": None,
        "uploaded_preview_b64": None,
        "image_metadata": None,
        "upload_error": None,
        "analysis_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_scan_workflow() -> None:
    """Clear uploaded scan, analysis output, and reset the file uploader widget."""
    st.session_state.uploader_key += 1
    st.session_state.uploaded_file_sig = None
    st.session_state.uploaded_image = None
    st.session_state.uploaded_preview_b64 = None
    st.session_state.image_metadata = None
    st.session_state.upload_error = None
    st.session_state.analysis_result = None


def main() -> None:
    """Render the complete NeuroScan single-page medical imaging application."""
    _configure_page()
    _init_session_state()

    # Load custom light-theme editorial CSS without taking vertical layout space
    st.html(ASSETS_DIR / "styles.css")

    brain_b64 = _load_brain_asset_b64()

    render_flux_background()
    render_navbar()
    render_hero(brain_b64)
    render_about()
    render_upload_and_analysis(on_reset=reset_scan_workflow)
    render_results(on_reset=reset_scan_workflow)
    render_footer()


if __name__ == "__main__":
    main()
