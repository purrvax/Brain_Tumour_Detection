"""Download-only component for the case study PDF."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


_FRONTEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = _FRONTEND_DIR / "assets" / "reports" / "IP-Case Study.pdf"


@st.cache_data(show_spinner=False)
def _load_report(report_path_str: str) -> tuple[bool, str, bytes | None]:
    """Load the original PDF bytes for the download button."""
    report_path = Path(report_path_str)

    if not report_path.exists() or not report_path.is_file():
        return False, report_path.name, None

    return True, report_path.name, report_path.read_bytes()


def render_case_study_pdf(report_path: Path = DEFAULT_REPORT_PATH) -> None:
    """Render only a button to download the complete case study PDF."""
    exists, filename, pdf_bytes = _load_report(str(report_path))

    if not exists or pdf_bytes is None:
        st.error(
            "Case study PDF could not be located. "
            "Verify that 'IP-Case Study.pdf' exists in assets/reports/."
        )
        return

    st.download_button(
        label="Download Case Study PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        type="primary",
        key="btn_download_case_study_pdf",
        use_container_width=True,
    )
