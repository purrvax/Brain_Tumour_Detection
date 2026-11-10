"""UI components package for the NeuroScan Streamlit frontend."""

from __future__ import annotations

import streamlit as st


def render_html(markup: str) -> None:
    """Render raw HTML safely via Streamlit without Markdown 4-space code block pitfalls."""
    compact = "\n".join(line.lstrip() for line in markup.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


__all__ = ["render_html"]
