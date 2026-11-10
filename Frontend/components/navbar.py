"""Minimal top navigation bar component."""

from __future__ import annotations

from components import render_html

_BRAIN_LOGO_SVG = (
    '<svg width="21" height="21" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M9.5 4.2C7.4 4.2 5.7 5.7 5.5 7.7C4.1 8.4 3.2 9.9 3.4 11.6C3.5 12.7 4.1 13.7 5 14.3'
    'C4.8 15.8 5.6 17.4 7 18.1C7.9 18.6 9 18.7 10 18.3C10.6 18.1 11.1 17.5 11.1 16.8V5.8'
    'C11.1 4.9 10.4 4.2 9.5 4.2Z" stroke="#465343" stroke-width="1.5" stroke-linecap="round"/>'
    '<path d="M14.5 4.2C16.6 4.2 18.3 5.7 18.5 7.7C19.9 8.4 20.8 9.9 20.6 11.6C20.5 12.7 19.9 13.7 19 14.3'
    'C19.2 15.8 18.4 17.4 17 18.1C16.1 18.6 15 18.7 14 18.3C13.4 18.1 12.9 17.5 12.9 16.8V5.8'
    'C12.9 4.9 13.6 4.2 14.5 4.2Z" stroke="#59664F" stroke-width="1.5" stroke-linecap="round"/>'
    '<path d="M8.6 9.3C7.7 9.6 7.2 10.4 7.3 11.3M8.9 13.7C8 13.9 7.3 14.5 7.1 15.4'
    'M15.4 9.3C16.3 9.6 16.8 10.4 16.7 11.3M15.1 13.7C16 13.9 16.7 14.5 16.9 15.4" '
    'stroke="#87977B" stroke-width="1.35" stroke-linecap="round"/>'
    '</svg>'
)


def render_navbar() -> None:
    """Render the minimal glassmorphic top navigation bar."""
    render_html(
        f"""
        <nav class="ns-navbar" aria-label="Primary Navigation">
            <a class="ns-nav-brand" href="#home">
                <span class="ns-logo-badge">{_BRAIN_LOGO_SVG}</span>
                <span class="ns-brand-name">NeuroSegmentation-Classic</span>
            </a>
            <div class="ns-nav-right">
                <div class="ns-nav-links">
                    <a class="ns-nav-link" href="#home">Home</a>
                    <a class="ns-nav-link" href="#detection">Detection</a>
                    <a class="ns-nav-link" href="#about">About</a>
                </div>
            </div>
        </nav>
        """
    )
