"""Medical disclaimer and minimalist footer component."""

from __future__ import annotations

from components import render_html

_SHIELD_INFO_SVG = (
    '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9" stroke="#A8564E" stroke-width="1.6"/>'
    '<path d="M12 11V16.2" stroke="#A8564E" stroke-width="1.6" stroke-linecap="round"/>'
    '<circle cx="12" cy="7.8" r="1" fill="#A8564E"/>'
    '</svg>'
)


def render_footer() -> None:
    """Render the medical disclaimer banner and footer."""
    render_html(
        f"""
        <footer class="ns-footer" aria-label="Medical Disclaimer and Footer">
            <div class="ns-disclaimer-box">
                <span class="ns-disclaimer-icon">{_SHIELD_INFO_SVG}</span>
                <div>
                    <p class="ns-disclaimer-title">Medical &amp; Research Disclaimer</p>
                    <p class="ns-disclaimer-text">
                        NeuroScan is a frontend research prototype designed for medical imaging interface
                        demonstration only. All segmentation overlays and regional pixel measurements shown
                        in this interface are purely simulated illustrations — they do not represent real
                        machine learning predictions, clinical diagnoses, or medical advice.
                    </p>
                </div>
            </div>
            <div class="ns-footer-bottom">
                <span><strong>NeuroScan</strong> · Brain Tumour Detection Frontend Prototype</span>
                <span>Modular Architecture Ready for Backend Model Integration</span>
            </div>
        </footer>
        """
    )
