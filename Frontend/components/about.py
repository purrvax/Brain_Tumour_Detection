"""Educational section explaining brain tumours, MRI, and segmentation."""

from __future__ import annotations

from components import render_html

_MRI_ICON_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<circle cx="12" cy="12" r="8.2" stroke="#465343" stroke-width="1.5"/>'
    '<circle cx="12" cy="12" r="4.6" stroke="#87977B" stroke-width="1.4" stroke-dasharray="2.4 2.2"/>'
    '<path d="M12 2.2V5M12 19V21.8M2.2 12H5M19 12H21.8" stroke="#59664F" stroke-width="1.4" stroke-linecap="round"/>'
    '</svg>'
)

_SEGMENTATION_ICON_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="3.5" y="4.5" width="17" height="15" rx="3.5" stroke="#59664F" stroke-width="1.5"/>'
    '<path d="M10.4 9.2C12 8.2 14.4 8.6 15.5 10.2C16.4 11.5 16.1 13.7 14.8 14.8'
    'C13.4 15.9 10.9 15.8 9.8 14.4C8.7 13 8.9 10.2 10.4 9.2Z" '
    'fill="rgba(199, 187, 211, 0.45)" stroke="#756389" stroke-width="1.5"/>'
    '</svg>'
)

_AI_ICON_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M4.5 8.5H9.5M4.5 12H13.5M4.5 15.5H10.5" stroke="#59664F" stroke-width="1.5" stroke-linecap="round"/>'
    '<circle cx="16.5" cy="9.5" r="2.2" stroke="#A8564E" stroke-width="1.5"/>'
    '<circle cx="17.5" cy="15.5" r="1.8" stroke="#87977B" stroke-width="1.5"/>'
    '<path d="M16.5 11.7L17.1 13.7" stroke="#77786F" stroke-width="1.3" stroke-linecap="round"/>'
    '</svg>'
)


def render_about() -> None:
    """Render the Understanding Brain Tumours section with three glass cards."""
    render_html(
        f"""
        <div class="ns-divider"></div>
        <span id="about" class="ns-anchor"></span>
        <section class="ns-section" aria-label="Understanding Brain Tumours">
            <div class="ns-section-header">
                <span class="ns-eyebrow">Imaging Context</span>
                <h2 class="ns-section-title">Understanding Brain Tumours</h2>
                <p class="ns-section-desc">
                    Brain tumours are abnormal growths of cells within the brain or surrounding tissues.
                    Magnetic Resonance Imaging (MRI) provides detailed images of brain structures and helps
                    medical professionals examine abnormalities.
                </p>
            </div>

            <div class="ns-about-grid">
                <article class="ns-info-card ns-card-sage">
                    <div class="ns-card-icon ns-icon-sage">{_MRI_ICON_SVG}</div>
                    <span class="ns-card-tag">01 · Acquisition</span>
                    <h3 class="ns-card-title">Brain MRI</h3>
                    <p class="ns-card-text">
                        MRI uses strong magnetic fields and radio waves to generate high-contrast,
                        cross-sectional images of soft brain tissues without ionising radiation.
                    </p>
                </article>

                <article class="ns-info-card ns-card-lavender">
                    <div class="ns-card-icon ns-icon-lavender">{_SEGMENTATION_ICON_SVG}</div>
                    <span class="ns-card-tag">02 · Delineation</span>
                    <h3 class="ns-card-title">Tumour Segmentation</h3>
                    <p class="ns-card-text">
                        Image segmentation identifies and highlights distinct anatomical or abnormal
                        regions within medical scans to support clearer visual inspection.
                    </p>
                </article>

                <article class="ns-info-card ns-card-peach">
                    <div class="ns-card-icon ns-icon-peach">{_AI_ICON_SVG}</div>
                    <span class="ns-card-tag">03 · Interpretation</span>
                    <h3 class="ns-card-title">AI-Assisted Analysis</h3>
                    <p class="ns-card-text">
                        Machine learning models can help analyse MRI images and visualise suspected
                        abnormal regions alongside existing clinical review workflows.
                    </p>
                </article>
            </div>
        </section>
        """
    )
