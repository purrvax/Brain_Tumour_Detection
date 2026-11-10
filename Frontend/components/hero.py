"""Editorial hero section featuring the 3D translucent glass brain visual."""

from __future__ import annotations

from components import render_html


def render_hero(brain_image_b64: str) -> None:
    """Render the two-column hero section with editorial typography and glass brain."""
    arrow_svg = (
        '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<path d="M3.5 8H12.5M12.5 8L8.5 4M12.5 8L8.5 12" '
        'stroke="#F8F7F2" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )

    render_html(
        f"""
        <span id="home" class="ns-anchor"></span>
        <section class="ns-hero" aria-label="Hero">
            <div class="ns-hero-copy">
                <span class="ns-hero-eyebrow"> CLASSICAL IMAGE PROCESSING | CASE STUDY </span>
                <h1 class="ns-hero-title">
                    <span>BRAIN TUMOUR</span>
                    <span class="ns-title-accent">DETECTION</span>
                </h1>
                <p class="ns-hero-lead">
                    From noisy MRI to a measured tumour: filtering, thresholding and morphology locate the suspected region and estimate its area.
                </p>
                <div class="ns-hero-actions">
                    <a class="ns-btn-primary" href="#detection">
                        <span>Explore Detection</span>
                        {arrow_svg}
                    </a>
                    <a class="ns-btn-secondary" href="#about">Learn More</a>
                </div>
            </div>

            <div class="ns-brain-stage">
                <div class="ns-brain-aura"></div>
                <div class="ns-brain-highlight"></div>
                <div class="ns-brain-float">
                    <img
                        class="ns-brain-img"
                        src="data:image/png;base64,{brain_image_b64}"
                        alt="Translucent 3D glass anatomical human brain illustration"
                    />
                </div>
                <div class="ns-brain-shadow"></div>
            </div>
        </section>
        """
    )
