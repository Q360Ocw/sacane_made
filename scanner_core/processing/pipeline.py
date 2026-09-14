"""
Full processing pipeline that turns a raw photo + a chosen 4-point quad into
a finished document image, according to one of the five processing modes.

    ORIGINAL PHOTO
        -> PERSPECTIVE CORRECTION (crop + straighten, background removed)
        -> [mode-dependent steps]
        -> FINAL DOCUMENT
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config
from .perspective import warp_document
from .shadow_removal import remove_shadows_luminance
from .paper_whitening import build_paper_mask, whiten_paper, brighten_paper_lightness
from .color_preservation import protect_original_colors, to_natural_black_and_white
from .text_enhancement import enhance_text, local_contrast_enhance


@dataclass
class PipelineResult:
    warped_original: np.ndarray   # perspective-corrected, unprocessed (for Before/After + QC)
    final: np.ndarray             # fully processed result for the chosen mode
    mode: str


def _clean_and_whiten(bgr_image: np.ndarray) -> np.ndarray:
    """Shared shadow-removal + paper-whitening stage used by Clean/Document modes."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    lab = remove_shadows_luminance(lab)
    de_shadowed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    paper_mask = build_paper_mask(de_shadowed)
    whitened = whiten_paper(de_shadowed, paper_mask)
    whitened = brighten_paper_lightness(whitened, paper_mask, strength=0.55)

    # Independent safety net: guarantee ink/stamps/photos stay close to the
    # colors of the *shadow-corrected* image (not the raw original, since we
    # do want shadows lifted off content too, just not recolored).
    protected = protect_original_colors(de_shadowed, whitened, extra_strength=0.9)
    return protected


def process_document(
    bgr_image: np.ndarray,
    quad: np.ndarray,
    mode: str = config.MODE_DOCUMENT,
) -> PipelineResult:
    warped = warp_document(bgr_image, quad)

    if mode == config.MODE_ORIGINAL:
        final = warped.copy()

    elif mode == config.MODE_COLOR:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        lab = remove_shadows_luminance(lab)
        final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        final = protect_original_colors(warped, final, extra_strength=0.6)

    elif mode == config.MODE_CLEAN:
        final = _clean_and_whiten(warped)

    elif mode == config.MODE_DOCUMENT:
        cleaned = _clean_and_whiten(warped)
        enhanced = enhance_text(cleaned, strength="normal")
        final = protect_original_colors(warped, enhanced, extra_strength=0.85)

    elif mode == config.MODE_BW:
        cleaned = _clean_and_whiten(warped)
        final = to_natural_black_and_white(cleaned)

    else:
        final = warped.copy()

    return PipelineResult(warped_original=warped, final=final, mode=mode)
