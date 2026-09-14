"""
Shadow / uneven-illumination removal.

Approach (a well known and stable classical technique):
  1. Estimate a smooth "background illumination" map per channel by
     dilating the image (removes thin dark text/strokes) then heavily
     median/Gaussian blurring the result -> this approximates the paper's
     lighting gradient without the text.
  2. Divide the original channel by the estimated background and rescale.
     This flattens illumination (shadows, hotspots) while leaving the
     *relative* darkness of text strokes intact, because both the text
     and its local background are normalized together only by the smooth
     background trend, not by a texture-destroying blur of the text itself.

This operates on the luminance (L channel of LAB) so hue/saturation of ink,
stamps, and photos is not touched here — that is handled by color_preservation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ..utils import safe_odd


def _estimate_background(channel: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.SHADOW_DILATE_KERNEL,) * 2)
    dilated = cv2.dilate(channel, kernel)
    bg = cv2.medianBlur(dilated, safe_odd(config.SHADOW_BLUR_KERNEL))
    return bg


def remove_shadows_luminance(lab_image: np.ndarray) -> np.ndarray:
    """lab_image: image in LAB color space (uint8). Returns a new LAB image
    with the L channel shadow-corrected."""
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    bg = _estimate_background(l_channel)
    bg_f = bg.astype(np.float32) + 1.0
    l_f = l_channel.astype(np.float32)

    # Normalize: bring background towards a uniform target while preserving
    # the *contrast* of foreground details relative to their local background.
    target = float(np.percentile(bg_f, 90))
    corrected = l_f * (target / bg_f)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return cv2.merge([corrected, a_channel, b_channel])
