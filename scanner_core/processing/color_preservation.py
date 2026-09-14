"""
Color preservation safety net.

Even though paper_whitening.py already restricts corrections to the paper
mask, this module adds an extra, independent safety pass: it detects
strongly colored / saturated regions (blue ink, red stamps, logos, photos,
signatures) directly on the ORIGINAL image and guarantees that the final
processed image keeps those pixels close to their original color, by
blending back toward the original wherever content confidence is high.

This is intentionally conservative: when in doubt, favor the original pixel.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config


def build_content_protection_mask(original_bgr: np.ndarray) -> np.ndarray:
    """Returns float32 mask [0,1]: 1.0 = definitely protect (strong ink/photo/
    stamp color or dark text stroke), 0.0 = safe to adjust freely."""
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    saturated = np.clip((s - config.INK_PROTECT_SATURATION) / max(255 - config.INK_PROTECT_SATURATION, 1), 0, 1)
    dark_text = np.clip((config.TEXT_PROTECT_MAX_VALUE - gray) / max(config.TEXT_PROTECT_MAX_VALUE, 1), 0, 1)

    mask = np.maximum(saturated, dark_text)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)
    return np.clip(mask, 0.0, 1.0)


def protect_original_colors(original_bgr: np.ndarray, processed_bgr: np.ndarray, extra_strength: float = 1.0) -> np.ndarray:
    """Blend `processed_bgr` back toward `original_bgr` in protected regions."""
    if original_bgr.shape != processed_bgr.shape:
        processed_bgr = cv2.resize(processed_bgr, (original_bgr.shape[1], original_bgr.shape[0]))

    mask = build_content_protection_mask(original_bgr) * np.clip(extra_strength, 0, 1)
    mask3 = mask[..., None]

    orig_f = original_bgr.astype(np.float32)
    proc_f = processed_bgr.astype(np.float32)

    blended = proc_f * (1 - mask3) + orig_f * mask3
    return np.clip(blended, 0, 255).astype(np.uint8)


def to_natural_black_and_white(bgr_image: np.ndarray) -> np.ndarray:
    """Professional B/W conversion for the *Black & White* mode (explicit user
    choice only, never the default). Uses a weighted channel mix rather than
    a naive average so colored text/stamps convert to sensible gray values
    instead of disappearing."""
    b, g, r = cv2.split(bgr_image.astype(np.float32))
    # Perceptual luminance weights.
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
