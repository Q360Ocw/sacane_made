"""
Smart, region-aware paper whitening.

Core idea: build a soft "paper mask" that identifies pixels that are the
BACKGROUND PAPER (low saturation, bright) versus DOCUMENT CONTENT (ink,
stamps, logos, photos, signatures -> higher saturation and/or dark strokes).

Only the paper region is whitened / color-cast corrected. The correction is
blended smoothly (not a hard cutoff) so there are no visible seams around
text or stamps, and it never uses a single global saturation/brightness
shift that would wash out colored ink.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config


def build_paper_mask(bgr_image: np.ndarray) -> np.ndarray:
    """Returns a float32 mask in [0,1]: 1.0 = confidently paper/background,
    0.0 = confidently content (ink, stamp, photo, dark text)."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32)
    v = v.astype(np.float32)

    # Paper likelihood grows as saturation drops and brightness rises.
    # Full confidence (1.0) at/under PAPER_MAX_SATURATION (typical for real
    # paper, even under a mild color-tinted light source), fading linearly
    # to 0 at INK_PROTECT_SATURATION (clearly colored ink/stamp/photo).
    lo, hi = config.PAPER_MAX_SATURATION, config.INK_PROTECT_SATURATION
    sat_score = 1.0 - np.clip((s - lo) / max(hi - lo, 1), 0, 1)
    val_score = np.clip((v - config.TEXT_PROTECT_MAX_VALUE) / max(255 - config.TEXT_PROTECT_MAX_VALUE, 1), 0, 1)

    mask = sat_score * val_score

    # Smooth the mask so transitions between paper and content are gradual,
    # avoiding harsh edges/halos around text and stamps.
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    return np.clip(mask, 0.0, 1.0)


def estimate_white_point(bgr_image: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    """Estimate the actual color of the paper (to correct color casts) using
    only high-confidence paper pixels. Returns a BGR triple (float)."""
    confident = paper_mask > 0.6
    if np.count_nonzero(confident) < 500:
        # Not enough confident paper pixels — fall back to the brightest
        # 5% of all pixels by luminance (still conservative).
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        thresh = np.percentile(gray, 95)
        confident = gray >= thresh

    pixels = bgr_image[confident].astype(np.float32)
    if pixels.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # Use a high percentile per channel (robust "white point") rather than the
    # mean, since we want the *brightest representative* paper tone.
    white_point = np.percentile(pixels, config.WHITE_TARGET_PERCENTILE, axis=0)
    white_point = np.clip(white_point, 30, 255)
    return white_point.astype(np.float32)


def whiten_paper(bgr_image: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    """Apply a gray-world style color-cast correction (removes yellow/orange/
    blue tint from lighting) scaled by the paper mask, and gently lift the
    paper's lightness towards white — content pixels are left untouched."""
    white_point = estimate_white_point(bgr_image, paper_mask)

    # Gains that would map the estimated white point to neutral 255,255,255.
    gains = 255.0 / np.clip(white_point, 1, 255)
    # Keep correction conservative: don't over-amplify any single channel,
    # which is what causes unnatural color shifts. The range is wide enough
    # to neutralize a typical warm/cool light cast (tungsten/fluorescent)
    # while still refusing extreme, implausible corrections.
    gains = np.clip(gains, 0.65, 1.55)

    img_f = bgr_image.astype(np.float32)
    corrected = img_f * gains.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0, 255)

    mask3 = paper_mask[..., None]
    blended = img_f * (1 - mask3) + corrected * mask3

    return np.clip(blended, 0, 255).astype(np.uint8)


def brighten_paper_lightness(bgr_image: np.ndarray, paper_mask: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Lift the L channel of paper regions toward a clean white, proportional
    to mask confidence and to how far the pixel already is from white -
    natural rather than a flat clip to 255."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    target = float(np.percentile(l_channel[paper_mask > 0.6], 95)) if np.any(paper_mask > 0.6) else 235.0
    target = min(max(target, 200.0), 245.0)  # never force pure blown-out white

    lift = (target - l_channel) * paper_mask * strength
    l_channel = np.clip(l_channel + lift, 0, 255)

    lab[:, :, 0] = l_channel
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
