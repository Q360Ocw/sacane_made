"""
Corner refinement.

Given an approximate quadrilateral (e.g. from document_detector), this module
tries to snap each corner to the strongest nearby edge intersection using
cv2.cornerSubPix, which improves precision without inventing new geometry:
if no strong local edge is found, the original point is kept unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..utils import order_points


def refine_corners(bgr_image: np.ndarray, quad: np.ndarray, window: int = 25) -> np.ndarray:
    """Refine 4 corner points to sub-pixel accuracy using local corner detection.

    quad: (4,2) array in ORIGINAL image coordinates, any order.
    Returns an ordered (TL,TR,BR,BL) refined quad, same coordinate space.
    """
    quad = order_points(quad)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    pts = quad.reshape(-1, 1, 2).astype(np.float32)

    win = max(5, window)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    try:
        refined = cv2.cornerSubPix(gray, pts.copy(), (win, win), (-1, -1), criteria)
        refined = refined.reshape(4, 2)
    except cv2.error:
        return quad

    # Guard against cornerSubPix drifting a point far outside a sane radius
    # (this can happen on flat, low-texture regions). If it drifted too much,
    # keep the original point instead of trusting a bogus correction.
    max_drift = max(h, w) * 0.03
    out = quad.copy()
    for i in range(4):
        drift = np.linalg.norm(refined[i] - quad[i])
        if drift <= max_drift:
            out[i] = refined[i]

    # Clamp to image bounds
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)

    return order_points(out)
