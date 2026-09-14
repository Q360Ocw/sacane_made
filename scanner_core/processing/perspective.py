"""
Perspective correction using a homography computed from the 4 document
corners. The output size is derived from the actual measured edge lengths
of the quadrilateral (not a fixed size), so the document's true aspect
ratio is preserved and nothing gets stretched or squeezed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ..utils import order_points, distance


def compute_output_size(quad: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = quad

    width_top = distance(tl, tr)
    width_bottom = distance(bl, br)
    max_width = max(width_top, width_bottom)

    height_left = distance(tl, bl)
    height_right = distance(tr, br)
    max_height = max(height_left, height_right)

    max_width = max(max_width, 1.0)
    max_height = max(max_height, 1.0)

    # Scale so the longer side does not exceed OUTPUT_MAX_DIM, and the
    # shorter side does not fall below OUTPUT_MIN_DIM (unless the source
    # was already smaller than that).
    longer = max(max_width, max_height)
    if longer > config.OUTPUT_MAX_DIM:
        s = config.OUTPUT_MAX_DIM / longer
        max_width *= s
        max_height *= s

    shorter = min(max_width, max_height)
    if shorter < config.OUTPUT_MIN_DIM and shorter > 0:
        s = min(config.OUTPUT_MIN_DIM / shorter, 3.0)  # don't over-upscale blurry sources
        max_width *= s
        max_height *= s

    return max(1, int(round(max_width))), max(1, int(round(max_height)))


def warp_document(bgr_image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Apply perspective transform. quad may be in any order; it is normalized
    to TL,TR,BR,BL first. Returns the warped (straightened) document image."""
    quad = order_points(quad)
    out_w, out_h = compute_output_size(quad)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    warped = cv2.warpPerspective(
        bgr_image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return warped
