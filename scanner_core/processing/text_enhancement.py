"""
Text clarity enhancement using only classical, information-preserving
operations: denoising, local contrast enhancement (CLAHE) and a controlled
unsharp mask. No generative/hallucinatory reconstruction of characters is
performed anywhere in this module.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config


def denoise(bgr_image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        bgr_image, None, h=config.DENOISE_H, hColor=config.DENOISE_H, templateWindowSize=7, searchWindowSize=21
    )


def local_contrast_enhance(bgr_image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def unsharp_mask(bgr_image: np.ndarray, amount: float = None, radius: int = None) -> np.ndarray:
    amount = config.SHARPEN_AMOUNT if amount is None else amount
    radius = config.SHARPEN_RADIUS if radius is None else radius
    if amount <= 0:
        return bgr_image
    blurred = cv2.GaussianBlur(bgr_image, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(bgr_image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_text(bgr_image: np.ndarray, strength: str = "normal") -> np.ndarray:
    """strength: 'light' | 'normal' | 'strong' — controls how aggressive the
    sharpening pass is; denoise + CLAHE stay the same for stability."""
    out = denoise(bgr_image)
    out = local_contrast_enhance(out)

    amount_map = {"light": 0.3, "normal": config.SHARPEN_AMOUNT, "strong": 0.9}
    out = unsharp_mask(out, amount=amount_map.get(strength, config.SHARPEN_AMOUNT))
    return out
