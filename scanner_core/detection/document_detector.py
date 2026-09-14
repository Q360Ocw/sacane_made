"""
Multi-stage document detection.

Pipeline:
  1. Resize for analysis
  2. Noise reduction (bilateral filter -> preserves edges)
  3. Grayscale
  4. Contrast normalization (CLAHE)
  5. Edge detection (Canny, tried at a couple of thresholds)
  6. Morphological closing (bridges broken edges)
  7. Contour detection
  8. Polygon approximation (several epsilon ratios attempted)
  9. Rectangle scoring (area, rectangularity, angle sanity, centering)
 10. Fallback: largest 4-point hull, then full-image quad if nothing plausible found.

The result is always a 4-point quadrilateral in ORIGINAL image coordinates,
plus a confidence score and a human readable status describing how it was
obtained (used by the UI to warn the user when confidence is low).
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .. import config
from ..utils import order_points, resize_keep_aspect


@dataclass
class DetectionResult:
    quad: np.ndarray            # (4,2) float32, ORIGINAL image coordinates, ordered TL,TR,BR,BL
    confidence: float           # 0..1
    method: str                 # description of how it was found
    success: bool               # False if we had to fall back to the full image


def _angle_cos(p0, p1, p2) -> float:
    d1, d2 = (p0 - p1).astype("float64"), (p2 - p1).astype("float64")
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))


def _rectangularity_score(quad: np.ndarray) -> float:
    """1.0 = perfect rectangle (all interior angles ~90deg), lower otherwise."""
    pts = quad.reshape(4, 2)
    cosines = []
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        cosines.append(_angle_cos(p0, p1, p2))
    max_cos = max(cosines)  # 0 == perfect right angle
    return float(max(0.0, 1.0 - max_cos))


def _score_candidate(quad: np.ndarray, img_area: float, img_center) -> float:
    area = cv2.contourArea(quad.astype("float32"))
    area_ratio = area / img_area
    if area_ratio < config.MIN_AREA_RATIO or area_ratio > config.MAX_AREA_RATIO:
        return -1.0

    rect_score = _rectangularity_score(quad)

    # Convexity: a real document should be convex.
    if not cv2.isContourConvex(quad.astype("float32").reshape(-1, 1, 2).astype(np.int32)):
        rect_score *= 0.5

    # centering: candidates whose centroid is close to the image centre score higher
    m = cv2.moments(quad.astype("float32"))
    if m["m00"] != 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    else:
        cx, cy = img_center
    center_dist = np.hypot(cx - img_center[0], cy - img_center[1])
    max_dist = np.hypot(img_center[0], img_center[1])
    centering_score = 1.0 - min(1.0, center_dist / (max_dist + 1e-9))

    score = (
        config.SCORE_WEIGHT_AREA * min(area_ratio, 1.0)
        + config.SCORE_WEIGHT_RECTANGULARITY * rect_score
        + config.SCORE_WEIGHT_CENTERING * centering_score
    )
    return float(score)


def _find_candidates(edges: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri < 50:
            continue
        for eps_ratio in config.APPROX_EPSILON_RATIOS:
            approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
            if len(approx) == 4:
                candidates.append(approx.reshape(4, 2).astype("float32"))
                break
        else:
            # try convex hull reduced to 4 points as a fallback candidate
            hull = cv2.convexHull(c)
            if len(hull) >= 4:
                peri_h = cv2.arcLength(hull, True)
                for eps_ratio in config.APPROX_EPSILON_RATIOS:
                    approx = cv2.approxPolyDP(hull, eps_ratio * peri_h, True)
                    if len(approx) == 4:
                        candidates.append(approx.reshape(4, 2).astype("float32"))
                        break
    return candidates


def _build_edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    """Produce a few complementary binary edge/foreground maps so the detector
    is robust to low contrast between the document and the background."""
    maps = []

    # 1) Classic Canny on a denoised, contrast-normalized image.
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    normalized = clahe.apply(denoised)
    edges = cv2.Canny(normalized, config.CANNY_LOW, config.CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.MORPH_KERNEL_SIZE,) * 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITER)
    edges = cv2.dilate(edges, kernel, iterations=1)
    maps.append(edges)

    # 2) Adaptive threshold -> good when document/background have similar edges
    #    but different average brightness.
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    maps.append(thresh)

    # 3) Otsu global threshold -> good for a clearly lighter/darker document.
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_edges = cv2.Canny(otsu, 30, 100)
    otsu_edges = cv2.dilate(otsu_edges, kernel, iterations=1)
    maps.append(otsu_edges)

    return maps


def detect_document(bgr_image: np.ndarray) -> DetectionResult:
    """Main entry point. Returns a DetectionResult with a quad in ORIGINAL coordinates."""
    h, w = bgr_image.shape[:2]
    analysis_img, scale = resize_keep_aspect(bgr_image, config.DETECTION_MAX_DIM)
    ah, aw = analysis_img.shape[:2]
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)

    img_area = float(ah * aw)
    img_center = (aw / 2.0, ah / 2.0)

    best_quad = None
    best_score = -1.0

    for edge_map in _build_edge_maps(gray):
        for cand in _find_candidates(edge_map):
            score = _score_candidate(cand, img_area, img_center)
            if score > best_score:
                best_score = score
                best_quad = cand

    if best_quad is not None and best_score > 0:
        quad_full = order_points(best_quad) / scale
        confidence = float(min(1.0, best_score / (config.SCORE_WEIGHT_AREA + config.SCORE_WEIGHT_RECTANGULARITY + config.SCORE_WEIGHT_CENTERING)))
        return DetectionResult(quad=quad_full, confidence=confidence, method="contour+scoring", success=True)

    # Fallback 1: use the minAreaRect of the largest foreground blob from Otsu.
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / img_area
        if area_ratio > 0.1:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            quad_full = order_points(box) / scale
            return DetectionResult(quad=quad_full, confidence=0.35, method="min-area-rect fallback", success=True)

    # Fallback 2: give up gracefully — return the full image as the quad so the
    # user can correct corners manually instead of the app crashing or refusing.
    margin_x, margin_y = w * 0.03, h * 0.03
    quad_full = np.array(
        [
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y],
        ],
        dtype="float32",
    )
    return DetectionResult(quad=quad_full, confidence=0.0, method="full-image fallback (manual correction required)", success=False)
