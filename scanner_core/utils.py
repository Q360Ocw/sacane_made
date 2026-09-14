"""
General purpose helper functions shared across the pipeline.
"""
from __future__ import annotations

import numpy as np
import cv2


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as: top-left, top-right, bottom-right, bottom-left.
    Works even when the quadrilateral is rotated, using the sum/diff trick.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]        # top-left     -> smallest sum
    ordered[2] = pts[np.argmax(s)]        # bottom-right -> largest sum
    ordered[1] = pts[np.argmin(diff)]     # top-right    -> smallest (y-x)
    ordered[3] = pts[np.argmax(diff)]     # bottom-left  -> largest (y-x)
    return ordered


def distance(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1, dtype="float64") - np.array(p2, dtype="float64")))


def clamp_point(pt, width, height):
    x, y = pt
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return (x, y)


def to_qimage(cv_img):
    """Convert a BGR or grayscale OpenCV image into a QImage (imported lazily to avoid
    requiring PyQt in headless / test contexts)."""
    from PyQt5.QtGui import QImage

    if cv_img is None:
        return None

    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        bytes_per_line = w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return qimg.copy()

    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return qimg.copy()


def resize_keep_aspect(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize so the longer side equals max_dim (only downscales). Returns (image, scale)."""
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= max_dim:
        return img.copy(), 1.0
    scale = max_dim / float(longer)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def safe_odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def load_image_unicode_safe(path: str):
    """cv2.imread fails on some unicode paths on Windows; this works around that
    by decoding through numpy's fromfile."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return cv2.imread(path)


def save_image_unicode_safe(path: str, img: np.ndarray, quality: int | None = None) -> bool:
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ".png"
    params = []
    if ext in (".jpg", ".jpeg") and quality is not None:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(path)
    return True
