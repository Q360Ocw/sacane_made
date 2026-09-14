"""
Export the final processed document to PNG, JPEG, or PDF, without
unnecessary resolution loss.
"""
from __future__ import annotations

import io
import os

import cv2
import numpy as np

from .. import config
from ..utils import save_image_unicode_safe


def export_image(bgr_image: np.ndarray, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        ok = save_image_unicode_safe(path, bgr_image, quality=config.JPEG_QUALITY)
    elif ext == ".png":
        ok = save_image_unicode_safe(path, bgr_image)
    else:
        raise ValueError(f"صيغة غير مدعومة: {ext}")
    if not ok:
        raise IOError(f"فشل حفظ الملف في: {path}")


def export_pdf(bgr_image: np.ndarray, path: str, dpi: int = None) -> None:
    """Wrap the processed document image into a single-page PDF at the given
    DPI, preserving full resolution (no down-sampling)."""
    import img2pdf

    dpi = dpi or config.PDF_DPI

    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    if not ok:
        raise IOError("فشل تجهيز الصورة لإنشاء ملف PDF.")

    pdf_bytes = img2pdf.convert(
        buf.tobytes(),
        dpi=dpi,
    )
    with open(path, "wb") as f:
        f.write(pdf_bytes)


def export(bgr_image: np.ndarray, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        export_pdf(bgr_image, path)
    else:
        export_image(bgr_image, path)
