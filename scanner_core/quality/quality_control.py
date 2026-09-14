"""
Post-processing quality control.

Runs a set of conservative heuristic checks and returns a list of
human-readable warnings (empty list = all good). The application must
NOT silently auto-save when warnings are present — the UI shows them and
lets the user go back to manual corner correction instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config


@dataclass
class QualityReport:
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.warnings) == 0


def check_detection_confidence(confidence: float, success: bool) -> list[str]:
    warnings = []
    if not success:
        warnings.append("تعذر اكتشاف الوثيقة تلقائيًا بثقة كافية. يرجى تحديد الزوايا يدويًا.")
    elif confidence < 0.35:
        warnings.append("مستوى الثقة في اكتشاف حدود الوثيقة منخفض. يُنصح بمراجعة الزوايا يدويًا.")
    return warnings


def check_aspect_ratio(quad: np.ndarray) -> list[str]:
    from ..utils import distance, order_points

    q = order_points(quad)
    tl, tr, br, bl = q
    top = distance(tl, tr)
    bottom = distance(bl, br)
    left = distance(tl, bl)
    right = distance(tr, br)

    warnings = []
    if min(top, bottom) > 0 and abs(top - bottom) / max(top, bottom) > config.QC_MAX_ASPECT_DISTORTION:
        warnings.append("الحواف العلوية والسفلية للوثيقة غير متناسقة بشكل كبير — تحقق من الزوايا.")
    if min(left, right) > 0 and abs(left - right) / max(left, right) > config.QC_MAX_ASPECT_DISTORTION:
        warnings.append("الحواف الجانبية للوثيقة غير متناسقة بشكل كبير — تحقق من الزوايا.")
    return warnings


def check_exposure(final_bgr: np.ndarray) -> list[str]:
    warnings = []
    gray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    overexposed_ratio = float(np.mean(gray >= 250))
    burned_white_ratio = float(np.mean(gray == 255))

    if mean_brightness < config.QC_UNDEREXPOSED_MEAN:
        warnings.append("الصورة الناتجة داكنة جدًا — حاول تحسين الإضاءة عند التصوير أو أعد المعالجة.")
    if overexposed_ratio > config.QC_OVEREXPOSED_RATIO:
        warnings.append("قد تكون الصورة الناتجة شديدة السطوع (Overexposed) في مناطق واسعة منها.")
    if burned_white_ratio > config.QC_ARTIFICIAL_WHITE_RATIO:
        warnings.append("تبييض الورقة يبدو مصطنعًا (بياض شديد التوحد) — قد يكون تم فقد بعض التفاصيل.")
    return warnings


def check_content_area(quad: np.ndarray, image_shape) -> list[str]:
    h, w = image_shape[:2]
    area = cv2.contourArea(quad.astype("float32"))
    ratio = area / float(h * w)
    warnings = []
    if ratio < config.QC_MIN_DOC_AREA_RATIO:
        warnings.append("المساحة المكتشفة للوثيقة صغيرة جدًا مقارنة بالصورة — تحقق من الاكتشاف.")
    return warnings


def run_quality_control(
    *,
    original_image: np.ndarray,
    quad: np.ndarray,
    detection_confidence: float,
    detection_success: bool,
    final_image: np.ndarray,
) -> QualityReport:
    warnings: list[str] = []
    warnings += check_detection_confidence(detection_confidence, detection_success)
    warnings += check_aspect_ratio(quad)
    warnings += check_content_area(quad, original_image.shape)
    warnings += check_exposure(final_image)
    return QualityReport(warnings=warnings)
