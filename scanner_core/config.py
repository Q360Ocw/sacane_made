"""
Configuration module for Smart Document Scanner.
All tunable constants live here so behaviour can be changed
without touching the algorithm code.
"""

# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------
DETECTION_MAX_DIM = 1000          # image is downscaled to this size for analysis
CANNY_LOW = 40
CANNY_HIGH = 120
MORPH_KERNEL_SIZE = 7
MORPH_CLOSE_ITER = 2
MIN_AREA_RATIO = 0.15             # min fraction of image area a document may occupy
MAX_AREA_RATIO = 0.98
APPROX_EPSILON_RATIOS = (0.01, 0.02, 0.03, 0.05, 0.08)  # tried in order

# Rectangle scoring weights (higher = more important)
SCORE_WEIGHT_AREA = 1.0
SCORE_WEIGHT_RECTANGULARITY = 2.0
SCORE_WEIGHT_ANGLE = 1.5
SCORE_WEIGHT_CENTERING = 0.5

# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------
OUTPUT_MAX_DIM = 3500              # cap on the longer side of the warped output
OUTPUT_MIN_DIM = 600

# ---------------------------------------------------------------------------
# Shadow removal / illumination
# ---------------------------------------------------------------------------
SHADOW_BLUR_KERNEL = 55            # background estimation kernel (must be odd)
SHADOW_DILATE_KERNEL = 9

# ---------------------------------------------------------------------------
# Paper whitening / color preservation
# ---------------------------------------------------------------------------
# A pixel is considered "paper" (background) if its saturation is low
# and it is reasonably bright. Ink / stamps / photos have higher
# saturation or much lower lightness and are excluded from whitening.
PAPER_MAX_SATURATION = 55          # 0-255 (HSV S channel)
PAPER_MIN_VALUE = 120              # 0-255 (HSV V channel)
INK_PROTECT_SATURATION = 95        # pixels above this saturation are treated as ink/content
                                    # (kept generous so a color-tinted light source, which raises
                                    # the paper's own saturation, is not mistaken for colored ink)
TEXT_PROTECT_MAX_VALUE = 110       # dark pixels (text strokes) are protected from whitening

WHITE_TARGET_PERCENTILE = 90       # percentile of paper-region lightness used as "white" reference
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ---------------------------------------------------------------------------
# Text enhancement
# ---------------------------------------------------------------------------
SHARPEN_AMOUNT = 0.6                # unsharp mask strength (0 = off)
SHARPEN_RADIUS = 3
DENOISE_H = 6                       # fastNlMeansDenoising strength

# ---------------------------------------------------------------------------
# Quality control thresholds
# ---------------------------------------------------------------------------
QC_MIN_DOC_AREA_RATIO = 0.1
QC_MAX_ASPECT_DISTORTION = 0.35     # relative change vs. detected quad allowed
QC_OVEREXPOSED_RATIO = 0.35         # fraction of near-white(255) pixels considered overexposed
QC_UNDEREXPOSED_MEAN = 40           # mean brightness below this -> too dark
QC_ARTIFICIAL_WHITE_RATIO = 0.985   # fraction of pixels == 255 that looks "burned"/artificial

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
JPEG_QUALITY = 95
PDF_DPI = 300

# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------
MODE_ORIGINAL = "Original"
MODE_COLOR = "Color"
MODE_CLEAN = "Clean"
MODE_DOCUMENT = "Document"
MODE_BW = "Black & White"

ALL_MODES = [MODE_ORIGINAL, MODE_COLOR, MODE_CLEAN, MODE_DOCUMENT, MODE_BW]
