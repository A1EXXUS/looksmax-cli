#!/usr/bin/env python3
"""Real-time facial-geometry CV worker for looksmax-cli.

Captures frames from a webcam, runs MediaPipe Face Mesh, and streams one
JSON object per line (JSON Lines) to stdout with a set of *heuristic*
geometric metrics. This is NOT a medical or scientific measurement tool --
see the project README for the full disclaimer.

The worker never writes frames or video to disk. Unless --no-preview is
passed, it also opens a live preview window (OpenCV highgui) with the
camera feed mirrored and the geometric heuristics drawn on top, purely for
visual feedback -- the JSON metrics on stdout are unaffected by it and
remain the actual contract with the Go core.

It is meant to be spawned as a subprocess by the Go core and communicates
metrics exclusively over stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time

import cv2
import mediapipe as mp

# Exit codes, kept distinct so the Go core can show a meaningful message.
EXIT_OK = 0
EXIT_CAMERA_UNAVAILABLE = 2
EXIT_RUNTIME_ERROR = 3

Point = tuple[float, float]

# --- MediaPipe Face Mesh landmark indices used for the heuristics below. ---
# These are approximate anatomical stand-ins, not clinically validated
# landmarks. Indices refer to the 468/478-point topology MediaPipe ships.
LM_FOREHEAD_TOP = 10
LM_CHIN = 152
LM_NOSE_BASE = 2
LM_UPPER_LIP = 0
LM_LOWER_LIP = 17

# Eye corners: (inner, outer) per eye.
LM_RIGHT_EYE_INNER = 133
LM_RIGHT_EYE_OUTER = 33
LM_LEFT_EYE_INNER = 362
LM_LEFT_EYE_OUTER = 263

# Jaw / gonion approximation and cheekbone / temple width points.
LM_JAW_LEFT = 172   # approx. left gonion (mandible angle)
LM_JAW_RIGHT = 397  # approx. right gonion
LM_CHEEK_LEFT = 234
LM_CHEEK_RIGHT = 454
LM_TEMPLE_LEFT = 127
LM_TEMPLE_RIGHT = 356
LM_EYEBROW_LEFT = 105
LM_EYEBROW_RIGHT = 334
LM_MOUTH_LEFT = 61
LM_MOUTH_RIGHT = 291

# Profile-view points (glabella/nasion/nose tip) -- these read as more
# meaningful when the head is turned toward a side profile, the same way
# canthal tilt reads best near-frontal. Still computed every frame either
# way, same as the rest of this module's heuristics.
LM_GLABELLA = 9    # approx. glabella (smooth area between the eyebrows)
LM_NASION = 6      # approx. nasion (nose bridge, between the eyes)
LM_NOSE_TIP = 4    # approx. nose tip (most anterior point of the nose)

# Landmark pairs mirrored across the face midline, used for the symmetry
# heuristic. Order is (left-of-image, right-of-image); which side is
# anatomically left/right doesn't matter since we only compare distances.
SYMMETRY_PAIRS: list[tuple[int, int]] = [
    (LM_RIGHT_EYE_OUTER, LM_LEFT_EYE_OUTER),
    (LM_RIGHT_EYE_INNER, LM_LEFT_EYE_INNER),
    (LM_EYEBROW_LEFT, LM_EYEBROW_RIGHT),
    (LM_MOUTH_LEFT, LM_MOUTH_RIGHT),
    (LM_CHEEK_LEFT, LM_CHEEK_RIGHT),
    (LM_JAW_LEFT, LM_JAW_RIGHT),
    (LM_TEMPLE_LEFT, LM_TEMPLE_RIGHT),
]


def _dist(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _to_z_values(landmarks, width: int) -> list[float]:
    """Convert MediaPipe's normalized z (relative depth) to a pixel-ish scale.

    Per MediaPipe's docs, z uses roughly the same scale as x, so
    multiplying by width puts it in units comparable to the x/y pixel
    coordinates. Smaller (more negative) z means closer to the camera.
    This is a monocular depth *estimate*, not a real depth sensor reading
    -- noisier and less validated than the plain 2D geometry above.
    """
    return [lm.z * width for lm in landmarks]


def _to_pixel_points(landmarks, width: int, height: int) -> list[Point]:
    """Convert MediaPipe's normalized (0-1) landmarks to pixel coordinates.

    This is required before computing any distance/angle so that the
    frame's aspect ratio doesn't distort the geometry.
    """
    return [(lm.x * width, lm.y * height) for lm in landmarks]


def compute_symmetry(points: list[Point]) -> float:
    """Estimate bilateral facial symmetry as a 0-100 score (100 = perfect).

    Heuristic: build a midline axis from forehead-top to chin, then for
    each mirrored landmark pair compare each point's perpendicular
    distance to that axis. The average relative difference across all
    pairs is converted into a 0-100 score. This is a rough approximation
    of symmetry, not a clinical measurement.
    """
    axis_top = points[LM_FOREHEAD_TOP]
    axis_bottom = points[LM_CHIN]
    ax, ay = axis_top
    bx, by = axis_bottom
    axis_dx, axis_dy = bx - ax, by - ay
    axis_len = math.hypot(axis_dx, axis_dy)
    if axis_len < 1e-6:
        return 0.0

    relative_diffs: list[float] = []
    for left_idx, right_idx in SYMMETRY_PAIRS:
        lx, ly = points[left_idx]
        rx, ry = points[right_idx]
        # Perpendicular (signed) distance from each point to the axis line,
        # via the 2D cross product magnitude divided by axis length.
        dist_left = abs((lx - ax) * axis_dy - (ly - ay) * axis_dx) / axis_len
        dist_right = abs((rx - ax) * axis_dy - (ry - ay) * axis_dx) / axis_len
        denom = (dist_left + dist_right) / 2.0
        if denom < 1e-6:
            continue
        relative_diffs.append(abs(dist_left - dist_right) / denom)

    if not relative_diffs:
        return 0.0

    avg_relative_diff = sum(relative_diffs) / len(relative_diffs)
    score = 100.0 * (1.0 - avg_relative_diff)
    return max(0.0, min(100.0, score))


def compute_canthal_tilt(points: list[Point]) -> float:
    """Estimate canthal tilt in degrees, averaged over both eyes.

    Canthal tilt is the angle of the inner-to-outer eye-corner line
    relative to horizontal. Positive values mean the outer corner sits
    higher than the inner corner ("positive"/upward tilt).
    """
    tilts: list[float] = []
    for inner_idx, outer_idx in (
        (LM_RIGHT_EYE_INNER, LM_RIGHT_EYE_OUTER),
        (LM_LEFT_EYE_INNER, LM_LEFT_EYE_OUTER),
    ):
        ix, iy = points[inner_idx]
        ox, oy = points[outer_idx]
        # Image y grows downward, so a higher outer corner means a smaller
        # oy, hence (iy - oy) > 0 when the outer corner is above the inner
        # one. abs(dx) removes the left/right mirroring ambiguity.
        dx = abs(ox - ix)
        dy = iy - oy
        if dx < 1e-6:
            continue
        tilts.append(math.degrees(math.atan2(dy, dx)))

    if not tilts:
        return 0.0
    return sum(tilts) / len(tilts)


def compute_gonial_angle(points: list[Point]) -> float:
    """Approximate the gonial (mandible) angle in degrees, averaged per side.

    Uses a temple/cheek point standing in for the ramus direction and the
    chin standing in for the mandible body direction, with the vertex at
    an approximate gonion landmark. This is a coarse visual approximation
    of the true cephalometric gonial angle.
    """
    angles: list[float] = []
    for temple_idx, gonion_idx in (
        (LM_TEMPLE_LEFT, LM_JAW_LEFT),
        (LM_TEMPLE_RIGHT, LM_JAW_RIGHT),
    ):
        tx, ty = points[temple_idx]
        gx, gy = points[gonion_idx]
        cx, cy = points[LM_CHIN]

        v1 = (tx - gx, ty - gy)
        v2 = (cx - gx, cy - gy)
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angles.append(math.degrees(math.acos(cos_angle)))

    if not angles:
        return 0.0
    return sum(angles) / len(angles)


def compute_bigonial_width_ratio(points: list[Point]) -> float:
    """Return jaw width (gonion-to-gonion) relative to face height."""
    jaw_width = _dist(points[LM_JAW_LEFT], points[LM_JAW_RIGHT])
    face_height = _dist(points[LM_FOREHEAD_TOP], points[LM_CHIN])
    if face_height < 1e-6:
        return 0.0
    return jaw_width / face_height


def compute_cheekbone_ratio(points: list[Point]) -> float:
    """Return cheekbone width relative to overall (temple-to-temple) face width.

    Note: this is a flat width comparison -- it says nothing about how far
    the cheekbone actually protrudes toward the camera (bone prominence).
    A narrow-vs-wide face can score the same here regardless of how
    developed the cheekbones are. See compute_cheekbone_prominence for a
    depth-based attempt at that instead.
    """
    cheek_width = _dist(points[LM_CHEEK_LEFT], points[LM_CHEEK_RIGHT])
    face_width = _dist(points[LM_TEMPLE_LEFT], points[LM_TEMPLE_RIGHT])
    if face_width < 1e-6:
        return 0.0
    return cheek_width / face_width


def compute_cheekbone_prominence(points: list[Point], z_values: list[float]) -> float:
    """Estimate how far the cheekbones protrude toward the camera vs. the temples.

    Uses MediaPipe's per-landmark relative depth (z) instead of a flat
    width ratio: positive means the cheek points sit closer to the camera
    than the temples (protruding/prominent cheekbones), ~0 or negative
    means roughly flat. Normalized by face width so it's comparable across
    distances from the camera.

    Monocular depth from a single 2D camera is inherently noisier and less
    validated than the plain x/y geometry the rest of this module uses --
    treat this as a rougher estimate than the ratio-based metrics.
    """
    baseline_z = (z_values[LM_TEMPLE_LEFT] + z_values[LM_TEMPLE_RIGHT]) / 2.0
    cheek_z = (z_values[LM_CHEEK_LEFT] + z_values[LM_CHEEK_RIGHT]) / 2.0
    face_width = _dist(points[LM_TEMPLE_LEFT], points[LM_TEMPLE_RIGHT])
    if face_width < 1e-6:
        return 0.0
    return (baseline_z - cheek_z) / face_width


def compute_facial_thirds_dev(points: list[Point]) -> float:
    """Return the deviation (in %) of the three facial "thirds" from equality.

    Uses brow-to-nose-base / nose-base-to-lip / lip-to-chin segments as a
    substitute for the classical trichion-based thirds, since the hairline
    is not reliably detected by Face Mesh. Returns the standard deviation
    of the three segment lengths as a percentage of their mean (0 = the
    three segments are perfectly equal).
    """
    brow_mid = (
        (points[LM_EYEBROW_LEFT][0] + points[LM_EYEBROW_RIGHT][0]) / 2.0,
        (points[LM_EYEBROW_LEFT][1] + points[LM_EYEBROW_RIGHT][1]) / 2.0,
    )
    third1 = _dist(brow_mid, points[LM_NOSE_BASE])
    third2 = _dist(points[LM_NOSE_BASE], points[LM_UPPER_LIP])
    third3 = _dist(points[LM_LOWER_LIP], points[LM_CHIN])

    lengths = [third1, third2, third3]
    mean_len = sum(lengths) / 3.0
    if mean_len < 1e-6:
        return 0.0

    variance = sum((length - mean_len) ** 2 for length in lengths) / 3.0
    std_dev = math.sqrt(variance)
    return (std_dev / mean_len) * 100.0


def _angle_deg_at_vertex(vertex: Point, a: Point, b: Point) -> float:
    """Return the angle (degrees) at `vertex` between rays to `a` and `b`."""
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (b[0] - vertex[0], b[1] - vertex[1])
    len1 = math.hypot(*v1)
    len2 = math.hypot(*v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def compute_facial_convexity_angle(points: list[Point]) -> float:
    """Approximate the profile "facial convexity" angle, in degrees.

    Angle at the nose tip between the lines to the glabella (brow) and to
    the chin. Reads best in a near-side profile; a straighter (larger)
    angle means a flatter profile, a smaller angle a more convex one. This
    is a coarse 2D stand-in for the true cephalometric angle (which is
    normally measured from a true lateral X-ray), not a clinical measure.
    """
    return _angle_deg_at_vertex(points[LM_NOSE_TIP], points[LM_GLABELLA], points[LM_CHIN])


def compute_nasofrontal_angle(points: list[Point]) -> float:
    """Approximate the nasofrontal angle (forehead-to-nose-bridge), in degrees.

    Angle at the nasion between the line up to the glabella (forehead) and
    the line down to the nose tip (nasal dorsum). Reads best in profile.
    """
    return _angle_deg_at_vertex(points[LM_NASION], points[LM_GLABELLA], points[LM_NOSE_TIP])


def compute_nasolabial_angle(points: list[Point]) -> float:
    """Approximate the nasolabial angle (nose base-to-upper lip), in degrees.

    Angle at the nose base (subnasale stand-in) between the line up to the
    nose tip (columella direction) and the line down to the upper lip.
    Reads best in profile.
    """
    return _angle_deg_at_vertex(points[LM_NOSE_BASE], points[LM_NOSE_TIP], points[LM_UPPER_LIP])


def compute_frontality_offset(points: list[Point]) -> float:
    """Return how far off-center the nose tip sits, as a fraction of face width.

    ~0 means the nose is roughly centered between the temples (a frontal
    face); it grows toward ~0.4-0.5+ as the head turns toward profile. Most
    of the other metrics (symmetry, gonial angle, bigonial width,
    cheekbone ratio) rely on both sides of the face being visible in a
    comparable way, so this is used to decide whether the composite
    score/tier can be trusted for the current frame -- see
    is_frontal_enough and compute_composite_score.
    """
    temple_mid_x = (points[LM_TEMPLE_LEFT][0] + points[LM_TEMPLE_RIGHT][0]) / 2.0
    face_width = abs(points[LM_TEMPLE_LEFT][0] - points[LM_TEMPLE_RIGHT][0])
    if face_width < 1e-6:
        return 1.0
    return abs(points[LM_NOSE_TIP][0] - temple_mid_x) / face_width


# Above this, the head is turned far enough that the bilateral-pair metrics
# (symmetry, gonial angle, bigonial width, cheekbone ratio) can no longer be
# trusted, so the composite score/tier is hidden rather than shown wrong.
# Tune alongside core/tiers.go's maxFrontalityOffsetForScore (keep in sync).
_FRONTALITY_MAX = 0.22


def is_frontal_enough(metrics: dict[str, float]) -> bool:
    """True when the face is frontal enough to trust the composite score."""
    return metrics.get("frontality_offset", 0.0) <= _FRONTALITY_MAX


def compute_metrics(landmarks, width: int, height: int) -> dict[str, float]:
    """Compute the full set of raw geometric metrics for one detected face."""
    points = _to_pixel_points(landmarks, width, height)
    z_values = _to_z_values(landmarks, width)
    return {
        "symmetry": round(compute_symmetry(points), 2),
        "canthal_tilt_deg": round(compute_canthal_tilt(points), 2),
        "gonial_angle_deg": round(compute_gonial_angle(points), 2),
        "bigonial_width_ratio": round(compute_bigonial_width_ratio(points), 3),
        "cheekbone_ratio": round(compute_cheekbone_ratio(points), 3),
        "facial_thirds_dev": round(compute_facial_thirds_dev(points), 2),
        "facial_convexity_deg": round(compute_facial_convexity_angle(points), 2),
        "nasofrontal_angle_deg": round(compute_nasofrontal_angle(points), 2),
        "nasolabial_angle_deg": round(compute_nasolabial_angle(points), 2),
        "frontality_offset": round(compute_frontality_offset(points), 3),
        "cheekbone_prominence": round(compute_cheekbone_prominence(points, z_values), 3),
    }


# Decimal precision per metric key, reapplied after temporal smoothing
# (which otherwise reintroduces long float tails via the weighted average).
_METRIC_DECIMALS = {
    "symmetry": 2,
    "canthal_tilt_deg": 2,
    "gonial_angle_deg": 2,
    "bigonial_width_ratio": 3,
    "cheekbone_ratio": 3,
    "facial_thirds_dev": 2,
    "facial_convexity_deg": 2,
    "nasofrontal_angle_deg": 2,
    "nasolabial_angle_deg": 2,
    "frontality_offset": 3,
    "cheekbone_prominence": 3,
}


class MetricsSmoother:
    """Exponential moving average over compute_metrics' numeric fields.

    Cuts down frame-to-frame jitter -- worst at extreme head angles, where
    MediaPipe's landmark predictions are least confident -- without adding
    much lag. Resets on any gap in face detection so it never blends two
    unrelated sightings (e.g. the face was lost and a different one, or the
    same one at a very different angle, showed up next).
    """

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._state: dict[str, float] | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, metrics: dict[str, float]) -> dict[str, float]:
        if self._state is None:
            self._state = dict(metrics)
        else:
            self._state = {
                key: self.alpha * value + (1 - self.alpha) * self._state.get(key, value)
                for key, value in metrics.items()
            }
        return {key: round(value, _METRIC_DECIMALS.get(key, 2)) for key, value in self._state.items()}


# --- Composite score / tier -------------------------------------------------
#
# Mirrors core/tiers.go exactly (weights, ideals, sigmas, tier bands) so the
# preview window and the Go TUI never disagree on the headline number. If
# you tune the calibration in tiers.go, mirror the change here too.

_IDEAL_CANTHAL_TILT_DEG = 4.0
_CANTHAL_TILT_SIGMA = 6.0
_IDEAL_GONIAL_ANGLE_DEG = 120.0
_GONIAL_ANGLE_SIGMA = 15.0
_IDEAL_BIGONIAL_WIDTH = 0.78
_BIGONIAL_WIDTH_SIGMA = 0.12
_IDEAL_CHEEKBONE_RATIO = 0.88
_CHEEKBONE_RATIO_SIGMA = 0.15
_FACIAL_THIRDS_DEV_SIGMA = 8.0

_SCORE_MIN = 1.0
_SCORE_MAX = 8.5

_SCORE_WEIGHTS = {
    "symmetry": 0.25,
    "canthal_tilt": 0.15,
    "gonial_angle": 0.15,
    "bigonial_width": 0.15,
    "cheekbone": 0.15,
    "facial_thirds": 0.15,
}

_TIER_BANDS = (
    (3.5, "Subhuman"),
    (5.0, "LTN"),
    (6.0, "MTN"),
    (7.0, "HTN"),
    (8.0, "Chadlite"),
)


def _gaussian_subscore(value: float, ideal: float, sigma: float) -> float:
    """Score how close value is to ideal on a 0-100 scale (see tiers.go)."""
    if sigma <= 0:
        return 0.0
    diff = value - ideal
    return 100.0 * math.exp(-(diff * diff) / (2 * sigma * sigma))


def compute_composite_score(metrics: dict[str, float]) -> tuple[float, str]:
    """Return (score in [SCORE_MIN, SCORE_MAX], tier name), per tiers.go."""
    symmetry_sub = max(0.0, min(100.0, metrics["symmetry"]))
    canthal_sub = _gaussian_subscore(metrics["canthal_tilt_deg"], _IDEAL_CANTHAL_TILT_DEG, _CANTHAL_TILT_SIGMA)
    gonial_sub = _gaussian_subscore(metrics["gonial_angle_deg"], _IDEAL_GONIAL_ANGLE_DEG, _GONIAL_ANGLE_SIGMA)
    bigonial_sub = _gaussian_subscore(metrics["bigonial_width_ratio"], _IDEAL_BIGONIAL_WIDTH, _BIGONIAL_WIDTH_SIGMA)
    cheekbone_sub = _gaussian_subscore(metrics["cheekbone_ratio"], _IDEAL_CHEEKBONE_RATIO, _CHEEKBONE_RATIO_SIGMA)
    thirds_sub = _gaussian_subscore(metrics["facial_thirds_dev"], 0.0, _FACIAL_THIRDS_DEV_SIGMA)

    composite = (
        _SCORE_WEIGHTS["symmetry"] * symmetry_sub
        + _SCORE_WEIGHTS["canthal_tilt"] * canthal_sub
        + _SCORE_WEIGHTS["gonial_angle"] * gonial_sub
        + _SCORE_WEIGHTS["bigonial_width"] * bigonial_sub
        + _SCORE_WEIGHTS["cheekbone"] * cheekbone_sub
        + _SCORE_WEIGHTS["facial_thirds"] * thirds_sub
    )
    composite = max(0.0, min(100.0, composite))
    score = _SCORE_MIN + (composite / 100.0) * (_SCORE_MAX - _SCORE_MIN)

    tier = "Chad"
    for threshold, name in _TIER_BANDS:
        if score < threshold:
            tier = name
            break
    return score, tier


PREVIEW_WINDOW_NAME = "looksmax-cli - live preview"

# BGR colors (OpenCV convention) for the overlay -- kept distinct per
# heuristic so it's clear which line belongs to which metric.
_COLOR_AXIS = (150, 150, 150)
_COLOR_CANTHAL = (255, 255, 0)
_COLOR_GONIAL = (0, 165, 255)
_COLOR_BIGONIAL = (0, 255, 0)
_COLOR_CHEEKBONE = (255, 0, 255)
_COLOR_THIRDS = (255, 200, 200)
_COLOR_PROFILE = (255, 255, 150)
_COLOR_NASOLABIAL = (0, 255, 255)
_COLOR_TEXT = (255, 255, 255)


def _pt(point: Point) -> tuple[int, int]:
    """Round a float pixel point to int (x, y) for OpenCV drawing calls."""
    return (round(point[0]), round(point[1]))


# (metric key, label, min, max, unit, bar color) -- min/max mirror
# core/ui.go's barSpecs so the preview window's bars and the Go TUI's bars
# agree on scale. Bar color matches the corresponding overlay line's color
# above, so it's visually obvious which bar goes with which line.
_BAR_SPECS = [
    ("symmetry", "symmetry", 0.0, 100.0, "", (255, 255, 255)),
    ("canthal_tilt_deg", "canthal tilt", -10.0, 20.0, "deg", (255, 255, 0)),
    ("gonial_angle_deg", "gonial angle", 90.0, 150.0, "deg", (0, 165, 255)),
    ("bigonial_width_ratio", "bigonial ratio", 0.5, 1.1, "", (0, 255, 0)),
    ("cheekbone_ratio", "cheekbone ratio", 0.5, 1.2, "", (255, 0, 255)),
    ("facial_thirds_dev", "thirds dev", 0.0, 40.0, "%", (255, 200, 200)),
    ("facial_convexity_deg", "convexity", 140.0, 190.0, "deg", (255, 255, 150)),
    ("nasofrontal_angle_deg", "nasofrontal", 90.0, 190.0, "deg", (255, 255, 150)),
    ("nasolabial_angle_deg", "nasolabial", 70.0, 190.0, "deg", (0, 255, 255)),
    # EXPERIMENTAL -- range picked from a single test session, not real
    # calibration data. Expect to retune once more faces are seen.
    ("cheekbone_prominence", "cheek prominence", -0.06, 0.02, "", (255, 0, 255)),
]

_BAR_FONT_SCALE = 0.75
_BAR_FONT_THICKNESS = 2
_BAR_LABEL_X = 12
_BAR_WIDTH = 180
_BAR_HEIGHT = 24
_BAR_ROW_HEIGHT = 42

# Label column width is however wide the longest label actually renders at
# this font, plus padding -- computed once at import time rather than
# hardcoded, the same idea as core/ui.go's dynamic labelWidth.
_BAR_X = _BAR_LABEL_X + max(
    cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _BAR_FONT_SCALE, _BAR_FONT_THICKNESS)[0][0]
    for _, label, *_rest in _BAR_SPECS
) + 20


def _draw_bar_row(frame, y: int, label: str, value: float, lo: float, hi: float, unit: str, color) -> None:
    """Draw one label + filled progress bar + value row, Go-TUI style."""
    text_y = y + _BAR_HEIGHT - 5
    cv2.putText(frame, label, (_BAR_LABEL_X, text_y), cv2.FONT_HERSHEY_SIMPLEX, _BAR_FONT_SCALE, (0, 0, 0), _BAR_FONT_THICKNESS + 3, cv2.LINE_AA)
    cv2.putText(frame, label, (_BAR_LABEL_X, text_y), cv2.FONT_HERSHEY_SIMPLEX, _BAR_FONT_SCALE, _COLOR_TEXT, _BAR_FONT_THICKNESS, cv2.LINE_AA)

    frac = 0.0 if hi <= lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled_w = round(frac * _BAR_WIDTH)
    cv2.rectangle(frame, (_BAR_X, y), (_BAR_X + _BAR_WIDTH, y + _BAR_HEIGHT), (90, 90, 90), 1, cv2.LINE_AA)
    if filled_w > 0:
        cv2.rectangle(frame, (_BAR_X, y), (_BAR_X + filled_w, y + _BAR_HEIGHT), color, -1, cv2.LINE_AA)

    value_str = f"{value:.2f}{unit}"
    value_x = _BAR_X + _BAR_WIDTH + 12
    cv2.putText(frame, value_str, (value_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, _BAR_FONT_SCALE, (0, 0, 0), _BAR_FONT_THICKNESS + 3, cv2.LINE_AA)
    cv2.putText(frame, value_str, (value_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, _BAR_FONT_SCALE, _COLOR_TEXT, _BAR_FONT_THICKNESS, cv2.LINE_AA)


def draw_overlay(frame, points: list[Point], metrics: dict[str, float]) -> None:
    """Draw the geometric heuristics on top of the (already mirrored) frame.

    Purely cosmetic -- draws over `frame` in place, doesn't feed back into
    any of the compute_* metrics. Meant to make it visually obvious what
    each JSON field is actually measuring.
    """
    # Symmetry axis (forehead top -> chin).
    cv2.line(frame, _pt(points[LM_FOREHEAD_TOP]), _pt(points[LM_CHIN]), _COLOR_AXIS, 1, cv2.LINE_AA)

    # Canthal tilt: inner->outer corner line per eye.
    for inner_idx, outer_idx in (
        (LM_RIGHT_EYE_INNER, LM_RIGHT_EYE_OUTER),
        (LM_LEFT_EYE_INNER, LM_LEFT_EYE_OUTER),
    ):
        cv2.line(frame, _pt(points[inner_idx]), _pt(points[outer_idx]), _COLOR_CANTHAL, 2, cv2.LINE_AA)

    # Gonial angle: temple -> gonion -> chin, per side.
    for temple_idx, gonion_idx in ((LM_TEMPLE_LEFT, LM_JAW_LEFT), (LM_TEMPLE_RIGHT, LM_JAW_RIGHT)):
        cv2.line(frame, _pt(points[temple_idx]), _pt(points[gonion_idx]), _COLOR_GONIAL, 2, cv2.LINE_AA)
        cv2.line(frame, _pt(points[gonion_idx]), _pt(points[LM_CHIN]), _COLOR_GONIAL, 2, cv2.LINE_AA)

    # Bigonial (jaw) width and cheekbone/temple width.
    cv2.line(frame, _pt(points[LM_JAW_LEFT]), _pt(points[LM_JAW_RIGHT]), _COLOR_BIGONIAL, 2, cv2.LINE_AA)
    cv2.line(frame, _pt(points[LM_CHEEK_LEFT]), _pt(points[LM_CHEEK_RIGHT]), _COLOR_CHEEKBONE, 2, cv2.LINE_AA)
    cv2.line(frame, _pt(points[LM_TEMPLE_LEFT]), _pt(points[LM_TEMPLE_RIGHT]), _COLOR_CHEEKBONE, 1, cv2.LINE_AA)

    # Facial thirds: brow-mid -> nose base -> upper lip / lower lip -> chin.
    brow_mid = (
        (points[LM_EYEBROW_LEFT][0] + points[LM_EYEBROW_RIGHT][0]) / 2.0,
        (points[LM_EYEBROW_LEFT][1] + points[LM_EYEBROW_RIGHT][1]) / 2.0,
    )
    for a, b in (
        (brow_mid, points[LM_NOSE_BASE]),
        (points[LM_NOSE_BASE], points[LM_UPPER_LIP]),
        (points[LM_LOWER_LIP], points[LM_CHIN]),
    ):
        cv2.line(frame, _pt(a), _pt(b), _COLOR_THIRDS, 1, cv2.LINE_AA)

    # Profile silhouette: glabella -> nasion -> nose tip -> chin. The
    # facial-convexity and nasofrontal angles are both measured along this
    # one polyline (at the nose-tip and nasion vertices respectively), so
    # it reads as a single connected profile line.
    for a, b in (
        (LM_GLABELLA, LM_NASION),
        (LM_NASION, LM_NOSE_TIP),
        (LM_NOSE_TIP, LM_CHIN),
    ):
        cv2.line(frame, _pt(points[a]), _pt(points[b]), _COLOR_PROFILE, 2, cv2.LINE_AA)

    # Nasolabial angle: nose base -> upper lip.
    cv2.line(frame, _pt(points[LM_NOSE_BASE]), _pt(points[LM_UPPER_LIP]), _COLOR_NASOLABIAL, 2, cv2.LINE_AA)

    # Mirrored pairs used for the symmetry score.
    for left_idx, right_idx in SYMMETRY_PAIRS:
        cv2.circle(frame, _pt(points[left_idx]), 3, _COLOR_AXIS, -1, cv2.LINE_AA)
        cv2.circle(frame, _pt(points[right_idx]), 3, _COLOR_AXIS, -1, cv2.LINE_AA)

    # Bar readout, top-left corner -- one filled progress bar per metric,
    # same min/max scale as the Go TUI's bars (see _BAR_SPECS). Composite
    # score/tier goes last, as bigger text, as the headline result.
    y = 16
    for key, label, lo, hi, unit, color in _BAR_SPECS:
        _draw_bar_row(frame, y, label, metrics[key], lo, hi, unit, color)
        y += _BAR_ROW_HEIGHT

    # The score is only shown when the face is frontal enough (see
    # is_frontal_enough) -- past that point symmetry/gonial/bigonial/
    # cheekbone are all computed on a face pose they weren't designed for,
    # and showing a confident-looking number would just be misleading.
    frontal_ok = is_frontal_enough(metrics)
    if frontal_ok:
        score, tier = compute_composite_score(metrics)
        score_line = f"score: {score:.2f}/{_SCORE_MAX:.1f} -- {tier}"
    else:
        score_line = "score: turn face frontal to rate"
    pose_line = f"pose offset: {metrics['frontality_offset']:.2f} ({'frontal' if frontal_ok else 'turned'})"

    y += 14
    for line, font_scale, thickness in ((pose_line, 0.9, 2), (score_line, 1.3, 3)):
        y += int(38 * font_scale)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _COLOR_TEXT, thickness, cv2.LINE_AA)
        y += 10


class StdinWatcher:
    """Watches stdin in a background thread and signals when it is closed.

    The worker doesn't expect any input, but the parent process closing
    stdin (e.g. on shutdown) should make this process exit cleanly rather
    than keep the camera open indefinitely.
    """

    def __init__(self) -> None:
        self.closed = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _watch(self) -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if line == "":
                    break
        except (OSError, ValueError):
            pass
        finally:
            self.closed.set()


def emit(payload: dict) -> None:
    """Write one JSON object as a line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _stdout_is_terminal() -> bool:
    """True when stdout is a live terminal rather than a pipe/redirect.

    Used to decide whether to skip the JSON-lines stream: when a human runs
    this script directly with the preview window on, printing every frame
    as JSON just scrolls the terminal behind the window for no benefit. When
    the Go core spawns this as a subprocess, stdout is always a pipe, so
    the metrics stream (its actual data contract) is never affected.
    """
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def run(camera_index: int, show_preview: bool = True) -> int:
    """Open the camera, run the detection loop, and stream metrics as JSON lines.

    If show_preview is set, also opens a live OpenCV window with the
    mirrored camera feed and the heuristics drawn on top (see
    draw_overlay). Purely visual -- doesn't affect the JSON stream.

    Returns a process exit code (see the EXIT_* constants).
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        sys.stderr.write(
            f"error: could not open webcam at index {camera_index}. "
            "Check that a camera is connected and not in use by another app.\n"
        )
        return EXIT_CAMERA_UNAVAILABLE

    stop_event = threading.Event()

    def handle_signal(signum, frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    stdin_watcher = StdinWatcher()
    stdin_watcher.start()

    # Skip the JSON-lines stdout stream only when a human is directly
    # watching the preview window from a real terminal -- the Go core
    # always sees a pipe here, so its metrics feed is untouched.
    stream_json = not (show_preview and _stdout_is_terminal())

    mp_face_mesh = mp.solutions.face_mesh
    exit_code = EXIT_OK
    smoother = MetricsSmoother()

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while not stop_event.is_set() and not stdin_watcher.closed.is_set():
                ok, frame = capture.read()
                if not ok:
                    # Transient camera glitch: report no-face and keep trying.
                    if stream_json:
                        emit({"face_detected": False, "ts": time.time()})
                    continue

                # Mirror for a natural selfie-view; done before detection so
                # the preview and the computed geometry always agree on
                # which side of the image is which.
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if not results.multi_face_landmarks:
                    smoother.reset()
                    if stream_json:
                        emit({"face_detected": False, "ts": time.time()})
                    if show_preview:
                        cv2.imshow(PREVIEW_WINDOW_NAME, frame)
                else:
                    landmarks = results.multi_face_landmarks[0].landmark
                    metrics = smoother.update(compute_metrics(landmarks, width, height))
                    if stream_json:
                        emit({"face_detected": True, "ts": time.time(), **metrics})
                    if show_preview:
                        points = _to_pixel_points(landmarks, width, height)
                        draw_overlay(frame, points, metrics)
                        cv2.imshow(PREVIEW_WINDOW_NAME, frame)

                if show_preview:
                    # Pumps the GUI event loop; also lets the user close the
                    # window (X button) or press q/Esc to stop the worker.
                    key = cv2.waitKey(1) & 0xFF
                    window_closed = cv2.getWindowProperty(PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                    if key in (ord("q"), 27) or window_closed:
                        stop_event.set()
    except Exception as exc:  # noqa: BLE001 - surface any failure to the parent
        sys.stderr.write(f"error: face worker crashed: {exc}\n")
        exit_code = EXIT_RUNTIME_ERROR
    finally:
        capture.release()
        if show_preview:
            cv2.destroyAllWindows()

    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the CV worker."""
    parser = argparse.ArgumentParser(description="looksmax-cli face-geometry worker")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index to capture from (default: 0)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="disable the live OpenCV preview window; stream JSON metrics only",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Entry point."""
    args = parse_args()
    return run(args.camera_index, show_preview=not args.no_preview)


if __name__ == "__main__":
    sys.exit(main())
