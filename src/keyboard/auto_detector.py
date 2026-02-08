"""
Automatic Keyboard Detection via Classical Computer Vision

Detects the piano keyboard region from raw video frames using:
    1. Preprocessing  — grayscale, adaptive histogram equalisation, Gaussian blur
    2. Canny edge detection (multiple thresholds, automatic Otsu-based)
    3. Hough line transform — horizontal lines for top/bottom edges
    4. Line clustering (y-coordinate) — merge nearby horizontal lines
    5. Keyboard ROI selection — brightness-validated pair with best aspect ratio
    6. Black-key segmentation — threshold + contour analysis to refine boundaries
    7. Multi-frame consensus — sample N frames and vote for the best detection
    8. IoU evaluation — compare auto-detected bbox with corner-annotation ground truth

References:
    - Akbari & Cheng (2015) — Real-time piano key detection via Hough Transform
    - Moryossef et al. (2023) — Black-key pattern identification for keyboard localisation

Usage:
    from src.keyboard.auto_detector import AutoKeyboardDetector
    detector = AutoKeyboardDetector()
    result = detector.detect_from_video(video_path)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from .detector import KeyboardDetector, KeyboardRegion


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LineCluster:
    """A cluster of similar horizontal lines."""
    y_mean: float
    x_min: float
    x_max: float
    count: int
    lines: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def mean_line_length(self) -> float:
        """Average length of individual lines in this cluster."""
        if not self.lines:
            return 0.0
        lengths = [np.hypot(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in self.lines]
        return float(np.mean(lengths))

    @property
    def max_line_length(self) -> float:
        """Length of the longest individual line in this cluster."""
        if not self.lines:
            return 0.0
        return max(np.hypot(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in self.lines)


@dataclass
class AutoDetectionResult:
    """Full result of automatic keyboard detection."""
    keyboard_region: Optional[KeyboardRegion]
    # Intermediate CV artefacts (for visualisation)
    edges: Optional[np.ndarray] = None
    horizontal_lines: Optional[List[Tuple[int, int, int, int]]] = None
    vertical_lines: Optional[List[Tuple[int, int, int, int]]] = None
    line_clusters: Optional[List[LineCluster]] = None
    top_line_y: Optional[float] = None
    bottom_line_y: Optional[float] = None
    black_key_contours: Optional[List[np.ndarray]] = None
    detection_frame_idx: Optional[int] = None
    iou_vs_corners: Optional[float] = None
    # Multi-frame voting
    per_frame_bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None
    consensus_bbox: Optional[Tuple[int, int, int, int]] = None
    success: bool = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cluster_horizontal_lines(
    lines: List[Tuple[int, int, int, int]],
    y_tolerance: int = 15,
) -> List[LineCluster]:
    """
    Group horizontal lines whose y-coordinates are within *y_tolerance*.
    Returns clusters sorted by mean y.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
    clusters: List[LineCluster] = []
    current: Optional[LineCluster] = None

    for (x1, y1, x2, y2) in sorted_lines:
        y_mid = (y1 + y2) / 2.0
        if current is None or abs(y_mid - current.y_mean) > y_tolerance:
            current = LineCluster(
                y_mean=y_mid,
                x_min=min(x1, x2),
                x_max=max(x1, x2),
                count=1,
                lines=[(x1, y1, x2, y2)],
            )
            clusters.append(current)
        else:
            n = current.count
            current.y_mean = (current.y_mean * n + y_mid) / (n + 1)
            current.x_min = min(current.x_min, x1, x2)
            current.x_max = max(current.x_max, x1, x2)
            current.count += 1
            current.lines.append((x1, y1, x2, y2))

    return clusters


def _compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AutoKeyboardDetector:
    """
    Robust automatic piano-keyboard detector using classical CV.

    This wraps and extends :class:`KeyboardDetector` with:
    * Adaptive preprocessing (CLAHE, Otsu-based Canny)
    * Horizontal-line clustering and intelligent pair selection
    * Brightness-based content validation (white keys are the brightest
      horizontal band in the frame)
    * Black-key contour refinement
    * Multi-frame consensus voting
    * IoU comparison against corner-based ground truth
    """

    # Plausible aspect-ratio range for a keyboard bounding box
    # width / height should be roughly 5–20 for a standard 88-key piano
    MIN_ASPECT = 3.0
    MAX_ASPECT = 25.0
    # Minimum width fraction of frame that a valid keyboard should span
    MIN_WIDTH_FRAC = 0.30

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Canny parameters (will also try automatic Otsu-based)
        self.canny_low = self.config.get("canny_low", 50)
        self.canny_high = self.config.get("canny_high", 150)
        # Hough parameters — use long min line length to filter out texture
        self.hough_threshold = self.config.get("hough_threshold", 100)
        self.min_line_length = self.config.get("min_line_length", 200)
        self.hough_max_gap = self.config.get("hough_max_gap", 20)
        # Clustering
        self.y_cluster_tol = self.config.get("y_cluster_tolerance", 15)
        # Multi-frame
        self.num_sample_frames = self.config.get("num_sample_frames", 7)
        # Black-key detection
        self.black_threshold = self.config.get("black_key_threshold", 70)

        # Reuse base detector for key layout computation
        self._base_detector = KeyboardDetector(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_single_frame(
        self,
        frame: np.ndarray,
        return_intermediates: bool = False,
    ) -> AutoDetectionResult:
        """
        Run the full Canny → Hough → cluster → select pipeline on one frame.

        Args:
            frame: BGR image (H, W, 3)
            return_intermediates: keep intermediate artefacts for visualisation

        Returns:
            AutoDetectionResult
        """
        h, w = frame.shape[:2]
        result = AutoDetectionResult(keyboard_region=None)

        # --- 1. Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE for contrast normalisation (handles varying lighting)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # --- 2. Canny edge detection (auto-threshold via Otsu) ---
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_low = max(10, int(otsu_thresh * 0.5))
        canny_high = min(255, int(otsu_thresh * 1.0))
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Also try fixed thresholds and merge
        edges_fixed = cv2.Canny(blurred, self.canny_low, self.canny_high)
        edges = cv2.bitwise_or(edges, edges_fixed)

        # Morphological close with horizontal kernel to connect fragmented
        # keyboard-edge segments while suppressing vertical noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if return_intermediates:
            result.edges = edges.copy()

        # --- 3. Hough Line Transform (strict parameters) ---
        raw_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.hough_max_gap,
        )

        if raw_lines is None or len(raw_lines) == 0:
            return result

        # Separate horizontal and vertical lines
        horiz: List[Tuple[int, int, int, int]] = []
        vert: List[Tuple[int, int, int, int]] = []
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < np.pi / 12:          # < 15° from horizontal
                horiz.append((x1, y1, x2, y2))
            elif angle > np.pi / 2 - np.pi / 12:   # < 15° from vertical
                vert.append((x1, y1, x2, y2))

        if return_intermediates:
            result.horizontal_lines = horiz
            result.vertical_lines = vert

        if len(horiz) < 2:
            return result

        # --- 4. Cluster horizontal lines ---
        clusters = _cluster_horizontal_lines(horiz, self.y_cluster_tol)
        if return_intermediates:
            result.line_clusters = clusters

        if len(clusters) < 2:
            return result

        # --- 5. Select top/bottom pair (brightness-validated) ---
        best_pair = self._select_keyboard_pair(clusters, gray, w, h)
        if best_pair is None:
            return result

        top_cluster, bot_cluster = best_pair
        x_min = int(min(top_cluster.x_min, bot_cluster.x_min))
        x_max = int(max(top_cluster.x_max, bot_cluster.x_max))
        y_top = int(top_cluster.y_mean)
        y_bot = int(bot_cluster.y_mean)

        # Sanity checks
        if y_bot - y_top < 20 or x_max - x_min < 100:
            return result

        if return_intermediates:
            result.top_line_y = float(y_top)
            result.bottom_line_y = float(y_bot)

        # --- 6. Black-key refinement (optional) ---
        bbox = (x_min, y_top, x_max, y_bot)
        refined_bbox, black_contours = self._refine_with_black_keys(frame, bbox)
        if refined_bbox is not None:
            bbox = refined_bbox

        if return_intermediates:
            result.black_key_contours = black_contours

        result.consensus_bbox = bbox

        # --- 7. Build KeyboardRegion ---
        kb_region = self._bbox_to_keyboard_region(bbox)
        result.keyboard_region = kb_region
        result.success = True

        return result

    def detect_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> AutoDetectionResult:
        """
        Multi-frame consensus detection.

        Samples *num_sample_frames* frames evenly from the video,
        runs single-frame detection on each, and picks the best via voting.

        Args:
            video_path: path to the video file
            max_frames: override total frame count (for speed)
            return_intermediates: keep artefacts from the best frame

        Returns:
            AutoDetectionResult with consensus bounding box
        """
        from ..data.video_utils import VideoProcessor

        vp = VideoProcessor()
        vp.open(video_path)
        total = int(vp.info.frame_count)
        if max_frames:
            total = min(total, max_frames)

        # Pick sample indices spread evenly (skip first/last 5 %)
        margin = max(1, total // 20)
        indices = np.linspace(margin, total - margin, self.num_sample_frames, dtype=int).tolist()

        bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        best_result: Optional[AutoDetectionResult] = None
        best_area = 0

        for idx in indices:
            frame = vp.get_frame(idx)
            if frame is None:
                bboxes.append(None)
                continue

            res = self.detect_single_frame(frame, return_intermediates=return_intermediates)
            bb = res.consensus_bbox
            bboxes.append(bb)

            if bb is not None:
                area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                if area > best_area:
                    best_area = area
                    best_result = res
                    best_result.detection_frame_idx = idx

        vp.close()

        if best_result is None:
            return AutoDetectionResult(keyboard_region=None, per_frame_bboxes=bboxes)

        # Consensus: median of all valid bboxes
        valid = [b for b in bboxes if b is not None]
        if len(valid) >= 3:
            arr = np.array(valid)
            med = np.median(arr, axis=0).astype(int)
            consensus = (int(med[0]), int(med[1]), int(med[2]), int(med[3]))
            best_result.consensus_bbox = consensus
            best_result.keyboard_region = self._bbox_to_keyboard_region(consensus)

        best_result.per_frame_bboxes = bboxes
        return best_result

    def evaluate_against_corners(
        self,
        auto_result: AutoDetectionResult,
        corners: Dict[str, str],
    ) -> float:
        """
        Compute IoU between the auto-detected bbox and the corner-annotation bbox.

        Args:
            auto_result: result from auto-detection
            corners: PianoVAM corner annotations

        Returns:
            IoU score in [0, 1]
        """
        if auto_result.consensus_bbox is None:
            return 0.0

        corner_region = self._base_detector.detect_from_corners(corners)
        iou = _compute_iou(auto_result.consensus_bbox, corner_region.bbox)
        auto_result.iou_vs_corners = iou
        return iou

    # ------------------------------------------------------------------
    # Visualisation helpers (called from notebook)
    # ------------------------------------------------------------------

    def visualize_edges(self, frame: np.ndarray, result: AutoDetectionResult) -> np.ndarray:
        """Return a side-by-side of original frame and edge map."""
        if result.edges is None:
            return frame
        edges_bgr = cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR)
        return np.hstack([frame, edges_bgr])

    def visualize_lines(
        self,
        frame: np.ndarray,
        result: AutoDetectionResult,
    ) -> np.ndarray:
        """Draw detected lines on the frame."""
        vis = frame.copy()
        if result.horizontal_lines:
            for (x1, y1, x2, y2) in result.horizontal_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if result.vertical_lines:
            for (x1, y1, x2, y2) in result.vertical_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        return vis

    def visualize_clusters(
        self,
        frame: np.ndarray,
        result: AutoDetectionResult,
    ) -> np.ndarray:
        """Draw line clusters and selected top/bottom lines."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        if result.line_clusters:
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
            ]
            for i, cl in enumerate(result.line_clusters):
                color = colors[i % len(colors)]
                y = int(cl.y_mean)
                cv2.line(vis, (int(cl.x_min), y), (int(cl.x_max), y), color, 2)
                cv2.putText(vis, f"C{i} n={cl.count}", (int(cl.x_min), y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if result.top_line_y is not None:
            cv2.line(vis, (0, int(result.top_line_y)), (w, int(result.top_line_y)), (0, 255, 255), 3)
        if result.bottom_line_y is not None:
            cv2.line(vis, (0, int(result.bottom_line_y)), (w, int(result.bottom_line_y)), (0, 255, 255), 3)

        return vis

    def visualize_detection(
        self,
        frame: np.ndarray,
        result: AutoDetectionResult,
        corner_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Draw the final auto-detected bbox (green) and optionally the GT bbox (red)."""
        vis = frame.copy()
        if result.consensus_bbox is not None:
            x1, y1, x2, y2 = result.consensus_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis, "Auto-detected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if corner_bbox is not None:
            cx1, cy1, cx2, cy2 = corner_bbox
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
            cv2.putText(vis, "Corner GT", (cx1, cy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if result.iou_vs_corners is not None:
            cv2.putText(vis, f"IoU = {result.iou_vs_corners:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        return vis

    def visualize_black_keys(
        self,
        frame: np.ndarray,
        result: AutoDetectionResult,
    ) -> np.ndarray:
        """Draw detected black key contours."""
        vis = frame.copy()
        if result.black_key_contours:
            cv2.drawContours(vis, result.black_key_contours, -1, (0, 255, 255), 2)
            cv2.putText(vis, f"{len(result.black_key_contours)} black keys found",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return vis

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_keyboard_pair(
        self,
        clusters: List[LineCluster],
        gray: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Tuple[LineCluster, LineCluster]]:
        """
        Among all pairs of clusters, select the one most likely to be the
        keyboard's top and bottom edges.

        Selection criteria:
            1. aspect ratio within expected range (wide and thin)
            2. both clusters span a large fraction of the frame
            3. the ROI between them is BRIGHT (white keys are the brightest
               horizontal band in a piano video)
            4. the ROI has high column-wise variance (alternating white/black)
            5. weighted by mean individual line length (keyboard edges produce
               long, continuous lines; texture produces many short lines)
        """
        best_score = -1.0
        best_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                top, bot = clusters[i], clusters[j]
                if top.y_mean > bot.y_mean:
                    top, bot = bot, top

                height = bot.y_mean - top.y_mean
                if height < 30:
                    continue

                x_min = min(top.x_min, bot.x_min)
                x_max = max(top.x_max, bot.x_max)
                width = x_max - x_min

                if width < 100:
                    continue

                aspect = width / height
                if not (self.MIN_ASPECT <= aspect <= self.MAX_ASPECT):
                    continue

                width_frac = width / frame_w
                if width_frac < self.MIN_WIDTH_FRAC:
                    continue

                # ---- Brightness validation ----
                # The keyboard region (white keys) is distinctively bright.
                # Compute mean brightness of the ROI between the two lines.
                y1 = max(0, int(top.y_mean))
                y2 = min(frame_h, int(bot.y_mean))
                x1 = max(0, int(x_min))
                x2 = min(frame_w, int(x_max))
                roi = gray[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                mean_brightness = float(np.mean(roi))
                # Column-wise std captures alternating white/black pattern
                col_profile = np.mean(roi, axis=0)
                col_std = float(np.std(col_profile))

                # White keys have brightness > 150 typically;
                # use relative brightness compared to frame
                frame_mean = float(np.mean(gray))
                brightness_ratio = mean_brightness / max(frame_mean, 1.0)

                # ---- Line quality ----
                # Prefer clusters with long individual lines (structural edges)
                # over clusters with many short lines (texture noise)
                mean_len_top = top.mean_line_length / frame_w
                mean_len_bot = bot.mean_line_length / frame_w
                line_quality = (mean_len_top + mean_len_bot) / 2.0

                # ---- Combined score ----
                # brightness_ratio > 1.5 for keyboard vs background
                # col_std > 30 for alternating white/black keys
                # line_quality high for true structural edges
                score = (
                    width_frac
                    * line_quality
                    * brightness_ratio
                    * (1.0 + col_std / 100.0)  # bonus for stripe pattern
                )

                if score > best_score:
                    best_score = score
                    best_pair = (top, bot)

        return best_pair

    def _validate_keyboard_content(
        self,
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """
        Check whether the ROI between the selected lines actually looks
        like a keyboard (bright white keys, high column-wise variance).
        """
        x1, y1, x2, y2 = bbox
        roi = gray[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return False

        mean_b = float(np.mean(roi))
        max_b = float(np.max(roi))
        col_profile = np.mean(roi, axis=0)
        col_std = float(np.std(col_profile))

        # White keys should push mean brightness well above background
        # and produce high column-wise variation
        return mean_b > 100 and max_b > 200 and col_std > 15

    def _refine_with_black_keys(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        Within the candidate bbox, look for the characteristic pattern
        of black piano keys (dark, vertically elongated rectangles).
        Use them to tighten the x-boundaries.
        """
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return None, []

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_roi, self.black_threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological clean-up
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_h = y2 - y1
        roi_w = x2 - x1

        # Expected black key size relative to the ROI
        min_bk_h = roi_h * 0.25
        max_bk_h = roi_h * 0.85
        min_bk_w = roi_w * 0.005
        max_bk_w = roi_w * 0.04

        candidates = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh < min_bk_h or bh > max_bk_h:
                continue
            if bw < min_bk_w or bw > max_bk_w:
                continue
            aspect = bh / bw if bw > 0 else 0
            if aspect < 1.5:
                continue
            # Black keys start near the top of the ROI
            if by > roi_h * 0.3:
                continue
            candidates.append(cnt)

        if len(candidates) < 5:
            return None, candidates

        # Tighten bbox using the black key extents
        all_x = []
        for cnt in candidates:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            all_x.extend([bx + x1, bx + bw + x1])

        new_x1 = max(0, min(all_x) - 30)  # small padding
        new_x2 = max(all_x) + 30

        # Offset contours to frame coordinates for visualisation
        shifted = [cnt + np.array([[[x1, y1]]]) for cnt in candidates]
        return (new_x1, y1, new_x2, y2), shifted

    def _bbox_to_keyboard_region(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> KeyboardRegion:
        """Convert a bounding box to a full KeyboardRegion with 88 key layout."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Build homography  (identity-like from bbox to normalised rect)
        src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        key_boundaries = self._base_detector._compute_key_boundaries_from_width(
            width, height, offset=(x1, y1)
        )

        return KeyboardRegion(
            bbox=bbox,
            homography=H,
            key_boundaries=key_boundaries,
            white_key_width=width / 52,
            black_key_width=(width / 52) * 0.6,
            corners={
                "LT": (x1, y1), "RT": (x2, y1),
                "RB": (x2, y2), "LB": (x1, y2),
            },
        )
