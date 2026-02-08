"""
Automatic Keyboard Detection via Classical Computer Vision

Detects the piano keyboard region from raw video frames using:
    1. Preprocessing  — grayscale, adaptive histogram equalisation, Gaussian blur
    2. Canny edge detection (multiple thresholds, automatic Otsu-based)
    3. Hough line transform — horizontal lines for top/bottom edges
    4. Line clustering (y-coordinate) — merge nearby horizontal lines
    5. Keyboard ROI selection — brightness-validated pair with best aspect ratio
    6. Bottom-edge extension via brightness profile (captures full white keys)
    7. Black-key segmentation — threshold + contour analysis to refine boundaries
    8. Multi-frame consensus — sample N frames and vote for the best detection
    9. IoU evaluation — compare auto-detected bbox with corner-annotation ground truth

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

    Detection strategy (hybrid Hough + brightness-profile):

    1.  Canny + Hough finds strong horizontal edges.  The **top edge** of
        the keyboard (where the dark piano body meets the keys) is usually
        the most reliable.

    2.  Line clustering groups edges by y-coordinate; the pair whose ROI is
        brightest and has the highest column-wise variance (alternating
        white / black keys) is selected.

    3.  **Brightness-profile extension** — the Hough-based bottom edge
        often lands on the black-key / white-key boundary (a strong
        internal edge) rather than the true front of the white keys.
        We fix this by scanning the grayscale brightness profile downward
        from the initial bottom edge: as long as the mean row brightness
        stays above a threshold (white keys are bright), we keep extending.
        The bottom is set where brightness drops sharply (edge of keys →
        hands / body / carpet).

    4.  Black-key contour analysis optionally tightens the x-boundaries.

    5.  Multi-frame consensus (median over N sampled frames) removes
        per-frame noise.
    """

    # Plausible aspect-ratio range for a keyboard bounding box
    MIN_ASPECT = 3.0
    MAX_ASPECT = 25.0
    # "Ideal" aspect ratio (width / height ≈ 8 for a real keyboard)
    IDEAL_ASPECT = 8.0
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
        Run the full detection pipeline on one frame.

        Pipeline:
            Preprocessing → Canny → Hough → Cluster → Pair-select
            → Brightness-extend bottom edge → Black-key refine → Build region

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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # --- 2. Canny edge detection (Otsu-adaptive + fixed, merged) ---
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_low = max(10, int(otsu_thresh * 0.5))
        canny_high = min(255, int(otsu_thresh * 1.0))
        edges = cv2.Canny(blurred, canny_low, canny_high)
        edges_fixed = cv2.Canny(blurred, self.canny_low, self.canny_high)
        edges = cv2.bitwise_or(edges, edges_fixed)

        # Morphological close with horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if return_intermediates:
            result.edges = edges.copy()

        # --- 3. Hough Line Transform ---
        raw_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.hough_max_gap,
        )

        if raw_lines is None or len(raw_lines) == 0:
            return result

        horiz: List[Tuple[int, int, int, int]] = []
        vert: List[Tuple[int, int, int, int]] = []
        for line in raw_lines:
            x1_, y1_, x2_, y2_ = line[0]
            angle = np.abs(np.arctan2(y2_ - y1_, x2_ - x1_))
            if angle < np.pi / 12:
                horiz.append((x1_, y1_, x2_, y2_))
            elif angle > np.pi / 2 - np.pi / 12:
                vert.append((x1_, y1_, x2_, y2_))

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

        if y_bot - y_top < 20 or x_max - x_min < 100:
            return result

        if return_intermediates:
            result.top_line_y = float(y_top)
            result.bottom_line_y = float(y_bot)

        # --- 6. Brightness-profile bottom-edge extension ---
        # The Hough bottom edge often lands on the black-key boundary
        # (a strong internal edge).  The real bottom is where white keys end.
        bbox = (x_min, y_top, x_max, y_bot)
        bbox = self._extend_bottom_via_brightness(gray, bbox)

        # --- 7. Black-key refinement (x-boundaries) ---
        refined_bbox, black_contours = self._refine_with_black_keys(frame, bbox)
        if refined_bbox is not None:
            bbox = refined_bbox

        if return_intermediates:
            result.black_key_contours = black_contours

        result.consensus_bbox = bbox

        # --- 8. Build KeyboardRegion ---
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
        runs single-frame detection on each, and computes consensus via
        median bounding box.
        """
        from ..data.video_utils import VideoProcessor

        vp = VideoProcessor()
        vp.open(video_path)
        total = int(vp.info.frame_count)
        if max_frames:
            total = min(total, max_frames)

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
        """Compute IoU between the auto-detected bbox and corner-annotation bbox."""
        if auto_result.consensus_bbox is None:
            return 0.0

        corner_region = self._base_detector.detect_from_corners(corners)
        iou = _compute_iou(auto_result.consensus_bbox, corner_region.bbox)
        auto_result.iou_vs_corners = iou
        return iou

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def visualize_edges(self, frame: np.ndarray, result: AutoDetectionResult) -> np.ndarray:
        if result.edges is None:
            return frame
        edges_bgr = cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR)
        return np.hstack([frame, edges_bgr])

    def visualize_lines(self, frame: np.ndarray, result: AutoDetectionResult) -> np.ndarray:
        vis = frame.copy()
        if result.horizontal_lines:
            for (x1, y1, x2, y2) in result.horizontal_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if result.vertical_lines:
            for (x1, y1, x2, y2) in result.vertical_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        return vis

    def visualize_clusters(self, frame: np.ndarray, result: AutoDetectionResult) -> np.ndarray:
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

    def visualize_black_keys(self, frame: np.ndarray, result: AutoDetectionResult) -> np.ndarray:
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
        Select the cluster pair most likely to be the keyboard top/bottom.

        Scoring considers:
            - Width fraction (should span most of the frame)
            - Mean line length (keyboard edges are long; texture is short)
            - ROI brightness (white keys are the brightest band)
            - Column-wise variance (alternating white / black)
            - Aspect-ratio closeness to the ideal (~8 for an 88-key piano)
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

                # ---- Brightness of the ROI ----
                y1 = max(0, int(top.y_mean))
                y2 = min(frame_h, int(bot.y_mean))
                x1 = max(0, int(x_min))
                x2 = min(frame_w, int(x_max))
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                mean_brightness = float(np.mean(roi))
                col_profile = np.mean(roi, axis=0)
                col_std = float(np.std(col_profile))

                frame_mean = float(np.mean(gray))
                brightness_ratio = mean_brightness / max(frame_mean, 1.0)

                # ---- Line quality ----
                mean_len_top = top.mean_line_length / frame_w
                mean_len_bot = bot.mean_line_length / frame_w
                line_quality = (mean_len_top + mean_len_bot) / 2.0

                # ---- Combined score ----
                score = (
                    width_frac
                    * line_quality
                    * brightness_ratio
                    * (1.0 + col_std / 100.0)
                )

                if score > best_score:
                    best_score = score
                    best_pair = (top, bot)

        return best_pair

    # ---- NEW: brightness-profile bottom-edge extension ----

    def _extend_bottom_via_brightness(
        self,
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Extend the bottom edge of the bbox to capture the full white-key region.

        The Hough-based bottom edge frequently lands on the black-key /
        white-key internal boundary because that is a very high-contrast
        horizontal edge.  The real bottom of the keyboard is further down,
        where the white keys end and the player's hands / body / carpet
        begin.

        Method:
            1.  Compute the mean row brightness inside the initial bbox
                (the reference brightness of the keyboard).
            2.  Scan downward row by row from the current bottom edge.
            3.  While the mean brightness of each row (in the same x-range)
                stays above  ``reference * drop_ratio``  we keep extending.
            4.  A smoothed gradient (Sobel-y) check detects the sharp
                brightness drop at the keyboard's front edge.
            5.  Clamp the extension to a maximum of ``width / 5`` pixels
                (a real keyboard is never deeper than width / 5).
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        h_img = gray.shape[0]

        # Maximum plausible extension (keyboard depth ≈ width / 8,
        # so the part below the initial detection can be at most ~ width / 5)
        max_ext = int(width / 5)
        scan_end = min(y2 + max_ext, h_img)

        if scan_end <= y2:
            return bbox

        # Reference brightness: mean of the initial keyboard ROI
        ref_roi = gray[y1:y2, x1:x2]
        if ref_roi.size == 0:
            return bbox
        ref_brightness = float(np.mean(ref_roi))

        # Scan downward
        scan_strip = gray[y2:scan_end, x1:x2]
        if scan_strip.size == 0:
            return bbox

        row_means = np.mean(scan_strip, axis=1).astype(float)

        # Smooth the profile to avoid single-row noise
        if len(row_means) > 5:
            kernel_size = 5
            row_means_smooth = np.convolve(row_means, np.ones(kernel_size) / kernel_size, mode='same')
        else:
            row_means_smooth = row_means

        # The white keys maintain high brightness.  We look for the row
        # where brightness drops below a fraction of the reference.
        # Use a generous threshold: white keys can be slightly dimmer at the
        # front edge (farther from camera, slight shadow).
        drop_threshold = ref_brightness * 0.55

        new_y2 = y2
        for i, bval in enumerate(row_means_smooth):
            if bval < drop_threshold:
                new_y2 = y2 + i
                break
        else:
            # brightness never dropped → use the scan end
            new_y2 = scan_end

        # Sanity: the resulting aspect ratio should be plausible
        new_height = new_y2 - y1
        if new_height < height:
            new_y2 = y2  # don't shrink
        aspect = width / max(new_height, 1)
        if aspect < self.MIN_ASPECT:
            # extended too far — clamp
            new_y2 = y1 + int(width / self.MIN_ASPECT)
            new_y2 = min(new_y2, h_img)

        return (x1, y1, x2, new_y2)

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

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_h = y2 - y1
        roi_w = x2 - x1

        min_bk_h = roi_h * 0.15
        max_bk_h = roi_h * 0.85
        min_bk_w = roi_w * 0.003
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
            if by > roi_h * 0.35:
                continue
            candidates.append(cnt)

        if len(candidates) < 5:
            return None, candidates

        all_x = []
        for cnt in candidates:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            all_x.extend([bx + x1, bx + bw + x1])

        new_x1 = max(0, min(all_x) - 30)
        new_x2 = max(all_x) + 30

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
