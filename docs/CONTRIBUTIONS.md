# Project Contributions: Beyond Moryossef et al. (2023)

## What is the Reference Paper About?

**Moryossef et al. (2023)** — *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"*

### Core Contribution of the Paper

The paper introduces a **video-based pipeline** for automatically detecting piano fingering from top-down camera recordings. Their key innovation is the **Gaussian probability model using x-distance only** for finger-key assignment.

**Key Insight**: In a top-down camera view, the y-axis measures depth into the keyboard, which varies systematically with finger length. The thumb (shortest finger) reaches less far into the keyboard than the middle finger. If both x and y distances are used, this introduces a systematic bias. **Using x-distance only eliminates this anatomical confound.**

### Paper's Methodology

1. **Keyboard Detection**: Uses computer vision (edge/line analysis) to detect keyboard from video
2. **Hand Pose Estimation**: MediaPipe 21-keypoint hand detection
3. **Finger-Key Assignment**: Gaussian probability model `P(finger→key) = exp(−dx²/2σ²)` where `dx` is **only the horizontal distance**
4. **Both-hands evaluation**: Try both left and right hands for each note, pick the higher confidence
5. **Max-distance gate**: Reject assignments when the hand is too far from the key

---

## How We Utilized the Paper

We **faithfully implemented** the core methodology from Moryossef et al. (2023):

| Paper Component | Our Implementation |
|----------------|-------------------|
| Gaussian x-only probability | ✅ Exact same formula: `P(finger→key) = exp(−dx²/2σ²)` |
| Auto-scaled σ | ✅ σ scales to mean white-key width |
| Max-distance gate (4σ) | ✅ Same 4σ rejection threshold |
| Both-hands evaluation | ✅ Try L & R, pick higher confidence |
| Temporal smoothing | ✅ Hampel + interpolation + Savitzky-Golay |

**The core finger assignment algorithm is a direct implementation of their methodology.**

---

## Our Contributions Beyond the Paper

### 1. **Enhanced Automatic Keyboard Detection Pipeline**

**What the paper does**: Mentions keyboard detection via edge/line analysis but doesn't detail the specific algorithm.

**What we contributed**:

#### a) **Brightness-Validated Pair Selection**
- **Problem**: Hough line detection finds many horizontal lines (keyboard edges, but also carpet patterns, piano body reflections, etc.)
- **Our solution**: Score candidate keyboard pairs not just by line count and width, but by:
  - **Mean brightness of the ROI** (white keys are the brightest horizontal band in the frame)
  - **Column-wise variance** (alternating white/black keys produce high variance)
  - **Mean individual line length** (keyboard edges produce long structural lines; texture produces many short lines)
- **Result**: Robustly selects the actual keyboard region even when there are many competing horizontal edges

#### b) **Brightness-Profile Bottom-Edge Extension**
- **Problem**: Hough-based bottom edge detection often lands on the **black-key/white-key internal boundary** (a very strong horizontal edge) instead of the true bottom of the keyboard, capturing only ~40% of the keyboard depth.
- **Our solution**: After initial Hough detection, scan downward row-by-row from the detected bottom edge. As long as the mean row brightness stays above 55% of the reference (white keys maintain high brightness), keep extending. Stop when brightness drops sharply (edge of keys → hands/carpet).
- **Result**: Captures the full white-key region, not just the black-key strip

#### c) **Multi-Frame Consensus with Median Voting**
- **Problem**: Single-frame detection can fail due to temporary occlusions (hands covering keyboard, page turns, etc.)
- **Our solution**: Sample N frames evenly across the video, run detection on each, compute the **median bounding box** across all successful detections
- **Result**: Robust against per-frame failures

#### d) **Dual-Threshold Canny with Otsu Adaptation**
- **Problem**: Fixed Canny thresholds don't adapt to varying lighting conditions
- **Our solution**: Compute Otsu threshold on CLAHE-enhanced image, derive adaptive Canny thresholds, merge with fixed thresholds via bitwise OR
- **Result**: Works across varying lighting conditions

#### e) **Black-Key Contour Refinement**
- **Problem**: Initial Hough detection may have loose x-boundaries
- **Our solution**: Within the candidate ROI, detect black keys via thresholding + contour analysis, use their extents to tighten the horizontal boundaries
- **Result**: More precise keyboard localization

**Technical Details**:
- CLAHE (Contrast-Limited Adaptive Histogram Equalisation) for lighting normalization
- Morphological closing with horizontal kernel (25×1) to connect fragmented edges
- Minimum line length filtering (200px) to eliminate texture noise
- Aspect ratio validation (3–25) and width fraction filtering (≥30% of frame)

### 2. **Live MediaPipe Detection (No Pre-Extracted Skeletons)**

**What the paper does**: Uses MediaPipe for hand pose estimation (likely on individual frames or pre-extracted data).

**What we contributed**:
- **Video-mode detection** (`static_image_mode=False`) with temporal tracking across consecutive frames
- **Optimized parameters** for piano performance:
  - `model_complexity=1` (full model for higher accuracy)
  - `min_detection_confidence=0.3` (lower threshold catches partially-occluded/fast-moving hands)
  - `min_tracking_confidence=0.3` (maintains tracking through motion blur)
- **Frame stride** option for speed (process every Nth frame)
- **Complete independence from pre-extracted skeleton data** — the pipeline runs entirely on raw video

**Result**: More robust hand detection, especially for fast-moving hands and partial occlusions common in piano performance.

### 3. **Full Computer Vision Approach (Zero Annotation Dependency)**

**What the paper does**: Uses computer vision for keyboard detection, but may rely on some manual annotations or pre-processed data.

**What we contributed**:
- **Zero manual annotations used during detection**: The entire pipeline operates on raw video + MIDI
- **Corner annotations used ONLY for evaluation**: PianoVAM's keyboard corner annotations serve solely as ground truth for computing IoU (Intersection-over-Union) to evaluate detection quality
- **No pre-extracted skeletons**: Hand landmarks are detected live from video frames
- **End-to-end CV pipeline**: From raw video → keyboard detection → hand detection → finger assignment → refinement

**Result**: A truly automatic system that doesn't require any manual annotation or pre-processing, making it applicable to any piano video with synchronized MIDI.

### 4. **Neural Refinement with BiLSTM + Constrained Viterbi Decoding**

**What the paper does**: Focuses on the baseline Gaussian assignment. Doesn't include neural refinement.

**What we contributed**:
- **BiLSTM model with self-attention** for temporal sequence modeling
- **Constrained Viterbi decoding** with biomechanical constraints:
  - Maximum finger stretch limits (in semitones)
  - Same-finger repetition penalization
  - Finger ordering rules (ascending/descending passages)
  - Thumb crossing rules from piano pedagogy
- **Feature engineering**: 20-dimensional input per note (pitch, one-hot finger, time delta, hand, pitch class)

**Note**: This is inspired by Ramoneda et al. (2022) but adapted for our video-based pipeline.

### 5. **Comprehensive Evaluation Framework**

**What the paper does**: Evaluates finger assignment accuracy (though details may vary).

**What we contributed**:
- **Keyboard detection IoU**: Measures quality of automatic keyboard detection against ground-truth corner annotations
- **Irrational Fingering Rate (IFR)**: Measures biomechanical violation rate (no ground-truth finger labels needed)
- **Framework for Accuracy/M_gen/M_high**: Fully implemented, ready for when ground-truth finger labels become available
- **Visualization tools**: Comprehensive intermediate visualizations (edges, Hough lines, clusters, black-key contours, IoU comparisons)

### 6. **Robust Implementation and Engineering**

**What we contributed**:
- **Modular architecture**: Clean separation of concerns (keyboard, hand, assignment, refinement modules)
- **Configuration system**: YAML-based configs for easy parameter tuning
- **Error handling**: Graceful degradation when detection fails
- **Caching system**: Efficient processing of multiple samples
- **Documentation**: Comprehensive docs, inline comments, and report

---

## Answer to "What is Your Contribution?"

### Short Answer (30 seconds)

> "We implemented and extended the methodology from Moryossef et al. (2023) for automatic piano fingering detection. Our main contributions are: (1) a robust automatic keyboard detection pipeline using brightness-validated scoring and brightness-profile edge extension that works without any manual annotations, (2) live MediaPipe hand detection optimized for piano performance with video-mode temporal tracking, and (3) a full end-to-end computer vision system that operates entirely on raw video with zero dependency on pre-extracted data or manual annotations."

### Detailed Answer (2-3 minutes)

> "Our project implements the core methodology from Moryossef et al. (2023), specifically their Gaussian x-only finger-key assignment model, which elegantly solves the finger-length bias problem in top-down camera views. However, we made several significant contributions beyond the paper:
>
> **First, we developed a robust automatic keyboard detection pipeline** that goes beyond what the paper describes. The paper mentions keyboard detection via edge/line analysis, but doesn't detail the algorithm. We implemented a multi-stage pipeline with:
> - Brightness-validated pair selection that uses the fact that white keys are the brightest horizontal band in the frame
> - A brightness-profile extension method that fixes the common problem where Hough detection lands on the black-key boundary instead of the true keyboard bottom
> - Multi-frame consensus with median voting for robustness against occlusions
> - Dual-threshold Canny with Otsu adaptation for varying lighting conditions
>
> **Second, we implemented live MediaPipe detection** optimized for piano performance, using video-mode temporal tracking and lower confidence thresholds to catch partially-occluded hands. This eliminates the need for pre-extracted skeleton data.
>
> **Third, we built a truly full-CV system** with zero dependency on manual annotations during detection. Corner annotations are used only for evaluation (IoU), not for detection itself.
>
> **Finally, we added neural refinement** with BiLSTM and constrained Viterbi decoding to improve temporal consistency, though this is inspired by other work.
>
> The result is a complete, end-to-end system that can process any piano video with synchronized MIDI, demonstrating that classical computer vision techniques can reliably solve this problem without manual intervention."

### Key Points to Emphasize

1. **We didn't just replicate the paper** — we extended it with novel techniques (brightness validation, brightness-profile extension)
2. **Full-CV approach** — zero annotation dependency is a significant engineering contribution
3. **Robustness** — multi-frame consensus, adaptive thresholds, video-mode tracking all improve reliability
4. **Complete system** — end-to-end pipeline from raw video to finger assignments
5. **Evaluation** — IoU for keyboard detection, IFR for assignment quality

---

## Comparison Table

| Aspect | Moryossef et al. (2023) | Our Implementation |
|--------|------------------------|-------------------|
| **Core assignment algorithm** | Gaussian x-only | ✅ Same (faithful implementation) |
| **Keyboard detection** | Mentions edge/line analysis | ✅ Detailed 9-step pipeline with brightness validation |
| **Hand detection** | MediaPipe (details unclear) | ✅ Live detection with video-mode tracking, optimized params |
| **Annotation dependency** | Unclear | ✅ Zero annotations used during detection |
| **Multi-frame robustness** | Not mentioned | ✅ Median consensus across N frames |
| **Neural refinement** | Not included | ✅ BiLSTM + Viterbi decoding |
| **Evaluation** | Accuracy metrics | ✅ IoU + IFR + framework for accuracy |
| **Bottom-edge detection** | Not addressed | ✅ Brightness-profile extension (novel) |

---

## Summary

**What we took from the paper**: The core Gaussian x-only finger assignment methodology — this is the paper's key contribution and we implemented it faithfully.

**What we added**:
1. **Novel keyboard detection techniques** (brightness validation, brightness-profile extension)
2. **Live hand detection** optimized for piano performance
3. **Full-CV independence** (zero annotation dependency)
4. **Robustness improvements** (multi-frame consensus, adaptive thresholds)
5. **Neural refinement** (BiLSTM + Viterbi)
6. **Comprehensive evaluation** (IoU, IFR, visualization)

**Bottom line**: We implemented the paper's core methodology and extended it with significant engineering contributions and novel techniques, particularly in keyboard detection and the full-CV approach.
