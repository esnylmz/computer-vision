# Pipeline Architecture

## Overview

The pipeline detects piano fingering from video in four stages. Each stage has a corresponding `src/` module and is demonstrated in `notebooks/piano_fingering_detection.ipynb`.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     PIANO FINGERING DETECTION PIPELINE                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │   STAGE 1    │   │   STAGE 2    │   │   STAGE 3    │   │   STAGE 4   │  │
│  │   Keyboard   │──▶│    Hand      │──▶│  Finger-Key  │──▶│   Neural    │  │
│  │  Detection   │   │  Processing  │   │  Assignment  │   │ Refinement  │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘  │
│        │                  │                   │                  │           │
│        ▼                  ▼                   ▼                  ▼           │
│   88 key bboxes      Filtered          Gaussian prob.      Refined          │
│   (pixel space)      landmarks         assignments         predictions      │
│                    (T × 21 × 3)                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Keyboard Detection (Automatic — No Annotations)

**Module**: `src/keyboard/` — primary class: `AutoKeyboardDetector` in `auto_detector.py`

**Goal**: Automatically detect the piano keyboard from raw video frames and map 88 keys to pixel-space bounding boxes **using only computer vision**. No dataset annotations are used in the detection pipeline — this is the full-CV approach.

### Automatic Detection Pipeline (Canny + Hough + Clustering)

**Process** (`AutoKeyboardDetector.detect_single_frame`):

1. **Preprocessing** — Convert to grayscale → CLAHE contrast enhancement → Gaussian blur (5×5)
2. **Canny edge detection** — Two passes: Otsu-adaptive thresholds + fixed thresholds (50/150), results merged with bitwise OR
3. **Morphological closing** — Horizontal structuring element (15×1) to connect fragmented edge segments along the keyboard
4. **Hough line transform** — `cv2.HoughLinesP` to detect line segments; classified as horizontal (<15°) or vertical (>75°)
5. **Line clustering** — Horizontal lines grouped by y-coordinate within a tolerance (15 px). Each cluster aggregates its member lines' x-extent and count.
6. **Keyboard pair selection** — Among all cluster pairs, select the one that:
   - Has a plausible aspect ratio (width/height between 3 and 25)
   - Spans at least 25% of the frame width
   - Maximises `width_fraction × evidence_count`
7. **Black-key refinement** — Within the candidate ROI: threshold to binary → morphological open/close → find contours → filter by aspect ratio, size, and position → tighten x-boundaries using the black-key extents
8. **88-key layout** — Divide detected bounding box into 52 white + 36 black keys directly in pixel space (no inverse homography needed)

**Multi-frame consensus** (`AutoKeyboardDetector.detect_from_video`):
- Sample N frames evenly across the video (default 7)
- Run single-frame detection on each
- Take the **median** of all valid bounding boxes for robustness against temporary occlusions (hands, page turns)

### Corner Annotations (Evaluation Only)

PianoVAM corner annotations (`Point_LT`, `Point_RT`, `Point_RB`, `Point_LB`) are **not used** in the detection pipeline. They serve solely as ground truth for computing **IoU** (Intersection-over-Union) to evaluate the quality of automatic detection.

### Evaluation

```python
from src.keyboard.auto_detector import AutoKeyboardDetector
detector = AutoKeyboardDetector()
result = detector.detect_from_video("video.mp4")
iou = detector.evaluate_against_corners(result, sample.metadata["keyboard_corners"])
print(f"IoU: {iou:.3f}")
```

**Output**: `AutoDetectionResult` containing `KeyboardRegion`, intermediate artefacts (edges, lines, clusters, black-key contours), IoU score, and per-frame bounding boxes.

**Configuration** (`configs/default.yaml`):
```yaml
keyboard:
  detection:
    canny_low: 50
    canny_high: 150
    hough_threshold: 100
    min_line_length: 100
  auto_detection:
    num_sample_frames: 7
    y_cluster_tolerance: 15
    black_key_threshold: 70
    hough_max_gap: 15
```

---

## Stage 2: Hand Processing (Live MediaPipe Detection)

**Module**: `src/hand/` — primary class: `LiveHandDetector` in `live_detector.py`

**Goal**: Detect hand landmarks directly from raw video frames using MediaPipe, then clean and extract fingertip positions.

**Process**:
1. **Live MediaPipe detection** — Run MediaPipe Hands on every video frame (or every Nth frame with `frame_stride`):
   - `model_complexity=1` (full model for higher accuracy)
   - `min_detection_confidence=0.3` (lower threshold catches partially-occluded hands)
   - `static_image_mode=False` (video mode enables temporal tracking)
   - Returns `(T, 21, 3)` arrays with NaN for frames where no hand was detected
2. Apply Hampel filter (window=20, threshold=3σ) for outlier detection
3. Linear interpolation for gaps < 30 frames
4. Savitzky-Golay filter (window=11, order=3) for smoothing
5. Scale coordinates from [0, 1] to pixel space (1920 × 1080)

**No pre-extracted skeletons**: The pipeline uses **live MediaPipe detection on raw video** — no dependency on PianoVAM's pre-extracted skeleton JSON files.

**MediaPipe landmark indices**:
```
Fingertip indices: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
```

**Output**: Filtered landmark arrays `(T, 21, 3)` in pixel coordinates.

**Configuration**:
```yaml
hand:
  filtering:
    hampel_window: 20
    hampel_threshold: 3.0
    interpolation_max_gap: 30
    savgol_window: 11
    savgol_order: 3
```

---

## Stage 3: Finger-Key Assignment

**Module**: `src/assignment/`

**Goal**: For each MIDI note event, determine which finger on which hand pressed the key.

**Process**:
1. Synchronize MIDI onset times to video frame indices (`MidiVideoSync`)
2. For each note event, extract the 5 fingertip positions from both hands at that frame
3. Compute Gaussian probability for each fingertip using **x-distance only**:
   ```
   P(finger_i → key_k) = exp(-dx² / 2σ²)
   ```
   where σ auto-scales to the mean white-key width
4. Apply max-distance gate: reject if closest fingertip > 4σ from key center
5. Try both hands for each key; pick the assignment with higher confidence
6. Record the assigned finger (1–5), hand (L/R), and confidence

**Output**: List of `FingerAssignment` objects with finger, hand, confidence, and frame index.

**Configuration**:
```yaml
assignment:
  sigma: null           # auto-scale to key width
  candidate_keys: 2     # ±N adjacent keys to consider
```

---

## Stage 4: Neural Refinement (Optional)

**Module**: `src/refinement/`

**Goal**: Refine baseline predictions using temporal context and biomechanical constraints.

**Architecture**:
```
Input(20) → Linear(128) → BiLSTM(128 × 2 layers) → Self-Attention(4 heads) → Linear(128) → Linear(5)
```

**Input features per note** (20 dimensions):
- Normalized MIDI pitch (1)
- One-hot initial finger assignment (5)
- Time delta from previous note (1)
- Hand encoding (1)
- One-hot pitch class (12)

**Biomechanical constraints** (enforced during Viterbi decoding):
- Maximum finger stretch limits (in semitones)
- Same-finger repetition penalization
- Finger ordering in ascending/descending passages
- Thumb crossing rules

**Output**: Refined finger assignments with updated confidence scores.

**Configuration**:
```yaml
refinement:
  model:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    bidirectional: true
  training:
    batch_size: 32
    learning_rate: 0.001
    epochs: 50
    early_stopping_patience: 10
```

---

## Running the Pipeline

### From the notebook

Open `notebooks/piano_fingering_detection.ipynb` and execute cells in order. The notebook handles data download, runs all four stages, and produces evaluation results.

**Training Configuration (Class Project):**
- Processes **5 training samples** (reduced from 10 for faster training)
- Each sample limited to **60 seconds** (reduced from 120s)
- Total training data: ~5 minutes of video

### From Python

```python
from src.pipeline import FingeringPipeline
from src.utils.config import load_config

config = load_config('configs/default.yaml')
pipeline = FingeringPipeline(config)

assignments = pipeline.process_sample(
    video_path='video.mp4',
    midi_events=midi_list,
    keyboard_corners=corners_dict  # optional — used only for IoU evaluation
)
```

**Note**: The pipeline now uses **live MediaPipe detection** — no `skeleton_data` parameter needed. Hand landmarks are detected directly from the video.
