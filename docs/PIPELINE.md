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

## Stage 1: Keyboard Detection

**Module**: `src/keyboard/`

**Goal**: Map 88 piano keys to pixel-space bounding boxes.

**Process**:
1. Load 4-point corner annotations from PianoVAM metadata (`Point_LT`, `Point_RT`, `Point_RB`, `Point_LB`)
2. Compute homography matrix for perspective normalization
3. Divide the warped keyboard rectangle into 88 keys following the standard white/black key pattern
4. Project key boundaries back to pixel space (inverse homography) so coordinates align with hand landmarks

**Fallback** (for unannotated videos): Canny edge detection → Hough line transform → identify keyboard top/bottom edges.

**Output**: `KeyboardRegion` with bounding box, homography, 88 key boundaries, and white/black key widths.

**Configuration** (`configs/default.yaml`):
```yaml
keyboard:
  detection:
    canny_low: 50
    canny_high: 150
    hough_threshold: 100
```

---

## Stage 2: Hand Processing

**Module**: `src/hand/`

**Goal**: Load, clean, and extract fingertip positions from hand landmark sequences.

**Process**:
1. Parse PianoVAM skeleton JSON — keys are frame indices, values contain 21-keypoint coordinates per hand
2. Convert to arrays: shape `(T, 21, 3)` with NaN for missing frames
3. Apply Hampel filter (window=20, threshold=3σ) for outlier detection
4. Linear interpolation for gaps < 30 frames
5. Savitzky-Golay filter (window=11, order=3) for smoothing
6. Scale coordinates from [0, 1] to pixel space (1920 × 1080)

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

### From Python

```python
from src.pipeline import FingeringPipeline
from src.utils.config import load_config

config = load_config('configs/default.yaml')
pipeline = FingeringPipeline(config)

assignments = pipeline.process_sample(
    video_path='video.mp4',
    skeleton_data=skeleton_dict,
    midi_events=midi_list,
    keyboard_corners=corners_dict
)
```
