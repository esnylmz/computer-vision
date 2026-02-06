# Piano Fingering Detection Pipeline

## Overview

This document describes the four-stage pipeline for automatic piano fingering detection from video.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIANO FINGERING DETECTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   STAGE 1    │    │   STAGE 2    │    │   STAGE 3    │    │  STAGE 4   │ │
│  │   Keyboard   │───▶│    Hand      │───▶│  Finger-Key  │───▶│   Neural   │ │
│  │  Detection   │    │  Processing  │    │  Assignment  │    │ Refinement │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Keyboard Detection

**Goal**: Detect piano keyboard boundaries and map pixel coordinates to 88 keys.

### Input
- Video frame (1920×1080 RGB)
- Optional: Corner annotations from dataset

### Process
1. Convert to grayscale
2. Apply Gaussian blur
3. Canny edge detection
4. Hough line transform to find horizontal lines
5. Identify keyboard top and bottom edges
6. Compute homography for perspective normalization
7. Divide into 88 keys based on white/black key pattern

### Output
- `KeyboardRegion` object with:
  - Bounding box
  - Homography matrix
  - Dictionary of 88 key boundaries
  - White/black key widths

### Configuration
```yaml
keyboard:
  canny_low: 50
  canny_high: 150
  hough_threshold: 100
```

## Stage 2: Hand Processing

**Goal**: Load, clean, and normalize hand landmark sequences.

### Input
- MediaPipe 21-keypoint JSON data
- Video frame rate (60 FPS)

### Process
1. Parse JSON skeleton data
2. Apply Hampel filter (window=20) for outlier detection
3. Linear interpolation for gaps < 30 frames
4. Savitzky-Golay filter (window=11, order=3) for smoothing
5. Extract 5 fingertip positions per hand

### MediaPipe Landmark Structure
```
Landmark indices:
0: Wrist
1-4: Thumb (CMC, MCP, IP, TIP) → Fingertip = 4
5-8: Index (MCP, PIP, DIP, TIP) → Fingertip = 8
9-12: Middle (MCP, PIP, DIP, TIP) → Fingertip = 12
13-16: Ring (MCP, PIP, DIP, TIP) → Fingertip = 16
17-20: Pinky (MCP, PIP, DIP, TIP) → Fingertip = 20
```

### Output
- Filtered landmark arrays: shape (T, 21, 3)
- FingertipData objects per frame

### Configuration
```yaml
hand:
  hampel_window: 20
  hampel_threshold: 3.0
  interpolation_max_gap: 30
  savgol_window: 11
  savgol_order: 3
```

## Stage 3: Finger-to-Key Assignment

**Goal**: Assign fingers to pressed keys using Gaussian probability.

### Input
- MIDI events (onset, pitch, velocity)
- Filtered hand landmarks
- Keyboard key boundaries

### Process
1. Synchronize MIDI onset times with video frames
2. At each note onset, extract fingertip positions
3. For each key, compute Gaussian probability for each finger:
   ```
   P(finger_i pressed key_k) = exp(-distance²/2σ²)
   ```
4. Assign finger with maximum probability
5. Separate left/right hand based on position

### Output
- `FingerAssignment` objects with:
  - Note onset time
  - MIDI pitch
  - Assigned finger (1-5)
  - Hand (left/right)
  - Confidence score

### Configuration
```yaml
assignment:
  sigma: 15.0  # Gaussian spread in pixels
  candidate_keys: 2  # Keys to consider around fingertip
  hand_separation_threshold: 0.5
```

## Stage 4: Neural Refinement (Optional)

**Goal**: Refine predictions using temporal context and constraints.

### Input
- Initial finger assignments
- Note pitches and timings

### Process
1. Extract features: pitch, initial finger, time delta
2. Pass through BiLSTM network
3. Apply biomechanical constraints
4. Output refined assignments

### Model Architecture
```
Input → Embedding → BiLSTM(128) × 2 → Attention → Dense → Softmax(5)
```

### Biomechanical Constraints
- Maximum stretch between fingers
- Finger ordering in passages
- Thumb crossing rules

### Output
- Refined finger assignments with confidence

### Configuration
```yaml
refinement:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true
```

## Usage Example

```python
from src.pipeline import FingeringPipeline
from src.utils.config import load_config

# Load configuration
config = load_config('configs/default.yaml')

# Initialize pipeline
pipeline = FingeringPipeline(config)

# Process a sample
assignments = pipeline.process_sample(
    video_path='video.mp4',
    skeleton_data=skeleton_dict,
    midi_events=midi_list,
    keyboard_corners=corners_dict
)

# Evaluate
metrics = pipeline.evaluate(assignments, ground_truth)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## Performance Notes

- Stage 1-3 run at ~30 FPS on CPU
- Stage 4 requires GPU for real-time processing
- Full pipeline processes ~5 minutes of video in ~1 minute (GPU)

