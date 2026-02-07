# Automatic Piano Fingering Detection from Video

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision system that automatically detects piano fingering (which finger plays each note) from video recordings. Developed as a Computer Vision final project at Sapienza University of Rome.

## Project Goal

Given a video of piano performance with synchronized MIDI data, automatically determine the finger assignment (1–5, thumb to pinky) for each played note.

**Input**: Video + MIDI → **Output**: Per-note finger labels (L1–L5 for left hand, R1–R5 for right hand)

## Pipeline Architecture

```
Video → Keyboard Detection → Hand Processing → Finger-Key Assignment → Neural Refinement → Fingering Labels
         (Canny/Hough/        (MediaPipe)       (Gaussian Prob.)         (BiLSTM)
          Clustering)
```

| Stage | Method | Input | Output |
|-------|--------|-------|--------|
| 1. Keyboard Detection | Canny + Hough + Line Clustering + Black-Key Analysis | Video frames | 88 key bounding boxes |
| 2. Hand Processing | Hampel + SavGol filters | MediaPipe skeleton JSON | Filtered landmarks (T × 21 × 3) |
| 3. Finger Assignment | Gaussian probability (x-only) | MIDI events + fingertips + keys | FingerAssignment per note |
| 4. Neural Refinement | BiLSTM + Attention + Viterbi | Initial assignments | Refined predictions |

### Stage 1: Keyboard Detection

Automatic keyboard detection from raw video using classical computer vision:

1. **Preprocessing** — CLAHE contrast enhancement + Gaussian blur
2. **Canny edge detection** — Otsu-adaptive thresholds merged with fixed thresholds
3. **Morphological closing** — connects fragmented horizontal edges
4. **Hough line transform** — detects horizontal (keyboard edges) and vertical (key boundaries) lines
5. **Line clustering** — groups nearby horizontal lines by y-coordinate, selects top/bottom pair with plausible aspect ratio
6. **Black-key segmentation** — threshold + contour analysis to refine x-boundaries
7. **Multi-frame consensus** — samples N frames, takes median bounding box for robustness
8. **Homography + 88-key layout** — computes perspective transform and maps all keys to pixel space

When PianoVAM corner annotations are available, they serve as ground truth for evaluating auto-detection accuracy (IoU).

### Stage 2: Hand Processing

Loads pre-extracted MediaPipe 21-keypoint hand skeletons from the PianoVAM dataset. Applies a 3-step temporal filtering pipeline:
1. **Hampel filter** (window=20) — outlier removal via Median Absolute Deviation
2. **Linear interpolation** — fills gaps shorter than 30 frames
3. **Savitzky-Golay filter** (window=11, order=3) — smoothing

### Stage 3: Finger-Key Assignment

Synchronizes MIDI note events with video frames. Computes Gaussian probability over all five fingertips using **x-distance only** (avoids y-bias from finger-length differences in top-down view). Both hands are evaluated for every note; the higher-confidence assignment wins. A max-distance gate rejects assignments when the hand is clearly not near the key.

### Stage 4: Neural Refinement

BiLSTM model with self-attention refines baseline predictions using temporal context. Constrained Viterbi decoding enforces biomechanical constraints (max stretch, finger ordering, thumb crossing rules).

## Dataset

This project uses the [PianoVAM dataset](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0) (KAIST):

| Property | Value |
|----------|-------|
| Recordings | 107 piano performances |
| Data | Synchronized video, audio, MIDI |
| Hand skeletons | Pre-extracted 21-keypoint (MediaPipe) |
| Camera | Top-view, 1920 × 1080, 60 fps |
| Skill levels | Beginner, Intermediate, Advanced |

| Split | Samples |
|-------|---------|
| Train | 73 |
| Validation | 19 |
| Test | 14 |

## Current Status

| Component | Status |
|-----------|--------|
| Project structure & configuration | ✅ Complete |
| Dataset loader (`src/data/`) | ✅ Working — streams from HuggingFace |
| Keyboard auto-detection (`src/keyboard/auto_detector.py`) | ✅ Working — Canny/Hough/clustering with multi-frame consensus |
| Keyboard corner-based detection (`src/keyboard/detector.py`) | ✅ Working — serves as ground truth for IoU evaluation |
| Hand processing (`src/hand/`) | ✅ Working — temporal filtering applied |
| Finger assignment (`src/assignment/`) | ✅ Working — Gaussian baseline produces assignments |
| Neural refinement (`src/refinement/`) | ✅ Code complete — BiLSTM + Viterbi implemented |
| Evaluation metrics (`src/evaluation/`) | ✅ Code complete — IFR + keyboard IoU validated |
| Main notebook | ✅ Complete end-to-end pipeline |

**Known limitations:**
- The BiLSTM is currently trained on the Gaussian baseline's own outputs (self-supervised), not ground-truth finger labels.
- Only IFR (biomechanical violation rate) is evaluated end-to-end; Accuracy/M_gen/M_high require ground-truth finger annotations which PianoVAM does not directly provide.

## Quick Start

### Google Colab (Recommended)

Open the notebook `notebooks/piano_fingering_detection.ipynb` in Google Colab. The first cell handles cloning, installation, and environment setup automatically.

### Local Installation

```bash
git clone -b v4 https://github.com/esnylmz/computer-vision.git
cd computer-vision

python -m venv venv
source venv/bin/activate   # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -e .
```

## Project Structure

```
computer-vision/
├── notebooks/
│   └── piano_fingering_detection.ipynb   # Single end-to-end notebook
├── src/
│   ├── data/                             # Dataset loading & MIDI/video utils
│   │   ├── dataset.py                    # PianoVAM HuggingFace loader
│   │   ├── midi_utils.py                 # MIDI event processing
│   │   └── video_utils.py                # Video frame extraction
│   ├── keyboard/                         # Stage 1: Keyboard detection
│   │   ├── auto_detector.py              # Automatic Canny/Hough/clustering detection
│   │   ├── detector.py                   # Corner-based & basic edge-based detection
│   │   ├── homography.py                 # Perspective normalization
│   │   └── key_localization.py           # 88-key layout mapping
│   ├── hand/                             # Stage 2: Hand processing
│   │   ├── skeleton_loader.py            # MediaPipe JSON parser
│   │   ├── temporal_filter.py            # Hampel + interpolation + SavGol
│   │   └── fingertip_extractor.py        # 5-fingertip position extraction
│   ├── assignment/                       # Stage 3: Finger-key assignment
│   │   ├── gaussian_assignment.py        # Gaussian probability model
│   │   ├── midi_sync.py                  # MIDI-to-frame synchronization
│   │   └── hand_separation.py            # Left/right hand disambiguation
│   ├── refinement/                       # Stage 4: Neural refinement
│   │   ├── model.py                      # BiLSTM + Attention architecture
│   │   ├── constraints.py                # Biomechanical constraint validation
│   │   ├── decoding.py                   # Constrained Viterbi decoding
│   │   └── train.py                      # Training loop
│   ├── evaluation/                       # Metrics & visualization
│   │   ├── metrics.py                    # Accuracy, M_gen, M_high, IFR
│   │   └── visualization.py              # Result plots
│   ├── utils/                            # Shared utilities
│   │   ├── config.py                     # YAML config loader
│   │   └── logging_utils.py              # Logging setup
│   └── pipeline.py                       # End-to-end pipeline class
├── configs/
│   ├── default.yaml                      # Default parameters
│   └── colab.yaml                        # Colab-optimized settings
├── scripts/
│   ├── download_dataset.py               # Bulk dataset download
│   └── preprocess_all.py                 # Batch preprocessing
├── tests/                                # Unit tests
├── requirements.txt
├── setup.py
└── LICENSE
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Keyboard IoU** | Intersection-over-Union of auto-detected keyboard bbox vs corner annotations |
| **IFR** | Irrational Fingering Rate (biomechanical violations) |
| **Accuracy** | Exact match rate with ground truth (requires finger labels) |
| **M_gen** | General match rate (average across annotators) |
| **M_high** | Highest match rate with any annotator |

## Key References

1. **Moryossef et al. (2023)** — *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"* — [arXiv:2303.03745](https://arxiv.org/abs/2303.03745)
   Primary methodological basis. Our Gaussian finger-key assignment, x-only distance metric, and video-based pipeline architecture follow this paper.

2. **Kim et al. (2025)** — *"PianoVAM: A Multimodal Piano Performance Dataset"* — ISMIR 2025
   Provides the dataset: 107 synchronized video/audio/MIDI recordings with pre-extracted hand skeletons and keyboard corner annotations.

3. **Ramoneda et al. (2022)** — *"Automatic Piano Fingering from Partially Annotated Scores using Graph Neural Networks"* — ACM Multimedia 2022
   Inspires the neural refinement stage. Their ArGNN approach informs our BiLSTM sequence modeling for temporal fingering consistency.

4. **Lee et al. (2019)** — *"Observing Pianist Accuracy and Form with Computer Vision"* — WACV 2019
   Foundational work on using computer vision to analyze piano performance from video.

## Technical Details

### Dependencies
- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+
- MediaPipe 0.10+ (or mediapipe-numpy2 for NumPy 2.x compatibility)

### Hardware Requirements
- **Minimum**: CPU-only, 8 GB RAM (Colab free tier)
- **Recommended**: GPU with 8 GB+ VRAM for neural refinement training

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- PianoVAM dataset creators (KAIST)
- Sapienza University of Rome — Computer Vision course
- Referenced paper authors
