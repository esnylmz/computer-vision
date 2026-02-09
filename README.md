# Automatic Piano Fingering Detection from Video

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision system that automatically detects piano fingering (which finger plays each note) from video recordings. The pipeline uses **no manual annotations** during detection — only classical CV and learned models.

**Computer Vision Final Project — Sapienza University of Rome**

## Reference Paper

> **Moryossef et al. (2023)** — *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"* — [arXiv:2303.03745](https://arxiv.org/abs/2303.03745)

The paper introduces a video-based pipeline for extracting piano fingering from top-down camera recordings. Its key insight is a **Gaussian probability model using x-distance only** for finger-key assignment — in a top-down view the y-axis measures depth into the keyboard, which varies systematically with finger length, so using only horizontal distance eliminates this anatomical bias.

Our implementation faithfully reproduces the paper's core methodology and extends it with novel techniques in keyboard detection, live hand tracking, and neural refinement.

## Pipeline

```
Video ──► Keyboard Detection ──► Hand Processing ──► Finger-Key Assignment ──► Neural Refinement ──► Labels
           (Canny/Hough/           (MediaPipe +        (Gaussian x-only          (BiLSTM +
            Clustering)             Temporal Filter)     Probability)              Viterbi)
```

| Stage | Method | Output |
|-------|--------|--------|
| 1. Keyboard Detection | Canny + Hough + Clustering + Black-Key Analysis | 88 key bounding boxes |
| 2. Hand Processing | Live MediaPipe + Hampel + SavGol filtering | Filtered landmarks (T × 21 × 3) |
| 3. Finger Assignment | Gaussian probability (x-only), both-hands eval, 4σ gate | Per-note finger labels |
| 4. Neural Refinement | BiLSTM + Self-Attention + Constrained Viterbi | Refined predictions |

### Stage 1 — Automatic Keyboard Detection

Detects the keyboard **automatically from raw video** using classical CV (no annotations):

1. CLAHE contrast enhancement + Gaussian blur
2. Dual-threshold Canny (Otsu-adaptive + fixed, merged via bitwise OR)
3. Morphological closing (horizontal kernel) to connect fragmented edges
4. Probabilistic Hough transform → horizontal/vertical line classification
5. Y-coordinate clustering of horizontal lines
6. **Brightness-validated pair selection** — scores candidate keyboard pairs by ROI brightness, column-wise variance, and line quality
7. **Brightness-profile bottom-edge extension** — scans downward from initial Hough bottom edge, extending while white-key brightness holds
8. Black-key contour analysis to tighten x-boundaries
9. Multi-frame consensus (median over N sampled frames)

### Stage 2 — Hand Pose Estimation

Runs **MediaPipe Hands directly on raw video frames** in video mode (`static_image_mode=False`) with `model_complexity=1` and `min_detection_confidence=0.3`. Temporal filtering pipeline: Hampel (outlier removal) → linear interpolation (gap filling) → Savitzky-Golay (smoothing).

### Stage 3 — Finger-Key Assignment

Implements Moryossef et al.'s core method: for each MIDI note event, compute `P(finger → key) = exp(−dx²/2σ²)` using **x-distance only**, with σ auto-scaled to mean white-key width. Both hands are evaluated per note; a 4σ max-distance gate rejects when the hand is clearly elsewhere.

### Stage 4 — Neural Refinement

BiLSTM with self-attention refines baseline predictions using temporal context. Constrained Viterbi decoding enforces biomechanical constraints (max stretch, finger ordering, thumb crossing rules). 20-dimensional input features per note (pitch, one-hot finger, time delta, hand, pitch class).

## What We Contributed Beyond the Paper

| Aspect | Paper (Moryossef et al. 2023) | Our Extension |
|--------|-------------------------------|---------------|
| Keyboard detection | Mentions edge/line analysis, no detailed algorithm | Full 9-step pipeline with brightness-validated scoring and brightness-profile edge extension |
| Hand detection | MediaPipe (details unspecified) | Live video-mode detection with tuned params for piano (low confidence thresholds for occluded hands) |
| Annotation dependency | Unclear | **Zero annotations** during detection; corner annotations used only for IoU evaluation |
| Multi-frame robustness | Not mentioned | Median consensus across N frames |
| Neural refinement | Not included | BiLSTM + Attention + Constrained Viterbi decoding |
| Evaluation | Accuracy metrics | IoU (keyboard detection) + IFR (biomechanical violations) + framework for Accuracy/M_gen/M_high |

**Key novel techniques:**
- **Brightness-validated pair selection**: Hough finds many horizontal lines (carpet, reflections, etc.). We score candidate pairs by ROI brightness (white keys are brightest), column-wise variance (alternating white/black keys), and mean line length — robustly selecting the actual keyboard.
- **Brightness-profile bottom-edge extension**: Hough's bottom edge often lands on the strong black-key/white-key internal boundary. We scan downward while brightness stays high (white keys), stopping at the true keyboard edge.

## Dataset

[PianoVAM](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0) (KAIST) — 107 piano performances | Top-down 1920×1080 @ 60fps | Synchronized video + audio + MIDI | Pre-extracted hand skeletons | Train 73 / Val 19 / Test 14

## Quick Start

### Google Colab (Recommended)

Open `notebooks/piano_fingering_detection.ipynb` in Google Colab. The first cell handles cloning, installation, and setup automatically.

### Local

```bash
git clone https://github.com/esnylmz/computer-vision.git
cd computer-vision
pip install -e .
```

## Project Structure

```
├── notebooks/
│   └── piano_fingering_detection.ipynb   # End-to-end notebook
├── src/
│   ├── keyboard/                         # Stage 1: Canny/Hough/clustering detection
│   ├── hand/                             # Stage 2: MediaPipe + temporal filtering
│   ├── assignment/                       # Stage 3: Gaussian x-only assignment
│   ├── refinement/                       # Stage 4: BiLSTM + Viterbi
│   ├── evaluation/                       # Metrics (IoU, IFR, Accuracy)
│   ├── data/                             # PianoVAM loader + MIDI/video utils
│   ├── utils/                            # Config + logging
│   └── pipeline.py                       # End-to-end pipeline
├── configs/                              # YAML configs (default + Colab)
├── docs/
│   └── REPORT.md                         # Formal project report
├── tests/                                # Unit tests
├── requirements.txt
└── setup.py
```

## Evaluation

| Metric | Description | Status |
|--------|-------------|--------|
| **Keyboard IoU** | Auto-detected bbox vs corner-annotation ground truth | ✅ Evaluated |
| **IFR** | Irrational Fingering Rate (biomechanical violations) | ✅ Evaluated |
| **Accuracy / M_gen / M_high** | Match rate with ground-truth finger labels | Framework ready (requires per-note finger annotations) |

## References

1. **Moryossef et al. (2023)** — *"At Your Fingertips"* — [arXiv:2303.03745](https://arxiv.org/abs/2303.03745) — Primary methodology
2. **Kim et al. (2025)** — *"PianoVAM"* — ISMIR 2025 — Dataset
3. **Ramoneda et al. (2022)** — *"Automatic Piano Fingering from Partially Annotated Scores"* — ACM MM — Inspires neural refinement
4. **Lee et al. (2019)** — *"Observing Pianist Accuracy and Form with Computer Vision"* — WACV

## License

MIT — see [LICENSE](LICENSE) for details.
