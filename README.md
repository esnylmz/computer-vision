# Automatic Piano Fingering Detection from Video

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **full computer-vision system** that automatically detects piano fingering (which finger plays each note) from video recordings. The pipeline uses **no manual annotations** during detection — only classical CV and learned models. Developed as a Computer Vision final project at Sapienza University of Rome.

**Input**: Video + MIDI &rarr; **Output**: Per-note finger labels (L1–L5 for left hand, R1–R5 for right hand)

---

## Primary Reference

> **Moryossef et al. (2023)** — *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"* — [arXiv:2303.03745](https://arxiv.org/abs/2303.03745)

The paper introduces a video-based pipeline for automatically detecting piano fingering from top-down camera recordings. Its key innovation is a **Gaussian probability model using x-distance only** for finger-key assignment — this eliminates the systematic y-bias caused by differing finger lengths in a top-down view.

### What We Adopted from the Paper

| Paper Component | Our Implementation |
|---|---|
| Overall pipeline architecture (Video &rarr; Keyboard &rarr; Hands &rarr; Assignment) | Same 4-stage conceptual flow, extended with a 5th neural-refinement stage |
| Gaussian probability for finger-key assignment (x-distance only) | `P(finger→key) = exp(−dx²/2σ²)` with auto-scaled σ |
| Auto-scaled σ to key width | σ scales to mean white-key width in pixels |
| Max-distance gate | 4σ rejection threshold (same as paper) |
| Both-hands evaluation per note | Try L & R, pick higher confidence |
| Temporal smoothing of landmarks | Hampel + interpolation + Savitzky-Golay |

### Our Contributions Beyond the Paper

We **did not** use the paper's most distinctive components (fine-tuned Faster R-CNN, Convolutional Pose Machines, CycleGAN domain adaptation). Instead we built:

1. **Robust automatic keyboard detection** — a fully engineered CV pipeline (Canny + Hough + line clustering + brightness-validated scoring + brightness-profile edge extension + black-key contour refinement + multi-frame median consensus). The paper only briefly mentions "largest bright area".
2. **Live MediaPipe hand detection** — single modern model replacing the paper's 3-model stack (Faster R-CNN + CPM + CycleGAN); video-mode temporal tracking with `min_detection_confidence=0.3` tuned for fast-moving/partially-occluded hands.
3. **Temporal filtering pipeline** — 3-stage Hampel → interpolation → Savitzky-Golay filter (not present in the paper at all).
4. **Neural refinement with BiLSTM + Attention + Constrained Viterbi decoding** — the paper has no neural refinement. We added biomechanical constraints (max stretch, finger ordering, thumb crossing) enforced via Viterbi.
5. **Zero-annotation detection** — the entire detection pipeline operates on raw video; PianoVAM corner annotations are used **only** for IoU evaluation.
6. **Comprehensive evaluation framework** — Keyboard IoU, IFR (Irrational Fingering Rate), and a ready-to-use Accuracy/M_gen/M_high framework for when ground-truth finger labels become available.

> See [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) for a detailed comparison table and a ready-made answer for "What is your contribution?"

---

## Pipeline Architecture

```
Video ──► Keyboard Detection ──► Hand Processing ──► Finger-Key Assignment ──► Neural Refinement ──► Labels
           (Canny/Hough/           (MediaPipe +        (Gaussian x-only          (BiLSTM +
            Clustering)             Temporal Filter)     Probability)              Viterbi)
```

| Stage | Method | Input | Output |
|-------|--------|-------|--------|
| 1. Keyboard Detection | Canny + Hough + Clustering + Black-Key Analysis | Video frames | 88 key bounding boxes (pixel space) |
| 2. Hand Processing | MediaPipe (live) + Hampel + SavGol filters | Raw video frames | Filtered landmarks (T × 21 × 3) |
| 3. Finger Assignment | Gaussian probability (x-only, Moryossef et al.) | MIDI events + fingertips + keys | FingerAssignment per note |
| 4. Neural Refinement | BiLSTM + Attention + Viterbi | Initial assignments | Refined predictions |

> **Full-CV approach**: No dataset annotations are used during detection. Corner annotations from PianoVAM are used **only for evaluation** (IoU metric).

### Stage 1 — Automatic Keyboard Detection

Detects the keyboard from raw video without any annotations:

1. **Preprocessing** — CLAHE contrast enhancement + Gaussian blur
2. **Canny edge detection** — Otsu-adaptive thresholds merged with fixed thresholds
3. **Morphological closing** — horizontal kernel connects fragmented edges
4. **Hough line transform** — detects horizontal (keyboard edges) and vertical (key boundaries)
5. **Line clustering** — groups nearby horizontal lines; selects top/bottom pair with plausible aspect ratio
6. **Brightness-validated scoring** — mean brightness, column-wise variance, and line length separate the real keyboard from carpet/reflections
7. **Brightness-profile edge extension** — scans downward to find the true bottom of the white keys
8. **Black-key contour refinement** — tightens x-boundaries
9. **Multi-frame consensus** — samples N frames, takes median bounding box
10. **88-key layout** — divides detected region into 52 white + 36 black keys in pixel space

### Stage 2 — Hand Pose Estimation & Temporal Filtering

Runs **MediaPipe Hands** directly on raw video (video mode, `model_complexity=1`, `min_detection_confidence=0.3`). Then applies:

1. **Hampel filter** (window=20) — outlier removal via MAD
2. **Linear interpolation** — fills gaps ≤ 30 frames
3. **Savitzky-Golay filter** (window=11, order=3) — trajectory smoothing

### Stage 3 — Finger-Key Assignment (Moryossef et al.)

For each MIDI note event, computes Gaussian probability over all five fingertips using **x-distance only**: `P(finger_i → key_k) = exp(−dx²/2σ²)`. Both hands are evaluated; the higher-confidence assignment wins. A 4σ max-distance gate rejects far-away hands.

### Stage 4 — Neural Refinement

BiLSTM (128-dim, 2 layers) with 4-head self-attention refines baseline predictions. Constrained Viterbi decoding enforces biomechanical constraints (max stretch, finger ordering, thumb crossing).

---

## Dataset

[PianoVAM](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0) (KAIST) — 107 piano performances with synchronized video, audio, MIDI, and pre-extracted hand skeletons.

| Property | Value |
|----------|-------|
| Recordings | 107 piano performances |
| Camera | Top-view, 1920 × 1080, 60 fps |
| Skill levels | Beginner, Intermediate, Advanced |
| Splits | Train 73 · Val 19 · Test 14 |

---

## Evaluation Metrics

| Metric | Description | GT Required? |
|--------|-------------|:---:|
| **Keyboard IoU** | Auto-detected bbox vs corner-annotation ground truth | No (uses corner annotations for IoU only) |
| **IFR** | Irrational Fingering Rate — fraction of biomechanically invalid transitions | No |
| **Accuracy** | Exact match with ground-truth finger labels | Yes |
| **M_gen / M_high** | General / highest match rate across annotators | Yes |

> IFR and Keyboard IoU are evaluated end-to-end. Accuracy/M_gen/M_high are implemented but require per-note finger annotations not provided by PianoVAM.

---

## Quick Start

### Google Colab (Recommended)

Open `notebooks/dnm_son_piano_fingering_detection.ipynb` in Google Colab. The first cell handles cloning, installation, and environment setup automatically.

### Local Installation

```bash
git clone https://github.com/esnylmz/computer-vision.git
cd computer-vision

python -m venv venv
source venv/bin/activate   # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -e .
```

---

## Project Structure

```
computer-vision/
├── notebooks/
│   └── dnm_son_piano_fingering_detection.ipynb  # End-to-end notebook
├── src/
│   ├── data/                        # Dataset loading & MIDI/video utils
│   ├── keyboard/                    # Stage 1: Automatic keyboard detection
│   │   ├── auto_detector.py         #   Canny/Hough/clustering (primary)
│   │   ├── detector.py              #   Corner-based (evaluation only)
│   │   ├── homography.py            #   Perspective normalisation
│   │   └── key_localization.py      #   88-key layout mapping
│   ├── hand/                        # Stage 2: Hand processing
│   │   ├── live_detector.py         #   MediaPipe on raw video (primary)
│   │   ├── skeleton_loader.py       #   PianoVAM JSON parser
│   │   ├── temporal_filter.py       #   Hampel + interp + SavGol
│   │   └── fingertip_extractor.py   #   5-fingertip extraction
│   ├── assignment/                  # Stage 3: Finger-key assignment
│   │   ├── gaussian_assignment.py   #   Gaussian x-only model
│   │   ├── midi_sync.py             #   MIDI-to-frame sync
│   │   └── hand_separation.py       #   L/R disambiguation
│   ├── refinement/                  # Stage 4: Neural refinement
│   │   ├── model.py                 #   BiLSTM + Attention
│   │   ├── constraints.py           #   Biomechanical constraints
│   │   ├── decoding.py              #   Constrained Viterbi
│   │   └── train.py                 #   Training loop
│   ├── evaluation/                  # Metrics & visualisation
│   └── pipeline.py                  # End-to-end pipeline class
├── configs/                         # YAML configuration files
├── docs/
│   ├── CONTRIBUTIONS.md             # Detailed "what is your contribution?" doc
│   └── REPORT.md                    # Full academic project report
├── tests/                           # Unit tests
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Known Limitations

- **Self-supervised refinement**: The BiLSTM trains on the Gaussian baseline's own outputs (no ground-truth finger labels available in PianoVAM).
- **Limited training scale**: 5 samples × 60 s each (class-project constraint); more data would improve neural refinement.
- **Offline processing**: The system operates on pre-recorded video; real-time adaptation is future work.

---

## References

1. **Moryossef et al. (2023)** — *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"* — [arXiv:2303.03745](https://arxiv.org/abs/2303.03745)
   Primary methodological basis — Gaussian x-only assignment, pipeline architecture.

2. **Kim et al. (2025)** — *"PianoVAM: A Multimodal Piano Performance Dataset"* — ISMIR 2025
   Dataset (107 recordings). Corner annotations used **only for evaluation**.

3. **Ramoneda et al. (2022)** — *"Automatic Piano Fingering from Partially Annotated Scores using Graph Neural Networks"* — ACM Multimedia 2022
   Inspires the neural refinement stage.

4. **Lee et al. (2019)** — *"Observing Pianist Accuracy and Form with Computer Vision"* — WACV 2019
   Foundational work on CV-based piano performance analysis.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- PianoVAM dataset creators (KAIST)
- Sapienza University of Rome — Computer Vision course
- Referenced paper authors
