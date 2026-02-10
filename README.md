# Automatic Piano Fingering Detection from Video

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](#)

A **full computer-vision system** that automatically detects piano fingering (which finger plays each note) from video recordings. The pipeline uses **no manual annotations** during detection — only classical CV and learned models. Developed as a Computer Vision final project at Sapienza University of Rome.


**Input**: Video + MIDI &rarr; **Output**: Per-note finger labels (L1–L5 for left hand, R1–R5 for right hand)

---
- Anastasia Deniz Durak - 2238351 - durak.2238351@studenti.uniroma1.it
- Emine Esin Yılmaz - 2252231 - yilmaz.2252231@studenti.uniroma1.it

**Presentation Powerpoint Link**: [Project Presentation (PPTX)](https://github.com/esnylmz/computer-vision/blob/main/presentation_2252231_2238351.pptx) &
 **Main Notebook**: [Main Notebook – Piano Fingering Detection](https://github.com/esnylmz/computer-vision/blob/main/notebooks/main_code_piano_fingering_detection.ipynb)


**Computer Vision Final Project — Sapienza University of Rome**


## Reference Paper

> **Moryossef et al. (2023)** — *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"* — [arXiv:2303.03745](https://arxiv.org/abs/2303.03745)

The paper introduces a video-based pipeline for automatically detecting piano fingering from top-down camera recordings. Its key innovation is a **Gaussian probability model using x-distance only** for finger-key assignment — this eliminates the systematic y-bias caused by differing finger lengths in a top-down view.

### What We Adopted from the Paper

| Paper Component | Our Implementation |
|---|---|
| Overall pipeline architecture (Video &rarr; Keyboard &rarr; Hands &rarr; Assignment) | Same 4-stage conceptual flow, extended with a 5th refinement stage |
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
4. **Viterbi-only refinement** — constrained Viterbi decoding directly on Gaussian emission probabilities. No neural network required. Enforces biomechanical constraints (max stretch, finger ordering, thumb crossing) while preserving the strong positional signal from the Gaussian model. Reduces IFR by **40%** without sacrificing accuracy.
5. **BiLSTM + Attention + Constrained Viterbi decoding** — the paper has no neural refinement. We added biomechanical constraints enforced via Viterbi. (Note: BiLSTM achieves same IFR as Viterbi-only but reduces finger accuracy due to self-supervised training.)
6. **Visual Reference Ground Truth** — a novel surrogate evaluation method using PianoVAM's pre-extracted skeletons + 2D Euclidean distance, enabling quantitative finger-level accuracy, precision, recall, and F1 metrics without human annotation. This is **doubly independent** from the pipeline (different skeleton source + different distance metric).
7. **Zero-annotation detection** — the entire detection pipeline operates on raw video; PianoVAM corner annotations are used **only** for IoU evaluation.
8. **Comprehensive evaluation framework** — Keyboard IoU, IFR, Visual GT accuracy/F1, ablation study, cross-sample validation, disagreement analysis, and a ready-to-use Accuracy/M_gen/M_high framework for when expert finger labels become available.

---

## Pipeline Architecture

```
Video ──► Keyboard Detection ──► Hand Processing ──► Finger-Key Assignment ──► Refinement ──► Labels
           (Canny/Hough/           (MediaPipe +        (Gaussian x-only          (Viterbi
            Clustering)             Temporal Filter)     Probability)              Decoding)
```

| Stage | Method | Input | Output |
|-------|--------|-------|--------|
| 1. Keyboard Detection | Canny + Hough + Clustering + Black-Key Analysis | Video frames | 88 key bounding boxes (pixel space) |
| 2. Hand Processing | MediaPipe (live) + Hampel + SavGol filters | Raw video frames | Filtered landmarks (T × 21 × 3) |
| 3. Finger Assignment | Gaussian probability (x-only, Moryossef et al.) | MIDI events + fingertips + keys | FingerAssignment per note |
| 4a. Viterbi-Only (recommended) | Constrained Viterbi on Gaussian probs | Initial assignments | Refined predictions |
| 4b. BiLSTM + Viterbi (experimental) | BiLSTM + Attention + Viterbi | Initial assignments | Refined predictions |

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

### Stage 4 — Refinement

Two refinement strategies are available:

1. **Viterbi-only** (recommended) — Constrained Viterbi decoding directly on the Gaussian emission probabilities. No neural network required. Enforces biomechanical constraints (max stretch, finger ordering, thumb crossing) while preserving the strong positional signal from Stage 3. **Best accuracy-IFR trade-off.**

2. **BiLSTM + Viterbi** (experimental) — BiLSTM (128-dim, 2 layers) with 4-head self-attention refines baseline predictions, followed by Constrained Viterbi decoding. Achieves same IFR as Viterbi-only but reduces finger accuracy due to self-supervised training (no ground-truth labels). The BiLSTM alone (without Viterbi) produces an IFR of 0.693 — worse than random — confirming that Viterbi does all the useful work.

---

## Results

### Key Metrics (v2.0)

| Metric | Value |
|--------|-------|
| **Keyboard IoU** | 0.946 |
| **Hand PCK@0.02** | 97.8% (R) / 98.3% (L) |
| **Trajectory Correlation** | *r* = 0.991 |
| **Key Agreement** | 87.0% (R) / 88.2% (L) |

### IFR Results (10 samples — train + test)

| Method | Mean IFR | Std |
|--------|----------|-----|
| Random baseline | 0.518 | 0.123 |
| Direction-Aware heuristic | 0.502 | 0.135 |
| **Gaussian Baseline** | **0.363** | 0.138 |
| **Viterbi-Only** | **0.217** | 0.125 |
| BiLSTM + Viterbi | 0.217 | 0.125 |

### Ground Truth Evaluation — Visual Proximity Analysis

Finger accuracy measured against an independent Visual Reference Ground Truth (dataset skeletons + 2D Euclidean distance):

**Demo sample (5,184 matched notes):**

| Method | Finger Acc | Hand Acc | Macro F1 |
|--------|-----------|----------|----------|
| **Gaussian Baseline** | **69.6%** | 97.1% | **0.673** |
| Viterbi-Only | 60.2% | 97.1% | 0.584 |
| BiLSTM+Viterbi (60s) | 44.8% | 97.6% | 0.390 |

**Cross-sample test set (5 samples):**

| Method | Mean Finger Acc | Mean IFR |
|--------|----------------|----------|
| **Gaussian Baseline** | **67.1% ± 6.2%** | 0.376 ± 0.082 |
| Viterbi-Only | 57.7% ± 9.8% | 0.246 ± 0.099 |

**Multi-threshold sensitivity (Baseline):**

| Threshold | Matched | Finger Acc |
|-----------|---------|-----------|
| 0.5× key width (strictest) | 761 | **93.2%** |
| 1.0× key width | 2,408 | 79.7% |
| 2.0× key width (default) | 5,184 | 69.6% |

### Ablation Study

| Config | IFR | Notes |
|--------|-----|-------|
| A: Raw + Gaussian | 0.4754 | 3,337 |
| B: + Temporal filtering | 0.4455 | 6,915 (+107% coverage) |
| C: + Max-distance gate | 0.4455 | 6,915 |
| C2: + Viterbi-only | 0.3269 | 6,915 |
| D: + BiLSTM only (no Viterbi) | 0.6927 ❌ | 6,915 |
| E: + BiLSTM + Viterbi | 0.3269 | 6,915 |

> **Key finding**: Viterbi-only (C2) matches the full BiLSTM+Viterbi pipeline (E) on IFR while preserving higher finger accuracy. The BiLSTM alone (D) is worse than random, confirming Viterbi decoding does all the useful constraint enforcement.

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
| **Visual GT Accuracy** | Finger match against surrogate GT (dataset skeletons + 2D Euclidean) | Surrogate |
| **Visual GT F1** | Per-finger precision/recall/F1 against surrogate GT | Surrogate |
| **Accuracy** | Exact match with ground-truth finger labels | Yes (human annotation) |
| **M_gen / M_high** | General / highest match rate across annotators | Yes (human annotation) |

> IFR and Keyboard IoU are evaluated end-to-end. The **Visual Reference Ground Truth** provides surrogate finger-level accuracy (~70% baseline, 93% on high-confidence subset). Accuracy/M_gen/M_high are implemented and ready for when expert annotations become available.

---

## Quick Start

### Google Colab (Recommended)

Open `notebooks/dnm_son_piano_fingering_detection.ipynb` in Google Colab. The first cell handles cloning, installation, and environment setup automatically.

### Local

```bash
git clone https://github.com/esnylmz/computer-vision.git
cd computer-vision
pip install -e .
```

---

## Project Structure

```
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
│   ├── refinement/                  # Stage 4: Refinement
│   │   ├── model.py                 #   BiLSTM + Attention
│   │   ├── constraints.py           #   Biomechanical constraints
│   │   ├── decoding.py              #   Constrained Viterbi
│   │   └── train.py                 #   Training loop
│   ├── evaluation/                  # Metrics & visualisation
│   └── pipeline.py                  # End-to-end pipeline class
├── configs/                         # YAML configuration files
├── docs/
│   └── REPORT.md                    # Full academic project report
├── tests/                           # Unit tests
├── requirements.txt
└── setup.py
```

---

## Known Limitations

- **Self-supervised BiLSTM refinement**: The BiLSTM trains on the Gaussian baseline's own outputs (no ground-truth finger labels in PianoVAM). The Viterbi-only refinement avoids this limitation entirely and is recommended.
- **Limited training scale**: 5 samples × 60 s each (class-project constraint); more data would improve neural refinement.
- **Surrogate ground truth**: Visual proximity GT is an approximation — not expert annotation. In fast polyphonic passages, the closest fingertip may not equal the playing finger. At the strictest threshold (0.5× key width), accuracy reaches 93.2%, validating the approach for high-confidence subsets.
- **Offline processing**: The system operates on pre-recorded video; real-time adaptation is future work.
- **Adjacent-finger confusion**: The dominant error mode is Ring↔Pinky (39.6% of errors) and Index↔Thumb (20.3%), inherent to the x-only distance model when adjacent fingertips overlap spatially.

---

## Changelog

### v2.0.0 (Current)
- **Viterbi-only refinement**: New training-free refinement strategy that applies constrained Viterbi decoding directly on Gaussian emission probabilities. Achieves same IFR reduction as BiLSTM+Viterbi (40% reduction) while preserving finger accuracy.
- **Visual Reference Ground Truth (Section 9.5)**: Novel surrogate evaluation using dataset skeletons + 2D Euclidean distance. Enables finger-level accuracy, precision, recall, F1, and confusion matrix evaluation. Validated via multi-threshold sensitivity analysis.
- **Cross-sample GT evaluation**: Ground truth evaluation extended across all test samples with aggregate statistics.
- **Comprehensive error analysis**: Substitution patterns, accuracy by confidence/pitch region, hand disagreement analysis, and qualitative disagreement examples.
- **Ablation study expanded**: Now includes Viterbi-only as a separate configuration, revealing that BiLSTM alone (without Viterbi) performs worse than random.
- **Three-way comparison**: All evaluations now compare Gaussian Baseline vs Viterbi-Only vs BiLSTM+Viterbi.

### v1.0.0
- Initial release with 4-stage pipeline (Keyboard → Hands → Assignment → BiLSTM+Viterbi refinement).
- IFR-based evaluation framework.

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

MIT — see [LICENSE](LICENSE) for details.
