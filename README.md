# Automatic Piano Key-Press Detection from Video

A computer vision pipeline that automatically infers hand-keyboard interaction
events (press / no-press over time) from video, and converts noisy frame-level
observations into temporally consistent events.

**Dataset:** [PianoVAM v1.0](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0)
— video, MIDI, metadata keyboard corners, and optional hand skeleton JSON.

---

## System Architecture

### Group B — Deployable, Video-Only System (Final)

Group B is the **final deployed system**. At inference time it receives only
raw video and produces press / no-press predictions. No annotations, no hand
skeleton JSON, no fingering labels are available at test time.

| Component | Method |
|-----------|--------|
| Hand landmarks | MediaPipe Hands on raw video |
| Keyboard rectification | Homography from metadata corner points |
| Press classifier | CNN on fingertip-centered pixel crops |
| Temporal refinement | BiLSTM / GRU over frame sequences |

### Group A — Teacher / Analysis Only (Training)

Group A uses provided annotations (hand skeleton JSON, keyboard corners, MIDI
TSV) to generate high-quality **teacher labels** for training Group B. Group A
code is **never** used at inference time.

### Why a CNN?

The press/no-press decision is a **visual phenomenon** — subtle changes in
finger posture, nail angle, and skin deformation near the key surface contain
information that pure geometric (x, y) coordinates do not capture. A CNN
operating on pixel crops learns these visual cues directly from data.

---

## Quick Start

### Smoke Test (fast, ~3 videos)

```bash
python run_pipeline.py --mode smoke --step 1
```

### Full Run

```bash
python run_pipeline.py --mode full
```

### Google Colab

Open `notebooks/piano_cv_pipeline.ipynb` and run all cells.

### CLI Parameters

| Flag | Default (smoke) | Default (full) | Description |
|------|-----------------|----------------|-------------|
| `--N` | 3 | 60 | Number of videos |
| `--clip_duration` | 20 | 120 | Clip duration in seconds |
| `--frame_step` | 10 | 5 | Process every Nth frame |
| `--epochs` | 1 | 10 | CNN training epochs |
| `--cache_dir` | `./data/cache` | | Cache for downloaded files |
| `--output_dir` | `./outputs` | | Root output directory |
| `--seed` | 42 | | Random seed |

---

## Project Structure

```
computer-vision/
├── run_pipeline.py          # Single CLI entry point
├── notebooks/
│   └── piano_cv_pipeline.ipynb   # Colab notebook
├── src/
│   ├── __init__.py
│   ├── data.py              # Dataset loading, sampling, splitting, manifest
│   ├── mediapipe_extract.py # Hand landmark extraction (video-only)
│   ├── homography.py        # Keyboard rectification
│   ├── teacher_labels.py    # Group A teacher label generation
│   ├── crops.py             # Fingertip-centered crop extraction
│   ├── cnn.py               # CNN press/no-press classifier
│   ├── bilstm.py            # Temporal refinement model
│   ├── eval.py              # Evaluation metrics & reporting
│   └── viz.py               # Visualization utilities
├── configs/
│   ├── default.yaml
│   └── colab.yaml
├── outputs/                 # Generated outputs (gitignored)
├── data/cache/              # Cached downloads (gitignored)
├── requirements.txt
└── README.md
```

---

## Pipeline Steps

| Step | Description | Key Output |
|------|-------------|------------|
| 1 | Data & Split | manifest.json with video-level train/test split |
| 2 | Group B: CV extraction | MediaPipe landmarks + rectified keyboard coords |
| 3 | Group A: Teacher labels | Frame-aligned press labels for training |
| 4 | CNN training | Press/no-press classifier (pixel-based) |
| 5 | Temporal refinement | BiLSTM smoothing over CNN predictions |

---

## Evaluation

Evaluated on **test videos only** (no video appears in both train and test):

- Precision, Recall, F1
- ROC-AUC
- Confusion matrix
- Event-consistency metric (reduction of isolated single-frame presses)

---

## References

1. Kim et al. (2025) — "PianoVAM: A Multimodal Piano Performance Dataset" — ISMIR 2025
2. Moryossef et al. (2023) — "At Your Fingertips: Extracting Piano Fingering Instructions from Videos"
3. Lee et al. (2019) — "Observing Pianist Accuracy and Form with Computer Vision" — WACV 2019

## License

MIT License — see [LICENSE](LICENSE) for details.
