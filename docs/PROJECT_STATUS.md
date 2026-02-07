# Project Status

**Branch**: `v4`
**Last Updated**: February 2026

---

## Implemented Components

| Component | Files | Status |
|-----------|-------|--------|
| Dataset Loader | `src/data/dataset.py` | ✅ Working — streams metadata from HuggingFace, downloads files on demand |
| MIDI Utilities | `src/data/midi_utils.py` | ✅ Working |
| Video Utilities | `src/data/video_utils.py` | ✅ Working |
| **Auto Keyboard Detector** | `src/keyboard/auto_detector.py` | ✅ **Primary method** — Canny/Hough/clustering, no annotations used |
| Corner-based Detector | `src/keyboard/detector.py` | ✅ Evaluation only — IoU ground truth |
| Key Localization | `src/keyboard/key_localization.py` | ✅ Working — 52 white + 36 black keys |
| Homography | `src/keyboard/homography.py` | ✅ Working |
| Skeleton Loader | `src/hand/skeleton_loader.py` | ✅ Working — parses PianoVAM JSON format |
| Temporal Filter | `src/hand/temporal_filter.py` | ✅ Working — Hampel + interpolation + SavGol |
| Fingertip Extractor | `src/hand/fingertip_extractor.py` | ✅ Working |
| Gaussian Assignment | `src/assignment/gaussian_assignment.py` | ✅ Working — x-only distance, Moryossef et al. (2023) methodology |
| MIDI Sync | `src/assignment/midi_sync.py` | ✅ Working |
| Hand Separation | `src/assignment/hand_separation.py` | ✅ Working — both-hands-try strategy |
| BiLSTM Model | `src/refinement/model.py` | ✅ Code complete |
| Constraints | `src/refinement/constraints.py` | ✅ Code complete |
| Viterbi Decoding | `src/refinement/decoding.py` | ✅ Code complete |
| Training Loop | `src/refinement/train.py` | ✅ Code complete |
| Metrics | `src/evaluation/metrics.py` | ✅ Code complete — IFR + keyboard IoU |
| Visualization | `src/evaluation/visualization.py` | ✅ Code complete |
| Pipeline | `src/pipeline.py` | ✅ Code complete |
| Configuration | `configs/*.yaml` | ✅ Complete |
| Unit Tests | `tests/*.py` | ✅ Complete |
| Notebook | `notebooks/piano_fingering_detection.ipynb` | ✅ End-to-end pipeline |

## Known Limitations

1. **Self-supervised refinement**: The BiLSTM trains on Gaussian assignment outputs as labels, not ground-truth finger annotations. This means it learns to reproduce the baseline rather than improve upon it.

2. **No ground-truth finger labels**: PianoVAM's TSV files contain onset/note/velocity but not per-note finger annotations. Accuracy, M_gen, and M_high cannot be evaluated without ground truth. Only IFR (biomechanical violation rate) is currently measurable.

3. **Evaluation gap**: The evaluation framework (`src/evaluation/metrics.py`) is fully implemented but cannot produce meaningful Accuracy/M_gen/M_high numbers until ground-truth finger labels become available.

## What Works End-to-End

The following **full-CV pipeline** runs successfully on PianoVAM data in Google Colab — **no annotations used during detection**:

1. Load metadata from HuggingFace (no bulk download needed)
2. Download video + skeleton JSON + TSV annotations per sample
3. **Automatically detect keyboard** from video (Canny/Hough/clustering — no corner annotations)
4. Evaluate auto-detection against corner annotations (IoU — **evaluation only**)
5. Load & filter hand landmarks → smooth (T × 21 × 3) arrays
6. Synchronize MIDI events with video frames
7. Assign fingers using Gaussian x-only probability (Moryossef et al. 2023 — both hands tried per note)
8. Train BiLSTM refinement model (self-supervised)
9. Apply constrained Viterbi decoding
10. Evaluate IFR on train and test splits
11. Generate summary visualizations (edges, lines, clusters, black-key contours, IoU bar chart)

---

## Branch History

| Branch | Description |
|--------|-------------|
| `main` | Initial project scaffolding |
| `besn2` | First working notebook (corner-based detection, basic assignment) |
| `besn3` | Full pipeline with BiLSTM refinement, MediaPipe live detection, optical flow |
| `v4` | Cleaned codebase — integrated Canny/Hough auto-detection, single notebook, updated docs |
