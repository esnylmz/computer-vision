# Piano Fingering Detection Project - Status Report

**Repository**: [github.com/esnylmz/computer-vision](https://github.com/esnylmz/computer-vision)

---

## 1. Project Overview

### 1.1 Problem Statement

Piano fingering—determining which finger (1-5, thumb to pinky) plays each note—is crucial for piano education and performance analysis. Manual annotation is time-consuming and requires expert knowledge. This project aims to **automatically detect piano fingering from video recordings** using computer vision techniques.

### 1.2 Input/Output Specification

| Input | Output |
|-------|--------|
| Video of piano performance (1920×1080, 60fps) | Per-note finger labels |
| Synchronized MIDI data (note events) | Hand assignment (L/R) |
| Pre-extracted hand skeleton (MediaPipe 21-keypoint) | Confidence scores |

**Example Output**: For each played note → `R3` (right hand, middle finger) or `L1` (left hand, thumb)

### 1.3 Dataset

We use the **PianoVAM dataset** from KAIST, available on [HuggingFace](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0):

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 73 | Model training (Stage 4) |
| Validation | 19 | Hyperparameter tuning |
| Test | 14 | Final evaluation |

**Dataset Contents**:
- 🎬 **Video**: MP4 files, top-view camera angle
- 🎵 **MIDI**: Ground truth note events from Yamaha Disklavier
- 🖐️ **Hand Skeleton**: Pre-extracted 21-keypoint landmarks (JSON)
- 📊 **TSV Annotations**: Note timing with frame offsets
- 📐 **Keyboard Corners**: 4-point annotations for keyboard localization

---

## 2. Our Solution

### 2.1 Pipeline Architecture

We implement a **4-stage pipeline** inspired by recent research (Moryossef et al. 2023, Ramoneda et al. 2022):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINGERING DETECTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐      │
│   │  STAGE 1   │    │  STAGE 2   │    │  STAGE 3   │    │  STAGE 4   │      │
│   │  Keyboard  │───▶│    Hand    │───▶│   Finger   │───▶│   Neural   │    │
│   │ Detection  │    │ Processing │    │ Assignment │    │ Refinement │      │
│   └────────────┘    └────────────┘    └────────────┘    └────────────┘      │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│   Key boundaries    Filtered         Gaussian prob.     Refined            │
│   (88 keys)         landmarks        assignments        predictions        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Stage Details

#### Stage 1: Keyboard Detection (`src/keyboard/`)
- **Method**: Use corner annotations from PianoVAM (or Canny + Hough for new videos)
- **Output**: Homography matrix + 88 key bounding boxes
- **Status**: ✅ Implemented

#### Stage 2: Hand Processing (`src/hand/`)
- **Method**: 
  1. Load MediaPipe 21-keypoint JSON
  2. Hampel filter for outlier removal (window=20)
  3. Linear interpolation for gaps < 30 frames
  4. Savitzky-Golay smoothing (window=11, order=3)
- **Output**: Filtered landmark arrays (T × 21 × 3)
- **Status**: ✅ Implemented

#### Stage 3: Finger Assignment (`src/assignment/`)
- **Method**: Gaussian probability model
  ```
  P(finger_i → key_k) = exp(-distance² / 2σ²)
  ```
  - Compute distance from each fingertip to key center
  - Assign finger with maximum probability
  - Separate hands by x-position
- **Output**: FingerAssignment objects with confidence
- **Status**: ✅ Implemented

#### Stage 4: Neural Refinement (`src/refinement/`)
- **Method**: BiLSTM with attention
  - Input: (pitch, initial_finger, time_delta, hand)
  - Architecture: Embedding → BiLSTM(128) × 2 → Attention → Dense(5)
  - Constraints: Biomechanical penalties in loss function
- **Output**: Refined finger predictions
- **Status**: ⚠️ Implemented (needs training)

### 2.3 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Use pre-extracted skeletons | PianoVAM provides MediaPipe landmarks, saving computation |
| Gaussian assignment | Simple, interpretable, no training required for baseline |
| BiLSTM refinement | Captures temporal context, proven effective in music tasks |
| Colab-first design | Accessibility for thesis evaluation, no local GPU needed |

---

## 3. Current Progress

### 3.1 Completed Components

| Component | File(s) | Status |
|-----------|---------|--------|
| **Project Structure** | All directories | ✅ Complete |
| **Configuration** | `configs/*.yaml` | ✅ Complete |
| **Dataset Loader** | `src/data/dataset.py` | ✅ Complete |
| **MIDI Utilities** | `src/data/midi_utils.py` | ✅ Complete |
| **Video Utilities** | `src/data/video_utils.py` | ✅ Complete |
| **Keyboard Detector** | `src/keyboard/detector.py` | ✅ Complete |
| **Key Localization** | `src/keyboard/key_localization.py` | ✅ Complete |
| **Homography** | `src/keyboard/homography.py` | ✅ Complete |
| **Skeleton Loader** | `src/hand/skeleton_loader.py` | ✅ Complete |
| **Temporal Filter** | `src/hand/temporal_filter.py` | ✅ Complete |
| **Fingertip Extractor** | `src/hand/fingertip_extractor.py` | ✅ Complete |
| **Gaussian Assignment** | `src/assignment/gaussian_assignment.py` | ✅ Complete |
| **MIDI Sync** | `src/assignment/midi_sync.py` | ✅ Complete |
| **Hand Separation** | `src/assignment/hand_separation.py` | ✅ Complete |
| **BiLSTM Model** | `src/refinement/model.py` | ✅ Complete |
| **Constraints** | `src/refinement/constraints.py` | ✅ Complete |
| **Training Loop** | `src/refinement/train.py` | ✅ Complete |
| **Metrics** | `src/evaluation/metrics.py` | ✅ Complete |
| **Visualization** | `src/evaluation/visualization.py` | ✅ Complete |
| **Pipeline** | `src/pipeline.py` | ✅ Complete |
| **Unit Tests** | `tests/*.py` | ✅ Complete |
| **Scripts** | `scripts/*.py` | ✅ Complete |
| **Documentation** | `docs/*.md` | ✅ Complete |

### 3.2 Notebooks Status

| Notebook | Purpose | Status |
|----------|---------|--------|
| `01_data_exploration.ipynb` | Dataset loading & visualization | ⚠️ Started (needs completion) |
| `02_keyboard_detection.ipynb` | Keyboard detection implementation | ❌ Not created |
| `03_hand_processing.ipynb` | Hand skeleton processing | ❌ Not created |
| `04_finger_assignment.ipynb` | Finger-key assignment | ❌ Not created |
| `05_neural_refinement.ipynb` | BiLSTM training | ❌ Not created |
| `06_full_pipeline.ipynb` | End-to-end inference | ❌ Not created |
| `07_evaluation.ipynb` | Metrics & analysis | ❌ Not created |

### 3.3 Testing Status

```bash
# Run tests
cd piano-fingering-detection
pytest tests/ -v
```

All unit tests are implemented for:
- Keyboard detection (`test_keyboard.py`)
- Hand processing (`test_hand.py`)
- Finger assignment (`test_assignment.py`)

---

## 4. Next Steps

### 4.1 Immediate Tasks (Week 1)

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Complete `01_data_exploration.ipynb` | 🔴 High | 2-3 hours |
| Create `02_keyboard_detection.ipynb` | 🔴 High | 2-3 hours |
| Create `03_hand_processing.ipynb` | 🔴 High | 2-3 hours |
| Test data loading in Colab | 🔴 High | 1 hour |
| Verify PianoVAM JSON format | 🔴 High | 1 hour |

### 4.2 Core Implementation (Week 2)

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Create `04_finger_assignment.ipynb` | 🔴 High | 3-4 hours |
| Create `06_full_pipeline.ipynb` | 🔴 High | 3-4 hours |
| Run baseline (Gaussian only) on test set | 🔴 High | 2 hours |
| Debug and fix issues | 🟡 Medium | 4-6 hours |

### 4.3 Neural Refinement (Week 3)

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Create training data from assignments | 🟡 Medium | 3 hours |
| Create `05_neural_refinement.ipynb` | 🟡 Medium | 4 hours |
| Train BiLSTM model | 🟡 Medium | 4-6 hours |
| Hyperparameter tuning | 🟡 Medium | 3 hours |

### 4.4 Evaluation & Documentation (Week 4)

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Create `07_evaluation.ipynb` | 🔴 High | 3 hours |
| Compute final metrics on test set | 🔴 High | 2 hours |
| Error analysis & failure cases | 🟡 Medium | 3 hours |
| Write thesis chapter on implementation | 🔴 High | 8+ hours |
| Create demo video | 🟢 Low | 2 hours |

### 4.5 Known Issues to Address

1. **Skeleton JSON Format**: Need to verify exact structure of PianoVAM hand skeleton files
2. **Coordinate Normalization**: MediaPipe outputs [0,1] normalized coordinates; verify alignment with keyboard
3. **Left/Right Hand Detection**: May need to swap hands based on skeleton file structure
4. **MIDI-Video Sync**: TSV files provide frame offsets; ensure correct usage
5. **Thumb Occlusion**: Thumb-under passages cause severe occlusion; consider temporal interpolation

---

## 5. Expected Results

### 5.1 Baseline Performance (Gaussian Assignment Only)

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Accuracy | 70-75% | Without any learning |
| M_gen | 65-70% | Match with any annotator |
| M_high | 75-80% | Best annotator match |
| IFR | 5-10% | Biomechanical violations |

### 5.2 With Neural Refinement

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Accuracy | 80-85% | With BiLSTM |
| M_gen | 75-80% | Improved consistency |
| M_high | 85-90% | Close to human |
| IFR | 2-5% | Constraints enforced |

### 5.3 Comparison to Prior Work

| Method | Accuracy | Dataset |
|--------|----------|---------|
| Moryossef et al. (2023) | 78% | YouTube videos |
| Ramoneda et al. (2022) | 85% | PIG (symbolic only) |
| **Our Target** | **80-85%** | **PianoVAM** |

---

## 6. Repository Structure

```
piano-fingering-detection/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default parameters
│   └── colab.yaml             # Colab-optimized settings
├── docs/                       # Documentation
│   ├── PIPELINE.md            # Pipeline details
│   ├── EVALUATION.md          # Metrics guide
│   └── PROJECT_STATUS.md      # This file
├── notebooks/                  # Colab notebooks
│   └── 01_data_exploration.ipynb
├── scripts/                    # Utility scripts
│   ├── download_dataset.py    # Download PianoVAM
│   └── preprocess_all.py      # Batch preprocessing
├── src/                        # Source code
│   ├── data/                  # Data loading
│   ├── keyboard/              # Stage 1
│   ├── hand/                  # Stage 2
│   ├── assignment/            # Stage 3
│   ├── refinement/            # Stage 4
│   ├── evaluation/            # Metrics
│   ├── utils/                 # Utilities
│   └── pipeline.py            # Main pipeline
├── tests/                      # Unit tests
├── outputs/                    # Generated files (gitignored)
├── README.md
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## 7. How to Run

### 7.1 In Google Colab

```python
# Clone repository
!git clone https://github.com/esnylmz/computer-vision.git
%cd computer-vision/piano-fingering-detection

# Install dependencies
!pip install -e .

# Open notebooks in order
# Start with 01_data_exploration.ipynb
```

### 7.2 Locally

```bash
# Clone and setup
git clone https://github.com/esnylmz/computer-vision.git
cd computer-vision/piano-fingering-detection
pip install -e .

# Download dataset
python scripts/download_dataset.py --output_dir ./data

# Run tests
pytest tests/ -v

# Run pipeline (after completing notebooks)
python -m src.pipeline --config configs/default.yaml
```

---

## 8. Contact & Resources

- **Repository**: [github.com/esnylmz/computer-vision](https://github.com/esnylmz/computer-vision)
- **Dataset**: [PianoVAM on HuggingFace](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0)
- **Key Paper**: [At Your Fingertips (arXiv:2303.03745)](https://arxiv.org/abs/2303.03745)

---

*Last Updated: January 2026*

