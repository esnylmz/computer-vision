# 🎹 Automatic Piano Fingering Detection from Video

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/piano-fingering-detection/blob/main/notebooks/06_full_pipeline.ipynb)

A computer vision system that automatically detects piano fingering (which finger plays each note) from video recordings. Developed as a Master's thesis project at Sapienza University of Rome.

## 🎯 Project Goal

Given a video of piano performance with synchronized MIDI data, automatically determine the finger assignment (1-5, thumb to pinky) for each played note.

**Input**: Video + MIDI → **Output**: Per-note finger labels (L1-L5 for left hand, R1-R5 for right hand)

## 📊 Dataset

This project uses the [PianoVAM dataset](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0):
- 106 piano performances with synchronized video, audio, MIDI
- Pre-extracted 21-keypoint hand skeletons (MediaPipe)
- Multiple skill levels: Beginner, Intermediate, Advanced
- Top-view camera angle (1920×1080, 60fps)

## 🏗️ Pipeline Architecture

```
Video → Keyboard Detection → Hand Processing → Finger-Key Assignment → Neural Refinement → Fingering Labels
         (OpenCV)            (MediaPipe)       (Gaussian Prob.)         (BiLSTM)
```

### Stage 1: Keyboard Detection
Detects piano keys in video frames using edge detection and Hough transforms. Maps 88 keys to pixel coordinates.

### Stage 2: Hand Processing  
Loads pre-extracted MediaPipe hand landmarks, applies temporal filtering (Hampel + Savitzky-Golay), extracts fingertip positions.

### Stage 3: Finger-Key Assignment
Synchronizes MIDI events with video frames. Uses Gaussian probability distribution to assign fingers to pressed keys based on fingertip proximity.

### Stage 4: Neural Refinement (Optional)
BiLSTM model smooths predictions using temporal context and biomechanical constraints.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click the badge above or run notebooks in order:
1. `01_data_exploration.ipynb` - Load and visualize dataset
2. `02_keyboard_detection.ipynb` - Implement keyboard detection
3. `03_hand_processing.ipynb` - Process hand landmarks
4. `04_finger_assignment.ipynb` - Assign fingers to notes
5. `06_full_pipeline.ipynb` - Run complete inference

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/piano-fingering-detection.git
cd piano-fingering-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_dataset.py

# Run pipeline
python -m src.pipeline --config configs/default.yaml --input sample_video.mp4
```

## 📁 Project Structure

```
piano-fingering-detection/
├── notebooks/          # Colab notebooks for each pipeline stage
├── src/                # Source code
│   ├── data/           # Dataset loading
│   ├── keyboard/       # Keyboard detection
│   ├── hand/           # Hand landmark processing
│   ├── assignment/     # Finger-key assignment
│   ├── refinement/     # Neural refinement model
│   └── evaluation/     # Metrics and visualization
├── configs/            # Configuration files
├── tests/              # Unit tests
└── scripts/            # Utility scripts
```

## 📈 Evaluation Metrics

Following standard fingering evaluation protocols:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Exact match rate with ground truth |
| **M_gen** | General match rate (average across annotators) |
| **M_high** | Highest match rate with any annotator |
| **IFR** | Irrational Fingering Rate (impossible transitions) |

## 📚 Key References

1. Moryossef et al. (2023) - "At Your Fingertips: Extracting Piano Fingering Instructions from Videos" - [arXiv](https://arxiv.org/abs/2303.03745)
2. Lee et al. (2019) - "Observing Pianist Accuracy and Form with Computer Vision" - WACV 2019
3. Kim et al. (2025) - "PianoVAM: A Multimodal Piano Performance Dataset" - ISMIR 2025
4. Ramoneda et al. (2022) - "Automatic Piano Fingering from Partially Annotated Scores" - ACM MM 2022

## 🛠️ Technical Details

### Dependencies
- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+
- MediaPipe 0.10+

### Hardware Requirements
- **Minimum**: CPU-only, 8GB RAM (Colab free tier)
- **Recommended**: GPU with 8GB+ VRAM for neural refinement

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- PianoVAM dataset creators (KAIST)
- Sapienza University Computer Vision course
- Referenced paper authors

## 📧 Contact

[Your Name] - [your.email@example.com]

Project Link: [https://github.com/YOUR_USERNAME/piano-fingering-detection](https://github.com/YOUR_USERNAME/piano-fingering-detection)

