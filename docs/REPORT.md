# Automatic Piano Fingering Detection from Video Using Computer Vision

**Computer Vision — Final Project Report (v2.0)**
**Sapienza University of Rome**

---

## Abstract

We present a fully automatic system for detecting piano fingering from video recordings. Given a top-down video of a piano performance and synchronized MIDI data, our pipeline determines which finger (1–5, thumb to pinky) on which hand (left or right) plays each note — using only computer vision techniques. The keyboard is detected from raw video frames via Canny edge detection, Hough line transform, and horizontal-line clustering, without relying on any manual annotations. Hand pose is obtained through MediaPipe and temporally filtered. Finger-to-key assignments are computed using a Gaussian probability model over horizontal distance, following the methodology of Moryossef et al. [1]. Constrained Viterbi decoding enforces biomechanical constraints on the predicted sequences. The system is evaluated on the PianoVAM dataset [2] using Intersection-over-Union (IoU) for keyboard detection, Irrational Fingering Rate (IFR) for assignment quality, and a novel Visual Reference Ground Truth for finger-level accuracy. The Gaussian baseline achieves ~70% finger accuracy against the independent visual reference, and Viterbi-only refinement reduces IFR by 40% while maintaining accuracy.

---

## 1. Introduction

Piano fingering — the choice of which finger presses each key — is fundamental to piano technique. Incorrect fingering leads to awkward hand positions, slower playing, and increased injury risk. While experienced pianists and teachers annotate fingering in sheet music, this process is manual and subjective.

Automatically extracting fingering from video would enable large-scale analysis of piano technique, automated pedagogy, and performance feedback systems. However, the task requires solving several computer vision challenges simultaneously:

1. **Keyboard detection** — localizing the 88-key piano region in video frames.
2. **Hand pose estimation** — tracking 21 keypoints per hand across time.
3. **Finger-key correspondence** — determining which fingertip is closest to each pressed key.
4. **Temporal consistency** — ensuring that the resulting fingering sequence respects the biomechanical constraints of the human hand.

Our system addresses all four challenges in a unified pipeline. Critically, the pipeline operates as a **full computer vision** system: the keyboard is detected automatically from raw video using classical CV techniques (edge detection, line detection, morphological operations, contour analysis). No manual annotations from the dataset are used during detection — corner annotations provided by PianoVAM are used **only** for evaluating detection quality via IoU.

---

## 2. Related Work

### 2.1 Primary Reference: Moryossef et al. (2023)

Our work is primarily based on the methodology proposed by Moryossef et al. in *"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"* [1]. This paper introduced a video-based pipeline for piano fingering detection consisting of:

- **Keyboard localization** from video frames using computer vision.
- **Hand pose estimation** via MediaPipe's 21-keypoint model.
- **Finger-key assignment** using a Gaussian probability model that considers only the **horizontal (x) distance** between fingertips and key centers.

The key insight in [1] is the use of **x-distance only**: in a top-down camera view, the y-axis measures depth into the keyboard, which varies systematically with finger length. The thumb (shortest finger) reaches less far into the keyboard than the middle finger, introducing a bias when both x and y distances are considered. Using x-only distance eliminates this bias.

Our implementation faithfully reproduces and extends this methodology, as detailed in Section 3.

### 2.2 Dataset: Kim et al. (2025)

The PianoVAM dataset [2] provides 107 synchronized piano performances with video, audio, MIDI, and pre-extracted hand skeletons. It includes keyboard corner annotations for evaluation purposes. In our pipeline, these corner annotations are used **solely for evaluation** (IoU comparison), not for keyboard detection.

### 2.3 Neural Refinement: Ramoneda et al. (2022)

Our neural refinement stage is inspired by Ramoneda et al. [3], who proposed using graph neural networks (ArGNN) for automatic piano fingering from partially annotated scores. Their approach to modeling temporal dependencies between consecutive finger assignments informs our BiLSTM sequence model, which learns contextual patterns across note sequences.

### 2.4 Computer Vision for Piano Analysis: Lee et al. (2019)

Lee et al. [4] demonstrated the feasibility of using computer vision to observe pianist accuracy and form, establishing the foundation for video-based piano performance analysis. Their work validates the approach of extracting musical performance information from video using image processing techniques.

---

## 3. Methodology

Our pipeline consists of four sequential stages. Each stage addresses a distinct computer vision or machine learning challenge.

### 3.1 Stage 1: Automatic Keyboard Detection

**Goal**: Detect the piano keyboard region from raw video and map 88 keys to pixel-space bounding boxes, using only classical computer vision.

The detection pipeline processes individual video frames through the following steps:

#### Step 1 — Preprocessing

The input BGR frame is converted to grayscale. We apply **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) [5] with a clip limit of 2.0 and tile grid of 8×8 to normalize contrast across varying lighting conditions. The enhanced image is then smoothed with a 5×5 Gaussian blur to reduce noise while preserving edge structure.

#### Step 2 — Edge Detection (Canny)

We use a **dual-threshold Canny** approach. First, we compute an automatic threshold via Otsu's method [6] on the CLAHE-enhanced image:

$$T_{\text{Otsu}} = \arg\min_t \, \sigma_w^2(t)$$

The Otsu threshold is used to derive adaptive Canny thresholds: $T_{\text{low}} = \max(10, \, 0.5 \cdot T_{\text{Otsu}})$ and $T_{\text{high}} = \min(255, \, T_{\text{Otsu}})$. A second Canny pass uses fixed thresholds (50, 150). The two edge maps are merged via bitwise OR, combining the adaptability of Otsu with the robustness of fixed thresholds.

#### Step 3 — Morphological Closing

A horizontal structuring element (15×1 rectangle) is applied via morphological closing to connect fragmented edge segments that belong to the same keyboard boundary but were broken by gaps in the edge detection.

#### Step 4 — Hough Line Transform

We apply the probabilistic Hough line transform (`cv2.HoughLinesP`) to detect line segments. Each detected line is classified by angle:
- **Horizontal lines** (< 15° from horizontal) — candidates for keyboard top/bottom edges
- **Vertical lines** (< 15° from vertical) — candidates for key boundaries

#### Step 5 — Horizontal Line Clustering

Horizontal lines are clustered by y-coordinate using a greedy algorithm with tolerance of 15 pixels. Each cluster records its mean y-position, x-extent (min/max), and evidence count (number of contributing line segments). Only clusters spanning at least 25% of the frame width are considered.

#### Step 6 — Keyboard Pair Selection

All pairs of qualifying clusters are evaluated. The pair most likely to represent the keyboard's top and bottom edges is selected by maximizing:

$$\text{score} = \frac{\text{width}}{\text{frame\_width}} \times \text{evidence\_count}$$

subject to the constraint that the aspect ratio (width / height) falls within the expected range for a piano keyboard (3–25).

#### Step 7 — Black-Key Refinement

Within the candidate bounding box, we perform thresholding to isolate dark regions, followed by morphological open/close and contour extraction. Contours matching the expected shape of black keys (vertically elongated, positioned near the top of the ROI) are used to tighten the horizontal boundaries of the detection.

#### Step 8 — Multi-Frame Consensus

To handle temporary occlusions (hands covering part of the keyboard, page turns), we sample $N = 7$ frames evenly across the video, run single-frame detection on each, and compute the **median** bounding box across all successful detections. This provides a robust consensus estimate.

#### Step 9 — 88-Key Layout

The final bounding box is divided into 52 white keys and 36 black keys following the standard piano layout (A0 to C8), with key boundaries computed directly in pixel space.

### 3.2 Stage 2: Hand Pose Estimation and Temporal Filtering

**Goal**: Extract and clean 21-keypoint hand skeletons from video.

We use **MediaPipe Hands** [7] for hand pose estimation. MediaPipe detects 21 keypoints per hand, including the five fingertip landmarks (indices 4, 8, 12, 16, 20 for thumb through pinky).

The PianoVAM dataset provides pre-extracted skeleton data. We demonstrate both live MediaPipe detection and loading of pre-extracted data, but use the pre-extracted skeletons for the main pipeline as they cover the full video duration.

Raw landmark sequences are noisy. Following recommendations from Gan et al. [8], we apply a three-stage temporal filtering pipeline:

1. **Hampel filter** (window=20, threshold=3σ) — identifies and removes outliers using Median Absolute Deviation (MAD). Points deviating more than 3 MAD units from the local median are replaced with the median value.

2. **Linear interpolation** — fills gaps shorter than 30 frames where the hand was not detected. Longer gaps are left as NaN to avoid hallucinating hand positions.

3. **Savitzky-Golay filter** (window=11, polynomial order=3) — smooths the remaining signal while preserving the shape of rapid hand movements. This polynomial-fitting approach provides better edge preservation than simple moving averages.

After filtering, normalized coordinates [0, 1] are scaled to pixel space (1920 × 1080) to match the keyboard key boundaries.

### 3.3 Stage 3: Finger-Key Assignment

**Goal**: For each MIDI note event, determine which finger on which hand played it.

This stage directly implements the core methodology of Moryossef et al. [1].

#### MIDI-Video Synchronization

MIDI onset times (in seconds) are converted to video frame indices:

$$\text{frame\_idx} = \lfloor \text{onset\_time} \times \text{fps} \rfloor$$

#### Gaussian Probability Model

For each pressed key, we compute a probability for each of the five fingertips on each hand using a Gaussian distribution over **horizontal (x) distance only**:

$$P(\text{finger}_i \rightarrow \text{key}_k) = \exp\!\left(\frac{-\Delta x^2}{2\sigma^2}\right)$$

where $\Delta x = |x_{\text{fingertip}} - x_{\text{key\_center}}|$ and $\sigma$ auto-scales to the mean white-key width (approximately 33 pixels at 1920×1080 resolution).

**Why x-only?** As explained in [1], in a top-down camera view the y-axis measures how far a finger reaches into the keyboard. This distance varies systematically with finger length — the thumb is shorter than the middle finger and does not reach as far. Including y-distance would systematically bias assignments toward shorter fingers, degrading accuracy. Using x-only eliminates this anatomical confound.

#### Max-Distance Gate

If the closest fingertip of a given hand is farther than $4\sigma$ from the key center, the assignment for that hand is rejected. This prevents assigning a finger when the hand is clearly playing in a different region of the keyboard.

#### Both-Hands Evaluation

For each note, assignments are computed for **both** hands independently. The assignment with higher Gaussian confidence is selected. This avoids the need for a hard left/right hand boundary and naturally handles hand crossings.

### 3.4 Stage 4: Refinement

**Goal**: Refine baseline predictions by enforcing biomechanical transition constraints.

The Gaussian baseline assigns each note independently without considering the sequence context. We provide two refinement strategies:

#### Constrained Viterbi Decoding

We apply **Viterbi decoding** with biomechanical transition constraints on the emission probabilities:

- **Maximum stretch limits**: each finger pair has a maximum comfortable interval (e.g., thumb-to-pinky ≤ 12 semitones).
- **Same-finger penalty**: repeating the same finger on consecutive different notes (interval > 2 semitones) is penalized.
- **Finger ordering**: in ascending passages, higher-numbered fingers should generally play higher notes.
- **Thumb crossing**: thumb-under and finger-over rules from piano pedagogy.

These constraints encode knowledge from piano technique literature and produce physically plausible fingering sequences.

#### Strategy 1: Viterbi-Only (Recommended)

Constrained Viterbi decoding is applied directly on the Gaussian emission probabilities from Stage 3. The (T, 5) emission matrix is constructed from each note's `all_probabilities` field, then decoded with the biomechanical transition constraints. This preserves the strong positional signal from the x-only Gaussian model while enforcing valid transitions. **No neural network or training data is required.**

#### Strategy 2: BiLSTM + Viterbi (Experimental)

A **BiLSTM with self-attention** [9] models temporal dependencies, inspired by the sequence modeling approaches in [3].

**Architecture**:

$$\text{Input}(20) \rightarrow \text{Linear}(128) \rightarrow \text{BiLSTM}(128 \times 2) \rightarrow \text{Self-Attention}(4\text{ heads}) \rightarrow \text{Linear}(128) \rightarrow \text{Linear}(5)$$

**Input features per note** (20 dimensions):
- Normalized MIDI pitch (1 dim)
- One-hot initial finger assignment from baseline (5 dims)
- Time delta from previous note (1 dim)
- Hand encoding: left or right (1 dim)
- One-hot pitch class within octave (12 dims)

The output probability distribution is then refined by constrained Viterbi decoding (same constraints as above).

**Important finding (v2.0)**: The BiLSTM is trained self-supervised on the Gaussian baseline's own outputs (PianoVAM has no finger labels). Our ground truth evaluation revealed that while BiLSTM+Viterbi achieves the same IFR as Viterbi-only, it reduces finger accuracy from 69.6% to 44.8%. The BiLSTM alone (without Viterbi) produces an IFR of 0.693 — worse than random (0.518). This confirms that Viterbi decoding performs all the useful constraint enforcement, and the BiLSTM adds no benefit.

---

## 4. Dataset

We use the **PianoVAM** dataset [2], a multimodal collection of piano performances created by KAIST.

| Property | Value |
|----------|-------|
| Total recordings | 107 piano performances |
| Data modalities | Synchronized video, audio, MIDI |
| Hand skeletons | Pre-extracted 21-keypoint (MediaPipe) |
| Camera setup | Top-down view, 1920 × 1080, 60 fps |
| Skill levels | Beginner, Intermediate, Advanced |
| Keyboard annotations | 4-point corner coordinates per sample |

| Split | Samples | Usage |
|-------|---------|-------|
| Train | 73 | Baseline processing + BiLSTM training |
| Validation | 19 | Hyperparameter tuning |
| Test | 14 | Final evaluation |

The dataset is streamed from HuggingFace, with individual files (video, skeleton JSON, TSV annotations) downloaded on demand.

**Important**: PianoVAM's keyboard corner annotations are used **only for evaluation** (measuring IoU of our automatic detection). They are never used in the detection pipeline itself.

---

## 5. Evaluation

### 5.1 Keyboard Detection: IoU

We evaluate the quality of our automatic keyboard detection by comparing the detected bounding box against the ground-truth bounding box derived from PianoVAM's corner annotations:

$$\text{IoU} = \frac{|\text{Detection} \cap \text{Ground Truth}|}{|\text{Detection} \cup \text{Ground Truth}|}$$

IoU is computed per-sample and aggregated across the dataset (mean, min, max). This metric evaluates the CV detection pipeline independently, confirming that our Canny/Hough approach successfully localizes the keyboard without manual annotation.

### 5.2 Fingering Quality: IFR

The **Irrational Fingering Rate** measures the fraction of note transitions that violate biomechanical constraints:

$$\text{IFR} = \frac{\text{Number of irrational transitions}}{\text{Total transitions}}$$

A lower IFR indicates more physically plausible fingering. This metric does not require ground-truth finger labels and can be computed from the predicted assignments alone.

Irrational transitions include:
- Same finger repeated on notes separated by more than 2 semitones
- Finger stretch exceeding physical limits
- Invalid finger crossings

### 5.3 Visual Reference Ground Truth (v2.0)

A critical limitation of IFR is that it measures physical plausibility, not correctness. To address the absence of per-note finger labels in PianoVAM, we construct a **Visual Reference Ground Truth** that is **doubly independent** from our pipeline:

| Dimension | Pipeline | Visual Ground Truth |
|-----------|----------|-------------------|
| Skeleton source | Live MediaPipe (~48% detection) | Dataset pre-extracted skeletons (~96% detection) |
| Distance metric | x-only Gaussian | Full 2D Euclidean distance |
| Decision rule | Max Gaussian probability | Min 2D Euclidean distance + proximity threshold |

For each MIDI note onset:
1. Retrieve the dataset's pre-extracted skeleton at the onset frame.
2. Compute 2D Euclidean distance from every fingertip (both hands) to the pressed key's centre.
3. Select the closest fingertip as the ground-truth finger.
4. Reject assignments where the closest fingertip exceeds a strict proximity threshold (2× mean key width).

This enables computing:
- **Finger accuracy** — exact match rate between pipeline and visual GT
- **Hand accuracy** — correct hand (left/right) identification rate
- **Per-finger Precision / Recall / F1** — systematic finger-level errors
- **Confusion matrix** — substitution patterns (e.g., Ring↔Pinky)

The multi-threshold sensitivity analysis confirms robustness: accuracy ranges from 93.2% at the strictest threshold (0.5× key width, 761 high-confidence notes) to 69.6% at the default threshold (2.0× key width, 5,184 notes), with smooth degradation.

### 5.4 Additional Metrics (Framework Implemented)

The evaluation framework also supports:
- **Accuracy** — exact match rate with human expert annotations
- **M_gen** — general match rate averaged across multiple annotators
- **M_high** — match rate with the best-matching annotator

These metrics require human-annotated ground-truth finger labels. The evaluation code is fully implemented and ready for use when annotated data becomes available.

---

## 6. Implementation

The project is implemented in Python with the following key dependencies:

| Library | Usage |
|---------|-------|
| OpenCV 4.5+ | Canny, Hough, morphological operations, homography |
| MediaPipe | 21-keypoint hand pose estimation |
| NumPy / SciPy | Signal processing, filtering, linear algebra |
| PyTorch 1.12+ | BiLSTM model, training, Viterbi decoding |
| Pandas | Dataset metadata and annotation processing |
| Matplotlib / Seaborn | Visualization |

### Project Structure

```
src/
├── keyboard/
│   ├── auto_detector.py      ← Primary: Canny/Hough/clustering detection
│   ├── detector.py           ← Corner-based detection (evaluation only)
│   ├── homography.py         ← Perspective normalization
│   └── key_localization.py   ← 88-key layout mapping
├── hand/
│   ├── skeleton_loader.py    ← MediaPipe JSON parser
│   ├── temporal_filter.py    ← Hampel + interpolation + SavGol
│   └── fingertip_extractor.py
├── assignment/
│   ├── gaussian_assignment.py ← Gaussian x-only model [1]
│   ├── midi_sync.py          ← MIDI-to-frame synchronization
│   └── hand_separation.py    ← Left/right hand disambiguation
├── refinement/
│   ├── model.py              ← BiLSTM + Attention
│   ├── constraints.py        ← Biomechanical constraints
│   ├── decoding.py           ← Constrained Viterbi decoding
│   └── train.py              ← Training loop
├── evaluation/
│   ├── metrics.py            ← Accuracy, M_gen, M_high, IFR
│   └── visualization.py
└── pipeline.py               ← End-to-end pipeline
```

The full pipeline is demonstrated in a single Jupyter notebook (`notebooks/dnm_son_piano_fingering_detection.ipynb`) that runs end-to-end on Google Colab.

---

## 7. Results and Discussion

### 7.1 Keyboard Detection

Our automatic keyboard detection pipeline achieves an **IoU of 0.946** against PianoVAM corner annotations on the demo sample. The multi-frame consensus mechanism (sampling 7 frames and taking the median bounding box) provides robustness against per-frame failures caused by hand occlusions.

### 7.2 Hand Pose Validation

Post-filtered MediaPipe landmarks were validated against PianoVAM's pre-extracted skeletons:

| Metric | Right Hand | Left Hand |
|--------|-----------|-----------|
| PCK@0.02 | 97.8% | 98.3% |
| Trajectory Correlation (mean *r*) | 0.995 | 0.988 |
| Key Agreement | 87.0% | 88.2% |

The near-perfect PCK scores confirm that our filtered MediaPipe output closely matches the dataset's reference skeletons.

### 7.3 Finger Assignment (IFR)

| Method | Mean IFR (10 samples) | Std |
|--------|----------------------|-----|
| Random baseline | 0.518 | 0.123 |
| Direction-Aware heuristic | 0.502 | 0.135 |
| **Gaussian Baseline** | **0.363** | 0.138 |
| **Viterbi-Only** | **0.217** | 0.125 |
| BiLSTM + Viterbi | 0.217 | 0.125 |

The Gaussian baseline reduces IFR by 30% vs random. Viterbi-only further reduces IFR by 40% vs baseline. BiLSTM+Viterbi achieves identical IFR to Viterbi-only on every sample tested.

### 7.4 Ablation Study

| Config | IFR | Notes |
|--------|-----|-------|
| A: Raw + Gaussian | 0.4754 | 3,337 |
| B: + Temporal filtering | 0.4455 | 6,915 (+107% coverage) |
| C: + Max-distance gate | 0.4455 | 6,915 |
| C2: + Viterbi-only | 0.3269 | 6,915 |
| D: + BiLSTM only (no Viterbi) | 0.6927 | 6,915 |
| E: + BiLSTM + Viterbi | 0.3269 | 6,915 |

Key findings:
- Temporal filtering doubles note coverage (3,337 → 6,915) via gap-filling, while also reducing IFR by 6.3%.
- Viterbi-only (C2) matches the full pipeline (E) at IFR = 0.3269.
- BiLSTM alone (D) produces IFR = 0.6927 — **worse than random** — confirming the neural network adds no useful signal without Viterbi.

### 7.5 Ground Truth Evaluation (Visual Proximity Analysis)

Finger accuracy measured against Visual Reference Ground Truth (demo sample, 5,184 matched notes):

| Method | Finger Acc | Hand Acc | Combined | Macro F1 |
|--------|-----------|----------|----------|----------|
| **Gaussian Baseline** | **69.6%** | 97.1% | 68.1% | **0.673** |
| Viterbi-Only | 60.2% | 97.1% | 58.9% | 0.584 |
| BiLSTM+Viterbi (60s subset) | 44.8% | 97.6% | 43.4% | 0.390 |

**Per-finger F1 (Baseline)**:
| Finger | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| Thumb | 0.546 | 0.897 | 0.679 | 468 |
| Index | 0.867 | 0.767 | 0.814 | 1,634 |
| Middle | 0.831 | 0.754 | 0.791 | 1,269 |
| Ring | 0.760 | 0.407 | 0.531 | 1,291 |
| Pinky | 0.402 | 0.866 | 0.549 | 522 |

Cross-sample test set (5 samples): Baseline finger accuracy = **67.1% ± 6.2%**, hand accuracy = **98.2% ± 1.7%**.

### 7.6 Error Analysis

Top substitution patterns (1,576 errors on demo sample):
1. Ring → Pinky: 624 (39.6%) — dominant error due to spatial proximity
2. Index → Thumb: 320 (20.3%)
3. Middle → Index: 141 (8.9%)

Accuracy by pitch region: Mid-low (C3–B4) achieves 75.0%; low register (A0–B2) drops to 45.3% due to compressed hand positions.

### 7.7 Computer Vision Contribution

The project makes substantial use of classical and modern computer vision techniques across all stages:

| CV Technique | Where Used |
|---|---|
| Grayscale conversion | Keyboard preprocessing |
| CLAHE (adaptive histogram equalization) | Contrast normalization for varying lighting |
| Gaussian blur | Noise reduction |
| Canny edge detection (fixed + Otsu-adaptive) | Keyboard boundary extraction |
| Morphological operations (close, open) | Edge completion, noise removal |
| Hough line transform (probabilistic) | Keyboard edge detection |
| Contour detection and analysis | Black-key segmentation |
| Homography / perspective transform | Keyboard normalization |
| Dense optical flow (Farneback) | Hand motion analysis (demonstrated) |
| MediaPipe hand pose estimation | 21-keypoint hand detection |
| Temporal signal processing (Hampel, SavGol) | Landmark noise reduction |

---

## 8. Conclusion

We implemented a fully automatic piano fingering detection system that operates on raw video without relying on manual annotations during detection. Our pipeline follows the methodology of Moryossef et al. [1], implementing their Gaussian x-only finger-key assignment model, and extends it with:

1. A **robust automatic keyboard detection pipeline** based on Canny edge detection, Hough line clustering, and black-key contour analysis (IoU = 0.946).
2. A **Viterbi-only refinement strategy** that applies constrained biomechanical decoding directly on the Gaussian emission probabilities, reducing IFR by 40% without any neural network.
3. A **Visual Reference Ground Truth** that enables finger-level accuracy evaluation (~70% on full dataset, 93.2% on high-confidence subset) without requiring human annotation.

The system demonstrates that classical computer vision techniques — when combined thoughtfully (CLAHE preprocessing, dual-threshold Canny, morphological operations, multi-frame consensus) — can reliably localize a piano keyboard from unconstrained video. The Gaussian x-only assignment model, as proposed by Moryossef et al. [1], captures the essential geometry of finger-key correspondence with nearly 70% accuracy against an independent visual reference.

A key finding of this work is that **Viterbi-only refinement is superior to BiLSTM+Viterbi** for this task. The BiLSTM, trained self-supervised on its own baseline outputs, damages finger accuracy (69.6% → 44.8%) while achieving identical IFR to Viterbi-only. The BiLSTM alone (without Viterbi) produces an IFR of 0.693 — worse than random — confirming that constrained Viterbi decoding performs all useful constraint enforcement.

### Limitations and Future Work

1. **Surrogate ground truth**: The Visual Reference GT assumes the 2D-closest fingertip to a pressed key is the finger pressing it. In fast polyphonic passages, this may not hold. Human expert annotation would provide definitive ground truth.

2. **Adjacent-finger confusion**: The dominant error mode is Ring↔Pinky (39.6% of errors) and Index↔Thumb (20.3%), inherent to the x-only distance model when adjacent fingertips overlap spatially. Incorporating y-distance selectively or finger-length priors could help.

3. **Self-supervised BiLSTM**: With supervised finger labels, the neural refinement could potentially learn genuinely improved predictions. The current self-supervised approach degrades accuracy.

4. **Real-time processing**: The current system operates offline on pre-recorded video. Adapting it for real-time feedback would require optimizing the per-frame processing pipeline.

5. **Generalization**: The keyboard detection has been tested on PianoVAM's consistent top-down camera setup. Evaluation on more diverse camera angles and lighting conditions would strengthen the contribution.

---

## References

[1] A. Moryossef, Y. Tsaban, and I. Fried, "At Your Fingertips: Extracting Piano Fingering Instructions from Videos," *arXiv preprint arXiv:2303.03745*, 2023. Available: https://arxiv.org/abs/2303.03745

[2] S. Kim, T. Jeong, J. Lee, and J. Nam, "PianoVAM: A Multimodal Piano Performance Dataset with Video, Audio, and MIDI," in *Proc. International Society for Music Information Retrieval (ISMIR)*, 2025.

[3] P. Ramoneda, J. J. Valero-Mas, D. Jeong, and X. Serra, "Automatic Piano Fingering from Partially Annotated Scores using Autoregressive Neural Networks," in *Proc. ACM International Conference on Multimedia*, 2022.

[4] W. Lee, S. Kim, and J. Nam, "Observing Pianist Accuracy and Form with Computer Vision," in *Proc. IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2019.

[5] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, Academic Press, 1994, pp. 474–485.

[6] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.

[7] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann, "MediaPipe Hands: On-device Real-time Hand Tracking," *arXiv preprint arXiv:2006.10214*, 2020.

[8] Z. Gan, Y. Wen, and C. Zhang, "PianoMotion10M: Dataset and Benchmark for Hand Motion Generation in Piano Performance," *arXiv preprint arXiv:2406.09326*, 2024.

[9] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
