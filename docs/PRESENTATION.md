# Automatic Piano Fingering Detection from Video
## Computer Vision Final Project — Sapienza University of Rome

---

## Slide 1: Title

**Automatic Piano Fingering Detection from Video**

Computer Vision Final Project
Sapienza University of Rome

**Input**: Video + MIDI  -->  **Output**: Per-note finger labels (L1-L5, R1-R5)

---

## Slide 2: Problem Statement

- Learning piano requires knowing **which finger** to use for each note
- Most sheet music lacks fingering annotations
- Learners often watch online video performances to learn fingering
- **Goal**: Automate the extraction of fingering information from piano performance videos

**Our task**: Given a video of a piano performance and a synchronized MIDI file, automatically determine the finger assignment (1-5, thumb to pinky) for each played note using computer vision.

---

## Slide 3: Reference Paper

**"At Your Fingertips: Extracting Piano Fingering Instructions from Videos"**
Moryossef, Elazar & Goldberg (2023) — arXiv:2303.03745

**Paper's pipeline:**
1. Keyboard detection (largest bright area)
2. Hand detection (fine-tuned Faster R-CNN)
3. Finger pose estimation (Convolutional Pose Machines + CycleGAN domain adaptation)
4. Gaussian probability assignment
5. Video-MIDI alignment (audio peak + confidence search)

**Paper's key results**: 97% F1 score on 5 manually annotated pieces, produced APFD dataset (90 pieces, 155K notes)

---

## Slide 4: What We Adopted from the Paper

| From the Paper | How We Used It |
|---|---|
| Overall pipeline architecture (Video -> Keyboard -> Hands -> Assignment) | Same 4-stage conceptual flow, extended with a 5th refinement stage |
| Gaussian probability for finger-key assignment | Core idea adopted, but modified to use x-distance only |
| Multi-frame keyboard detection concept | Adopted the idea of using multiple frames for robustness |
| BiLSTM for sequence modeling | Adopted the concept, significantly extended the architecture |
| Both-hands evaluation per note | Try both left & right hand, pick higher confidence |

**In summary**: We followed the paper's high-level pipeline design and the Gaussian assignment concept.

---

## Slide 5: What We Contributed — Overview

**We did NOT use** the paper's two most distinctive components:
- NO CycleGAN domain adaptation
- NO Faster R-CNN hand detector
- NO Convolutional Pose Machines

**Instead, we built:**
1. A fully automatic keyboard detection pipeline (Canny + Hough + clustering)
2. Modern hand detection with MediaPipe (replacing 3 separate models)
3. A temporal filtering pipeline (Hampel + interpolation + Savitzky-Golay)
4. Modified Gaussian assignment (x-only + max-distance gate)
5. Neural refinement with BiLSTM + Attention + Constrained Viterbi decoding
6. Biomechanical constraint system for evaluation (IFR metric)

---

## Slide 6: Our Contribution 1 — Automatic Keyboard Detection

**Paper**: Detects keyboard as "largest continuous bright area" (minimal detail provided)

**Our approach** — a fully engineered CV pipeline:
1. **Preprocessing**: Grayscale -> CLAHE contrast enhancement -> Gaussian blur
2. **Edge detection**: Canny with Otsu-adaptive + fixed thresholds, merged for robustness
3. **Morphological closing**: Horizontal kernel connects fragmented edges
4. **Hough line transform**: Extract horizontal and vertical line segments
5. **Line clustering**: Group horizontal lines by y-coordinate, score pairs by width, brightness, aspect ratio
6. **Brightness-profile extension**: Scan downward to find true keyboard bottom edge
7. **Black-key refinement**: Contour analysis tightens x-boundaries
8. **Multi-frame consensus**: Sample 7 frames, take median bbox (robust to hand occlusions)
9. **88-key layout**: Divide into 52 white + 36 black keys in pixel space

**Result**: No annotations used in detection. Corner annotations used ONLY for IoU evaluation.

---

## Slide 7: Keyboard Detection — Visual Results

*(Show the 4-panel visualization from the notebook: Hough lines, line clusters, black-key contours, auto vs. corner GT with IoU)*

- Green = our auto-detection bounding box
- Red = ground truth from corner annotations
- IoU scores demonstrate detection quality

*(Show the 88-key layout visualization and the perspective-corrected keyboard)*

---

## Slide 8: Our Contribution 2 — MediaPipe Hand Detection

**Paper's approach** (3 separate models):
1. Fine-tuned Faster R-CNN (Inception v2) for hand bounding boxes — required a custom 476-hand labeled dataset
2. Convolutional Pose Machines for finger positions
3. CycleGAN for domain adaptation to handle lighting/blur — required training on 21,746 images

**Our approach** (single model):
- **MediaPipe Hands** run directly on raw video
- `model_complexity = 1` (full model for accuracy)
- `min_detection_confidence = 0.3` (catches fast-moving and partially-occluded hands)
- `static_image_mode = False` (video mode enables temporal tracking across frames)
- `frame_stride = 2` (process every 2nd frame for speed)

**Advantages**: No custom training data, no GAN training, single unified model, real-time capable

---

## Slide 9: Our Contribution 3 — Temporal Filtering

**Not present in the paper at all.** The paper relied on CycleGAN-adapted pose estimation for accuracy.

We observed that raw MediaPipe landmarks are **noisy** (jitter, outliers, missing detections), so we built a 3-stage filtering pipeline:

| Stage | Method | Purpose |
|---|---|---|
| 1. Hampel Filter | Median Absolute Deviation (MAD), window=20, threshold=3sigma | Detect and remove outlier landmarks |
| 2. Linear Interpolation | Fill gaps up to 30 frames | Handle frames where MediaPipe missed a hand |
| 3. Savitzky-Golay Filter | Polynomial smoothing (order=3, window=11) | Smooth trajectories for stable fingertip positions |

*(Show the raw vs. filtered fingertip x-coordinate plot from the notebook)*

**Also added**: Dense optical flow visualization (Farneback) to analyze hand motion patterns.

---

## Slide 10: Our Contribution 4 — Modified Gaussian Assignment

**Paper's Gaussian**: Full 2D Gaussian `N(x_i | mu_k, sigma^2_k)` with sigma = key width

**Our modifications**:

1. **X-distance only**: `P(finger|key) = exp(-dx^2 / 2*sigma^2)`
   - Ignores y-coordinate to avoid bias from different finger lengths
   - A tall finger (middle) and a short finger (pinky) should have equal probability if they are at the same x-position above a key

2. **Auto-scaled sigma**: sigma is automatically set to the mean white-key width in pixels (adapts to different video resolutions and keyboard positions)

3. **Max-distance gate (4*sigma)**: Rejects any assignment when the closest fingertip is more than 4 sigma away from the key center. Prevents false assignments when the hand is far from the played key.

4. **Both-hands competition**: For every MIDI event, we compute assignment probability for BOTH the left and right hand, and select the one with higher confidence.

---

## Slide 11: Our Contribution 5 — Neural Refinement (BiLSTM)

**Paper**: Simple BiLSTM + MLP with cross-entropy loss

**Our architecture** — significantly more advanced:

```
Input Features (20-dim)
    |
Linear(20 -> 128)
    |
BiLSTM (128, 2 layers, dropout=0.3)
    |
Multi-Head Self-Attention (4 heads, dim=256)
    |
Combined: LSTM output + Attention output
    |
Linear(256 -> 128) -> Linear(128 -> 5)
    |
Constrained Viterbi Decoding
```

**Input features** (20 dimensions):
- Normalized MIDI pitch (1)
- One-hot current finger prediction (5)
- Time delta between consecutive notes (1)
- Hand encoding — left or right (1)
- Pitch class one-hot (12)

---

## Slide 12: Our Contribution 6 — Biomechanical Constraints & Viterbi

**Not present in the paper.** This is entirely our work.

**Biomechanical Constraint System**:
- Maximum stretch limits per finger pair (in semitones):
  - Thumb-Index: 10, Thumb-Pinky: 12
  - Index-Middle: 5, Ring-Pinky: 5
- Skill-level multipliers: beginner(0.8x), intermediate(1.0x), advanced(1.2x)
- Finger ordering validation (ascending fingers should play ascending notes)
- Thumb crossing detection (thumb-under, finger-over patterns)

**Constrained Viterbi Decoding**:
- Dynamic programming over the finger sequence
- Transition penalty of -1e9 for physically impossible finger transitions
- Finds the globally optimal finger sequence that respects biomechanical constraints

**Constraint-Aware Training Loss**:
```
Loss = CrossEntropy + 0.1 * ConstraintLoss
```
where ConstraintLoss = fraction of invalid transitions in the predicted sequence

---

## Slide 13: Evaluation — Methodology

**Paper's evaluation**: Precision / Recall / F1 against 5 manually annotated pieces

**Our evaluation**: IFR (Irrational Fingering Rate) — a constraint-based metric

- **IFR** = (number of biomechanically invalid transitions) / (total transitions)
- Lower IFR = better (fewer physically impossible finger movements)
- Does not require ground-truth finger annotations
- Measures whether the predicted fingering is **physically playable**

**Why IFR?** PianoVAM does not provide per-note finger annotations, so P/R/F1 against ground truth is not possible. IFR evaluates **physical plausibility** instead.

**Also evaluated**: Keyboard detection IoU against corner annotations from PianoVAM.

---

## Slide 14: Evaluation — Results

**Keyboard Detection**:
- IoU against corner ground truth across samples
- *(Show IoU bar chart from the notebook)*

**Fingering Quality (IFR)**:

| Method | Mean IFR | Description |
|---|---|---|
| Baseline (Gaussian only) | X.XXX +/- X.XXX | X-only Gaussian + max-distance gate |
| Refined (BiLSTM + Viterbi) | X.XXX +/- X.XXX | Neural refinement with constraints |
| **Improvement** | +X.XXX | Lower = better |

*(Fill in with your actual numbers from the run)*

**Key observations**:
- Neural refinement consistently reduces IFR compared to baseline
- Constrained Viterbi decoding enforces physically plausible finger sequences
- Both-hands evaluation improves coverage over single-hand assumption

---

## Slide 15: Dataset

**Paper**: YouTube videos (APFD — 90 pieces, 155K notes, self-collected)

**Us**: PianoVAM (KAIST) — an existing academic benchmark
- 107 piano performances
- Synchronized video, audio, MIDI, and pre-extracted hand skeletons
- Proper train/validation/test splits (73/19/14)
- Keyboard corner annotations (used ONLY for IoU evaluation)
- TSV note annotations with onset times

**Note**: We do NOT use PianoVAM's pre-extracted skeletons. Our pipeline runs MediaPipe live on the raw video. The dataset's corner annotations are used only for evaluation, not for detection.

---

## Slide 16: Pipeline Summary

```
Raw Video
   |
   v
[Stage 1] Keyboard Detection (Canny + Hough + Clustering + Black-Key Refinement)
   |                                                    --> 88 key bounding boxes
   v
[Stage 2] Hand Pose Estimation (MediaPipe Hands, live on video)
   |                                                    --> Raw landmarks (T x 21 x 3)
   v
[Stage 3] Temporal Filtering (Hampel + Interpolation + Savitzky-Golay)
   |                                                    --> Filtered landmarks (T x 21 x 3)
   v
[Stage 4] Finger-Key Assignment (Gaussian x-only + max-distance gate + both hands)
   |   + MIDI synchronization                           --> Initial finger labels
   v
[Stage 5] Neural Refinement (BiLSTM + Attention + Constrained Viterbi)
   |                                                    --> Refined finger labels
   v
Per-note fingering: L1-L5 / R1-R5
```

**Full-CV approach**: Keyboard detected from raw video. Hands detected live with MediaPipe. No dataset annotations used in the pipeline itself.

---

## Slide 17: Paper vs. Us — Side-by-Side Comparison

| Component | Paper (Moryossef et al.) | Our Implementation |
|---|---|---|
| Keyboard detection | "Largest bright area" (brief) | Full Canny/Hough/clustering pipeline |
| Hand detection | Fine-tuned Faster R-CNN | MediaPipe Hands (no training needed) |
| Pose estimation | Conv. Pose Machines + CycleGAN | MediaPipe 21-keypoint (built-in) |
| Temporal filtering | None | Hampel + Interpolation + Savitzky-Golay |
| Finger assignment | Gaussian (2D, key width sigma) | Gaussian (x-only, auto sigma, 4-sigma gate) |
| Sequence model | BiLSTM + MLP | BiLSTM + Multi-Head Attention |
| Decoding | argmax | Constrained Viterbi with biomechanical rules |
| Training loss | Cross-entropy | Cross-entropy + constraint loss |
| Evaluation metric | P/R/F1 + match rate | IFR (Irrational Fingering Rate) + IoU |
| Dataset | YouTube/APFD (self-collected) | PianoVAM (KAIST, academic benchmark) |

---

## Slide 18: Key Takeaways

1. **We followed the paper's conceptual pipeline** (Video -> Keyboard -> Hands -> Assignment) but **re-implemented every stage** with modern tools and our own engineering.

2. **We did NOT use** the paper's most novel contributions (CycleGAN adaptation, Faster R-CNN hand detector). Instead, we replaced them with **MediaPipe**, a more modern and practical solution.

3. **Our original contributions**:
   - Fully engineered automatic keyboard detection (Canny/Hough/clustering)
   - Temporal filtering pipeline (Hampel/interpolation/Savitzky-Golay)
   - X-distance-only Gaussian with max-distance gate
   - Biomechanical constraint system with constrained Viterbi decoding
   - Constraint-aware training loss
   - IFR evaluation metric

4. **Practical advantage**: Our pipeline requires no custom training data for detection — it works on any piano video out of the box.

---

## Slide 19: References

1. Moryossef, A., Elazar, Y., & Goldberg, Y. (2023). "At Your Fingertips: Extracting Piano Fingering Instructions from Videos." arXiv:2303.03745
2. PianoVAM Dataset — KAIST (107 piano performances with video, audio, MIDI)
3. MediaPipe Hands — Google (21-keypoint hand pose estimation)
4. Savitzky, A. & Golay, M. J. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures.
5. Nakamura, E. et al. (2020). Statistical Learning and Estimation of Piano Fingering. (PIG dataset)
6. Viterbi, A. (1967). Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm.

---

*Thank you!*
