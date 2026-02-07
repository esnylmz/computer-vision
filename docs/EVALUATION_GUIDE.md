# Three-Group Evaluation Report

## Executive Summary

This document explains the **three-group evaluation strategy** and how to interpret the results after Step 5.

---

## Three Groups Explained

### Group A: Teacher (Training Only)

**Configuration:**
- **Keyboard:** Metadata corners (from PianoVAM annotations)
- **Hand Landmarks:** Pre-extracted JSON skeletons (from PianoVAM)
- **Temporal Filtering:** Hampel + SavGol smoothing

**Why it's the best:**
- JSON skeletons are more stable than raw MediaPipe
- Metadata corners give perfect keyboard alignment
- Temporal filtering removes detection noise

**NOT deployable** — requires annotations.

**Purpose:** Generate high-quality teacher labels for CNN training.

---

### Group B: Deployable (Pure CV)

**Configuration:**
- **Keyboard:** **Auto-detection** via Canny + Hough
- **Hand Landmarks:** MediaPipe Hands on raw video

**Why it's deployable:**
- No annotations required at inference
- Auto keyboard detection eliminates calibration
- Pure computer vision end-to-end

**This is the final system.**

**Expected Performance:** ~10-15% worse than Group A, but fully automatic.

---

### Group C: Ablation Study

**Configuration:**
- **Keyboard:** Metadata corners (like Group A)
- **Hand Landmarks:** MediaPipe on raw video (like Group B)

**Why it matters:**
Shows exactly what we lose/gain by using:
- **A → C:** Effect of landmark quality (JSON vs MediaPipe)
- **C → B:** Effect of auto keyboard detection (metadata vs auto)

---

## Evaluation Strategy

### 1. Train on Group A (Best Quality)

All models (CNN, BiLSTM) are trained using Group A's refined labels:

```
Train:
  - Extract crops from Group A videos (TRAIN split)
  - Labels = Group A's teacher labels (press_smooth)
  - Train CNN on these crops
  - Train BiLSTM on Group A sequences
```

**Why:** We want to train on the best possible labels to give the models the best chance to learn.

---

### 2. Apply to All Three Groups

After training, apply the CNN and BiLSTM to **all three groups**:

```
For each group (A, B, C):
  For each video:
    1. Extract crops from that group's landmarks
    2. Run CNN → get press_prob
    3. Build sequences [press_prob, dx, dy, speed]
    4. Run BiLSTM → get press_prob_refined
```

**Why:** This allows fair comparison — same models, different input quality.

---

### 3. Evaluate on TEST Split Only

For each group, compute metrics **only on TEST videos**:

```python
for group in [A, B, C]:
    test_videos = manifest[manifest['split'] == 'test']
    
    y_true = []  # Teacher labels (Group A always)
    y_pred = []  # Predictions (from this group's landmarks)
    
    for video in test_videos:
        if video in group:
            y_true.extend(group[video]['press_smooth'])
            y_pred.extend(group[video]['press_prob_refined'])
    
    metrics = evaluate(y_true, y_pred)
```

**Key:** `y_true` is always Group A's teacher labels (ground truth), but `y_pred` comes from each group's landmarks (different quality).

---

## Interpreting Results

### Expected Performance

| Group | Keyboard | Landmarks | F1 (CNN) | F1 (BiLSTM) | Deployable? |
|-------|----------|-----------|----------|-------------|-------------|
| **A** | Metadata | JSON (refined) | ~0.85 | ~0.88 | ❌ |
| **C** | Metadata | MediaPipe | ~0.80 | ~0.83 | ❌ |
| **B** | **Auto** | MediaPipe | ~0.72 | ~0.77 | ✅ |

### Key Comparisons

#### Comparison 1: Group A vs Group C
**Question:** What do we lose by using MediaPipe instead of refined JSON skeletons?

**Expected:** ~5% drop in F1

**Interpretation:**
- JSON skeletons are cleaner (pre-filtered, better quality)
- MediaPipe has occasional misdetections, jitter
- **But both use metadata corners** → same keyboard alignment

**Conclusion:** Landmark quality matters, but MediaPipe is "good enough."

---

#### Comparison 2: Group C vs Group B
**Question:** What do we lose by using auto keyboard detection?

**Expected:** ~8-10% drop in F1

**Interpretation:**
- Auto detection sometimes fails (~15% of videos)
- When it works, alignment is slightly off (5-10px)
- Keyboard coordinate errors propagate to press detection

**Conclusion:** This is the **cost of deployability**. We pay ~10% F1 to eliminate manual calibration.

---

#### Comparison 3: Group A vs Group B
**Question:** What's the overall gap from "ideal" to "deployable"?

**Expected:** ~10-15% drop in F1

**Interpretation:**
- This is the sum of landmark quality (~5%) + auto keyboard (~8-10%)
- **Still usable** if F1 > 0.70
- **Fully automatic** — no annotations needed

**Conclusion:** The system is practical. We sacrifice some accuracy for zero manual intervention.

---

#### Comparison 4: CNN Only vs CNN+BiLSTM (within each group)
**Question:** Does temporal refinement help?

**Expected:** ~3-5% improvement in F1

**Interpretation:**
- BiLSTM reduces per-frame jitter
- Smooths press onset/offset transitions
- Reduces isolated single-frame false positives

**Conclusion:** Temporal modeling is essential for clean event detection.

---

## Visualizations

### 1. Metrics Comparison Bar Chart

Shows precision/recall/F1/AUC for all three groups side-by-side.

**Look for:**
- Group A should be highest (best inputs)
- Group B should be lowest (auto detection penalty)
- Group C should be in between

**If not:** Something is wrong with the pipeline.

---

### 2. Timeline Comparison

Shows one test video with three panels:
- **Top:** Teacher labels (Group A)
- **Middle:** CNN predictions (Group B)
- **Bottom:** CNN+BiLSTM predictions (Group B)

**Look for:**
- Middle plot should be noisy (per-frame CNN)
- Bottom plot should be smoother (BiLSTM refinement)
- Both should roughly follow the teacher (top)

**If not:** CNN or BiLSTM failed to learn.

---

### 3. Per-Finger Performance

Bar chart showing F1 for each finger (thumb, index, middle, ring, pinky).

**Look for:**
- Thumb often lowest (biggest finger, different posture)
- Index/middle usually highest (most visible)
- Pinky often tricky (small, edge of keyboard)

**Insight:** Shows which fingers are hardest to detect — useful for future improvements.

---

### 4. Attention Heatmaps (Grad-CAM)

Overlays showing which pixels the CNN focuses on.

**Look for:**
- Heatmap should highlight fingertip, nail, key boundary
- "Press" crops should focus on contact region
- "No-press" crops should focus on hovering finger

**If not:** CNN is not learning visual features (likely overfitting to position).

---

### 5. Event Consistency Metric

```
Isolated presses = single-frame presses with no neighbors
Event consistency = 1 - (isolated_count / total_presses)
```

**Look for:**
- CNN only: ~0.60-0.70 (40-30% isolated)
- CNN+BiLSTM: ~0.85-0.90 (15-10% isolated)

**Interpretation:** Higher is better. BiLSTM should significantly reduce isolated presses.

---

## Addressing Professor's Questions

### Q1: "What's your CV contribution if annotations exist?"

**A:** The annotations are used **only for training**. At test time:
- Group B uses **only** raw video (no annotations)
- Auto keyboard detection finds keyboard automatically
- CNN learns visual press patterns (not geometric)
- BiLSTM refines temporally

**Analogy:** Using ImageNet labels to train ResNet doesn't make ResNet "not a CV contribution."

---

### Q2: "Why three groups? Isn't one enough?"

**A:** Each group isolates a specific contribution:
- **A → C:** Landmark quality (JSON vs MediaPipe)
- **C → B:** Auto keyboard detection
- **A → B:** Overall deployability cost

**Without ablation:** You can't prove which parts matter.

---

### Q3: "What if Group B's F1 is too low?"

**A:** If F1 < 0.65, you have options:
1. **Improve auto detection:** Add more parameter sets, try template matching
2. **Fallback to manual:** Ask user to click 4 corners (one-time calibration)
3. **Optical flow:** Add motion features to help CNN
4. **Multi-scale CNN:** Use crops at multiple resolutions

**But:** Even at F1=0.70, the system is **useful** for applications like practice analytics (where 100% accuracy isn't critical).

---

### Q4: "How do you compare to baseline?"

**Baselines to report:**
1. **Rule-based (proximity only):** If fingertip within 10px of active MIDI note → press
   - Expected F1: ~0.50-0.60 (no visual learning)
2. **Logistic regression on (x, y, confidence):** Simple classifier on geometric features
   - Expected F1: ~0.65-0.70 (no visual features)
3. **CNN (ours):** Learns from pixels
   - Expected F1: ~0.72-0.77 (Group B) or ~0.85-0.88 (Group A)

**Conclusion:** CNN significantly outperforms geometric baselines.

---

## Summary Table

| Aspect | Group A | Group B | Group C |
|--------|---------|---------|---------|
| **Keyboard** | Metadata | **Auto CV** | Metadata |
| **Hands** | JSON (refined) | MediaPipe | MediaPipe |
| **F1 (CNN)** | ~0.85 | ~0.72 | ~0.80 |
| **F1 (BiLSTM)** | ~0.88 | ~0.77 | ~0.83 |
| **Deployable?** | ❌ | ✅ | ❌ |
| **Annotation Cost** | High | **Zero** | Medium |
| **Purpose** | Training | **Production** | Ablation |

---

## Conclusion

**The story:** We use the best available annotations (Group A) to train a strong CNN, which Group B deploys using **only** raw video — no annotations, no calibration, pure computer vision.

**The trade-off:** We lose ~10-15% accuracy, but gain full automation.

**The contribution:** Learned visual press detection (CNN), auto keyboard localization (Canny+Hough), temporal consistency (BiLSTM), and rigorous ablation (3 groups).

**The proof:** The ablation study quantifies exactly what we lose and why it's worth it.
