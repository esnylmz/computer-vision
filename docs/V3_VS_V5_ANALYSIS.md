# Computer Vision Project Analysis: V3 vs V5 Approach

## Executive Summary

**Bottom Line:** The V5 pipeline I built adds **major CV contributions** (CNN on pixels, temporal refinement) but has a weaker story about "pure CV" compared to V3's fully automatic keyboard detection. **The best approach is a hybrid**: keep V3's auto keyboard detection for Group B + add V5's CNN/BiLSTM innovations.

---

## Detailed Comparison

### V3 Approach (Your Original)

| Component | Group A (Annotations) | Group B (Pure CV) |
|-----------|----------------------|-------------------|
| **Keyboard** | Metadata corners | **Canny + Hough (fully automatic)** |
| **Hand landmarks** | Pre-extracted JSON | **MediaPipe on raw video** |
| **Filtering** | Hampel + SavGol | Hampel + SavGol |
| **Assignment** | Gaussian proximity | Gaussian proximity |
| **Refinement** | BiLSTM (trained on A) | BiLSTM (applied to B) |

**CV Strengths:**
- ✅ Group B is **truly annotation-free** at inference (no metadata corners)
- ✅ Automatic keyboard detection is a **clear CV contribution**
- ✅ Comparison: "annotations vs pure vision" is easy to explain

**CV Weaknesses:**
- ❌ No learning from visual appearance (only geometry)
- ❌ Gaussian assignment is hand-crafted, not learned
- ❌ BiLSTM trained on Group A features, not visual features
- ❌ **Main problem:** Canny+Hough auto detection was **failing** on many videos

**Professor might ask:**
- "Your auto keyboard detection fails often — is this practical?"
- "Why not learn from visual features instead of only coordinates?"

---

### V5 Approach (What I Built)

| Component | Group A (Training only) | Group B (Deployable) |
|-----------|------------------------|----------------------|
| **Keyboard** | Metadata corners | **Metadata corners** |
| **Hand landmarks** | (not used) | **MediaPipe on raw video** |
| **Teacher labels** | MIDI + proximity | (not used at inference) |
| **Press classifier** | (trained on A) | **CNN on 64×64 pixel crops** |
| **Refinement** | (trained on A) | **BiLSTM on [prob, dx, dy, speed]** |

**CV Strengths:**
- ✅ **CNN learns visual press cues from pixels** (nail angle, skin deformation, shadow) — **major CV contribution**
- ✅ Temporal refinement with sequence modeling
- ✅ Group A is **only for training** — clean separation
- ✅ MediaPipe + homography work reliably

**CV Weaknesses:**
- ❌ Uses metadata keyboard corners at inference ← **this is the issue**
- ❌ Professor might say: "You're using annotations (corners) so this isn't pure CV"
- ❌ Story is more complex to explain

**Professor might ask:**
- "You use metadata corners — how is this different from Group A?"
- "Couldn't you just use the provided hand skeleton JSON at inference too?"

---

## The Core Issue: Metadata Corners

**The problem:** Using metadata keyboard corners at inference time makes Group B look like it's "using annotations."

**Counter-argument (valid but weak):**
- Keyboard corners are **camera calibration**, not musical/hand annotations
- In a real deployment, you'd calibrate the camera once per setup
- Similar to: using camera intrinsics in 3D reconstruction

**But professors might not buy this argument.**

---

## Recommended Solution: Hybrid Approach

**Keep the best of both worlds:**

### Recommended Pipeline Structure

```
GROUP A (Training/Analysis):
  - Use all annotations (JSON skeletons, corners, MIDI)
  - Better temporal filtering (as you mentioned)
  - Generate teacher labels
  - Train CNN and BiLSTM
  
GROUP B (Deployable):
  - Keyboard: Canny + Hough (automatic) ← from V3
  - Hand: MediaPipe on raw video
  - Press: CNN on pixel crops ← from V5
  - Refinement: BiLSTM ← from V5
  
GROUP C (Optional, for ablation):
  - Same as Group B but uses metadata corners
  - Shows: "auto detection vs calibrated corners"
```

### What This Achieves

1. **Clear CV narrative:**
   - "Group B uses ZERO annotations — fully automatic"
   - "We add learning: CNN sees visual press cues, not just geometry"
   - "Temporal refinement via BiLSTM"

2. **Addresses V3's weaknesses:**
   - Auto keyboard detection (you need to fix the Canny/Hough to work better)
   - Adds learned visual features (CNN)
   - Adds temporal modeling (BiLSTM)

3. **Keeps Group A stronger:**
   - As you said: don't change the temporal filtering on Group A
   - Use the better JSON skeletons
   - This gives cleaner teacher labels for CNN training

4. **Three-way comparison for ablation:**
   - Group A: "upper bound with perfect annotations"
   - Group B: "pure CV, automatic everything"
   - Group C (optional): "Group B + calibrated corners"
   
   Shows: How much do we lose by going fully automatic?

---

## Specific Recommendations

### 1. Fix the Auto Keyboard Detection (Critical)

**Problem in V3:** Canny+Hough fails on many videos.

**Solutions to try:**
- Add multiple parameter sets (already in V3, but expand)
- Try **deep learning** keyboard detector:
  - Train a small U-Net or Mask R-CNN on a few annotated frames
  - Input: video frame → Output: keyboard mask
  - Use metadata corners from TRAIN videos to generate training data
- Hybrid: Try auto detection first; if it fails, fall back to homography from approximate manual corners (but mark this in results)

**This is crucial** — if auto detection fails, your "pure CV" claim falls apart.

### 2. Keep Group A Temporal Filtering

You're right — don't change Group A's filtering. It should use:
- The provided hand skeleton JSON (higher quality than MediaPipe)
- Hampel + SavGol filtering (works well)
- This generates better teacher labels

### 3. Add the CNN Visual Learning (from V5)

This is the **biggest CV contribution**.

**Key points to emphasize:**
- CNN sees **pixels only**, no coordinates
- Learns visual cues: finger pad contact, nail angle, skin deformation
- **Why this matters:** Geometry alone (x, y of fingertip) is ambiguous
  - A finger hovering 2mm above a key has similar (x,y) to a pressing finger
  - Visual appearance disambiguates this

**Professor will like this because:**
- It's a learned feature representation
- It's visual, not just geometric
- It's a novel contribution (not in the PianoVAM paper)

### 4. Temporal Refinement (from V5)

Keep the BiLSTM refinement, but make it clear:
- Input: CNN probabilities + motion features
- Output: temporally consistent predictions
- **Different from V3**: trained on visual features, not just geometric assignments

### 5. Add More Visualizations (You're Right!)

**Needed visualizations:**

**Throughout processing:**
1. **Step 1:** Sample frames showing automatic keyboard detection success/failure
2. **Step 2:** 
   - Raw MediaPipe vs filtered landmarks (timeline plot)
   - Comparison: JSON skeletons (Group A) vs MediaPipe (Group B)
3. **Step 3:** Timeline showing teacher labels (raw vs smoothed)
4. **Step 4:** 
   - CNN attention maps / grad-CAM (show what CNN looks at)
   - Crop examples: press vs no-press side-by-side
   - Training curves (loss, accuracy)
   - Failure cases (where CNN gets confused)
5. **Step 5:**
   - Timeline: CNN-only vs CNN+BiLSTM
   - Attention weights from BiLSTM
   - Event consistency improvement

**Final comparison:**
- Side-by-side: Group A vs Group B vs Group C
- Precision-Recall curves for all three
- Confusion matrices
- Biomechanical violation rates
- **Qualitative:** Video overlay showing predictions on a test clip

### 6. Evaluation After Step 5

**Metrics to report:**

| Metric | Group A | Group B | Group C |
|--------|---------|---------|---------|
| Precision | | | |
| Recall | | | |
| F1 | | | |
| ROC-AUC | | | |
| Event consistency | | | |
| Biomechanical violations | | | |
| **Keyboard detection success rate** | N/A | **X%** | 100% |

**Additional analyses:**
- **Per-finger breakdown:** Which fingers are hardest? (usually thumb)
- **Per-difficulty:** Easier pieces vs harder pieces
- **Error analysis:** What causes failures?
  - Hand occlusion?
  - Fast passages?
  - Thumb crossings?

---

## Answering Your Specific Questions

### "What is your contribution with trying them with Group B? They already had annotations."

**Good answer:**
1. **Learning visual press indicators:** The dataset has MIDI (when notes play) but NOT labels for "is this finger pressing." We create these labels (Group A) then train a CNN to recognize press from **visual appearance** (Group B). This is novel.

2. **Automatic keyboard localization:** We eliminate the need for manual camera calibration, making the system deployable without per-video annotation.

3. **Temporal modeling of hand-keyboard interaction:** BiLSTM learns temporal patterns of finger movements during playing, reducing isolated false positives.

**Bad answer:**
- "We just tested if MediaPipe works" ← too weak

### "How can we make our project better and more aligned with computer vision?"

**Strong CV elements to emphasize:**
1. **Image-based learning:** CNN on pixels (not just coordinates)
2. **Automatic scene understanding:** Keyboard detection without annotations
3. **Multi-modal fusion:** Visual features + motion features + temporal context
4. **Ablation studies:** Show what each component contributes
5. **Failure analysis:** Understand when and why vision fails

**Weak elements to avoid:**
- "We just ran MediaPipe" ← not enough
- "We used provided annotations" ← defeats the CV purpose

### "Do we need more visualizations?"

**Yes, absolutely.** Visualizations:
- Make your contributions **visible**
- Help the professor understand your pipeline
- Show that you understand the failure modes
- Make your thesis/presentation much stronger

**Priority visualizations:**
1. Keyboard detection: success/failure examples
2. CNN attention maps: what the model sees
3. Timeline comparisons: Group A vs B vs C
4. Failure cases: where does it break and why?

---

## Action Plan

### If you want to keep V5 as-is (quickest):

1. **Change one thing:** Make keyboard rectification optional
   - If metadata corners available → use them (Group C)
   - If not → require manual 4-point click or auto-detection (Group B)
2. **Add more visualizations** (see list above)
3. **Emphasize CNN learning** in your writing

### If you want the strongest project (recommended):

1. **Merge V3 + V5:**
   - Copy auto keyboard detection from V3 → V5
   - Improve it to work reliably (try U-Net option)
   - Keep everything else from V5 (CNN, BiLSTM)

2. **Add Group C** for ablation

3. **Keep Group A filtering unchanged** (you're right about this)

4. **Add comprehensive visualizations**

5. **Write a strong evaluation section** comparing all groups

### Implementation Priority

**Week 1 (Critical):**
- [ ] Fix/improve auto keyboard detection
- [ ] Add Group C (metadata corners) as separate path
- [ ] Basic visualization: keyboard detection examples

**Week 2 (Important):**
- [ ] CNN attention maps / grad-CAM
- [ ] Timeline comparison plots
- [ ] Comprehensive evaluation metrics

**Week 3 (Polish):**
- [ ] Failure case analysis
- [ ] Video demo with predictions overlay
- [ ] Clean up code + documentation

---

## Conclusion

**Your instinct is correct** — using metadata corners weakens the "pure CV" story.

**Best path forward:**
1. Keep V5's CNN and BiLSTM (major contributions)
2. Add back V3's auto keyboard detection (but fix it)
3. Keep Group A's better filtering (as you suggested)
4. Add visualization throughout
5. Write a comprehensive 3-way comparison

**This gives you:**
- ✅ Strong CV contributions (CNN, auto detection)
- ✅ Clear narrative (fully automatic Group B)
- ✅ Thorough evaluation (3 groups compared)
- ✅ Publication-quality results

**The professor will be satisfied because:**
- You're learning visual features (CNN)
- You're eliminating manual annotation (auto detection)
- You understand the trade-offs (Group A vs B vs C)
- You have comprehensive evaluation and visualization
