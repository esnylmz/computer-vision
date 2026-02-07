# Evaluation Protocol

## Metrics

### 1. Keyboard Detection IoU

Intersection-over-Union between the automatically detected keyboard bounding box and the corner-annotation ground truth.

```
IoU = Area(Intersection) / Area(Union)
```

Evaluated per-sample and aggregated (mean, median) across the dataset. Tests the Canny/Hough detection pipeline independently.

### 2. IFR (Irrational Fingering Rate)

Fraction of note transitions that violate biomechanical constraints.

Irrational transitions include:
- Same finger on consecutive different notes (interval > 2 semitones)
- Stretch exceeding physical limits between finger pairs
- Invalid finger crossings

```
IFR = (Irrational Transitions) / (Total Transitions)
```

### 3. Accuracy
Exact match rate between predictions and ground truth (first annotator).

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### 4. M_gen (General Match Rate)
Average match rate across all annotators. For each note, compute the fraction of annotators that agree with the prediction, then average across all notes.

```
M_gen = (1/N) × Σ (matches_i / num_annotators_i)
```

### 5. M_high (Highest Match Rate)
Match rate with the closest ground truth among multiple annotators.

```
M_high = (Notes matching any annotator) / (Total Notes)
```

## Current Evaluation Status

| Metric | Status | Notes |
|--------|--------|-------|
| Keyboard IoU | ✅ Evaluated | Auto-detection vs corner annotations, per-sample and aggregated |
| IFR | ✅ Evaluated | Works without ground-truth finger labels |
| Accuracy | ⏳ Requires ground truth | PianoVAM TSV has onset/note/velocity but no finger labels |
| M_gen | ⏳ Requires ground truth | Needs multiple annotators |
| M_high | ⏳ Requires ground truth | Needs at least one annotator |

## Data Split

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 73 | BiLSTM training |
| Validation | 19 | Hyperparameter tuning |
| Test | 14 | Final evaluation |

## Usage

### Keyboard Detection IoU

```python
from src.keyboard.auto_detector import AutoKeyboardDetector

detector = AutoKeyboardDetector()
result = detector.detect_from_video("video.mp4")
iou = detector.evaluate_against_corners(result, sample.metadata["keyboard_corners"])
print(f"Keyboard IoU: {iou:.3f}")
```

### Fingering Metrics

```python
from src.evaluation.metrics import FingeringMetrics

metrics = FingeringMetrics()

# Evaluate with ground truth (when available)
result = metrics.evaluate(predictions, ground_truth, pitches=pitches)
print(f"Accuracy: {result.accuracy:.3f}")
print(f"M_gen:    {result.m_gen:.3f}")
print(f"M_high:   {result.m_high:.3f}")
print(f"IFR:      {result.ifr:.3f}")

# Evaluate IFR only (no ground truth needed)
from src.refinement.constraints import BiomechanicalConstraints
constraints = BiomechanicalConstraints()
violations = constraints.validate_sequence(fingers, pitches, hands)
ifr = len(violations) / max(1, len(fingers) - 1)
```
