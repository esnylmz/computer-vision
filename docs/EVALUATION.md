# Evaluation Protocol

## Metrics

### 1. Accuracy
Exact match rate between predictions and ground truth (first annotator).

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### 2. M_gen (General Match Rate)
Average match rate across all annotators. For each note, compute the fraction of annotators that agree with the prediction, then average across all notes.

```
M_gen = (1/N) × Σ (matches_i / num_annotators_i)
```

### 3. M_high (Highest Match Rate)
Match rate with the closest ground truth among multiple annotators.

```
M_high = (Notes matching any annotator) / (Total Notes)
```

### 4. IFR (Irrational Fingering Rate)
Fraction of note transitions that violate biomechanical constraints.

Irrational transitions include:
- Same finger on consecutive different notes (interval > 2 semitones)
- Stretch exceeding physical limits between finger pairs
- Invalid finger crossings

```
IFR = (Irrational Transitions) / (Total Transitions)
```

## Current Evaluation Status

| Metric | Status | Notes |
|--------|--------|-------|
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
