# Evaluation Protocol

## Metrics

### 1. Accuracy
Standard exact match rate between predictions and ground truth.

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### 2. M_gen (General Match Rate)
Average match rate across all annotators. For each note, compute the fraction of annotators that match the prediction, then average across all notes.

```
M_gen = (1/N) × Σ (matches_i / num_annotators_i)
```

### 3. M_high (Highest Match Rate)
Match rate with the closest ground truth among multiple annotators.

```
M_high = (Notes matching any annotator) / (Total Notes)
```

### 4. IFR (Irrational Fingering Rate)
Fraction of transitions that violate biomechanical constraints.

Irrational transitions include:
- Same finger on consecutive different notes
- Stretch exceeding physical limits
- Invalid finger crossings

```
IFR = (Irrational Transitions) / (Total Transitions)
```

## Evaluation Protocol

### Data Split
- **Train**: 73 recordings (for neural refinement training)
- **Validation**: 19 recordings (hyperparameter tuning)
- **Test**: 14 recordings (final evaluation)

### Per-Sample Evaluation
1. Run pipeline on test sample
2. Compare predictions with ground truth annotations
3. Compute all metrics
4. Record per-finger accuracy

### Aggregation
- Report mean and standard deviation across test samples
- Stratify by skill level (Beginner/Intermediate/Advanced)
- Analyze per-finger performance

## Expected Results

Based on literature:

| Method | Accuracy | M_gen | M_high | IFR |
|--------|----------|-------|--------|-----|
| Gaussian Assignment | 70-75% | 65-70% | 75-80% | 5-10% |
| + Neural Refinement | 80-85% | 75-80% | 85-90% | 2-5% |
| Human Agreement | ~85% | 80% | 90% | <2% |

## Usage

```python
from src.evaluation.metrics import FingeringMetrics

metrics = FingeringMetrics()

# Single evaluation
result = metrics.evaluate(predictions, ground_truth)
print(f"Accuracy: {result.accuracy:.3f}")
print(f"M_gen: {result.m_gen:.3f}")
print(f"M_high: {result.m_high:.3f}")
print(f"IFR: {result.ifr:.3f}")

# Per-finger accuracy
for finger, acc in result.per_finger_accuracy.items():
    print(f"Finger {finger}: {acc:.3f}")
```

## Visualization

```python
from src.evaluation.visualization import ResultVisualizer

viz = ResultVisualizer(output_dir='./outputs')

# Confusion matrix
viz.plot_confusion_matrix(result.confusion_matrix)

# Per-finger accuracy bar chart
viz.plot_per_finger_accuracy(result.per_finger_accuracy)

# Summary figure
viz.create_summary_figure(metrics_dict, per_finger, confusion)
```

