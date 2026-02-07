"""
Piano CV Pipeline — Vision-based hand–keyboard interaction detection.

New flat module structure (v5):
    src/data.py              – dataset loading, sampling, splitting, manifest
    src/mediapipe_extract.py – hand landmark extraction from raw video
    src/homography.py        – keyboard rectification via homography
    src/teacher_labels.py    – Group A teacher label generation
    src/crops.py             – fingertip-centered crop extraction
    src/cnn.py               – CNN press/no-press classifier
    src/bilstm.py            – temporal refinement model
    src/eval.py              – evaluation metrics and reporting
    src/viz.py               – visualization utilities

Legacy v3 code is preserved under src/_v3_data/, src/keyboard/, etc.
"""

__version__ = "5.0.0"
