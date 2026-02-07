#!/usr/bin/env python3
"""
run_pipeline.py — Single entry point for the Piano CV Pipeline.

Usage:
    python run_pipeline.py --mode smoke            # 3 videos, 20s clips, 1 epoch
    python run_pipeline.py --mode full             # 60 videos, 120s clips, 10 epochs
    python run_pipeline.py --step 1                # run only Step 1
    python run_pipeline.py --N 5 --clip_duration 30 --frame_step 5 --epochs 3

Modes:
    smoke   N=3   clip_duration=20   frame_step=10  epochs=1   (default)
    full    N=60  clip_duration=120  frame_step=5   epochs=10
"""

import argparse
import json
from pathlib import Path


# ─────────────── Default configurations ───────────────────────────

SMOKE_DEFAULTS = dict(N=3, clip_duration=20, frame_step=10, epochs=1)
FULL_DEFAULTS = dict(N=60, clip_duration=120, frame_step=5, epochs=10)


def get_args():
    p = argparse.ArgumentParser(
        description="Piano CV Pipeline — vision-based key-press detection",
    )
    p.add_argument(
        "--mode", choices=["smoke", "full"], default="smoke",
        help="Preset parameter set (default: smoke)",
    )
    p.add_argument(
        "--step", type=int, default=None,
        help="Run only this step (1-5). Omit to run all implemented steps.",
    )
    p.add_argument("--N", type=int, default=None, help="Number of videos")
    p.add_argument(
        "--clip_duration", type=float, default=None,
        help="Clip duration in seconds",
    )
    p.add_argument("--frame_step", type=int, default=None, help="Frame step")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs")
    p.add_argument(
        "--cache_dir", type=str, default="./data/cache",
        help="Cache directory for downloaded files",
    )
    p.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Root output directory",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_params(args) -> dict:
    """Merge mode defaults with any explicit CLI overrides."""
    defaults = SMOKE_DEFAULTS if args.mode == "smoke" else FULL_DEFAULTS
    params = dict(defaults)
    for key in ["N", "clip_duration", "frame_step", "epochs"]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
    params["cache_dir"] = args.cache_dir
    params["output_dir"] = args.output_dir
    params["seed"] = args.seed
    params["mode"] = args.mode
    return params


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Data & Split
# ═══════════════════════════════════════════════════════════════════

def run_step1(params: dict):
    """
    Sample N videos, create video-level train/test split, download
    with caching, save manifest.
    """
    from src.data import (
        sample_and_split,
        download_manifest_videos,
        save_manifest,
    )

    print("=" * 60)
    print("STEP 1 — DATA & SPLIT (no models)")
    print("=" * 60)
    print(f"  N              = {params['N']}")
    print(f"  clip_duration  = {params['clip_duration']}s")
    print(f"  frame_step     = {params['frame_step']}")
    print(f"  seed           = {params['seed']}")
    print()

    # 1) sample & split
    manifest, dataset = sample_and_split(
        N=params["N"],
        clip_duration=params["clip_duration"],
        frame_step=params["frame_step"],
        cache_dir=params["cache_dir"],
        seed=params["seed"],
    )

    # 2) download / cache videos (also verifies fps)
    local_paths = download_manifest_videos(manifest, dataset, verify_fps=True)
    manifest["local_video_path"] = manifest["video_id"].map(local_paths)

    # 3) save manifest
    out_dir = Path(params["output_dir"]) / (
        "smoke_test" if params["mode"] == "smoke" else "full"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    save_manifest(manifest, str(manifest_path))

    # 4) preview
    preview_cols = [
        "video_id", "split", "clip_start", "clip_duration",
        "fps", "frame_step", "composer", "piece",
    ]
    print("\n--- Manifest Preview ---")
    print(manifest[preview_cols].to_string(index=False))

    n_train = int((manifest["split"] == "train").sum())
    n_test = int((manifest["split"] == "test").sum())
    print(f"\nTrain videos : {n_train}")
    print(f"Test videos  : {n_test}")
    print(f"Manifest     : {manifest_path}")
    print("STEP 1 COMPLETE.\n")

    return manifest, dataset


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Group B: Video-only CV extraction
# ═══════════════════════════════════════════════════════════════════

def run_step2(params: dict, manifest=None, dataset=None):
    """
    For every video in the manifest:
      1. Extract hand landmarks via MediaPipe (video-only, no JSON).
      2. Compute homography from metadata corners → rectified keyboard.
      3. Warp fingertip coords into normalised keyboard space.
      4. Save ≥10 visualisation frames (overlay + rectified ROI).
      5. Save per-video landmark CSVs.
    """
    import json as _json
    from src.data import load_manifest
    from src.mediapipe_extract import extract_landmarks_from_video
    from src.homography import (
        parse_corners,
        corners_valid,
        compute_keyboard_homography,
        add_keyboard_coords_to_landmarks,
    )
    from src.viz import save_sanity_check_frames

    print("=" * 60)
    print("STEP 2 — GROUP B: VIDEO-ONLY CV EXTRACTION")
    print("=" * 60)

    # Load manifest from disk if not passed from Step 1
    if manifest is None:
        out_dir = Path(params["output_dir"]) / (
            "smoke_test" if params["mode"] == "smoke" else "full"
        )
        manifest = load_manifest(str(out_dir / "manifest.json"))
        print(f"  Loaded manifest with {len(manifest)} videos")

    sub_dir = "smoke_test" if params["mode"] == "smoke" else "full"
    viz_dir = Path(params["output_dir"]) / sub_dir / "viz"
    lm_dir = Path(params["output_dir"]) / sub_dir / "landmarks"
    lm_dir.mkdir(parents=True, exist_ok=True)

    all_landmarks = {}

    for row_idx, row in manifest.iterrows():
        vid_id = row["video_id"]
        video_path = row["local_video_path"]
        fps = row["fps"]
        corners = parse_corners(row["keyboard_corners"])

        print(f"\n[{vid_id}]")

        # ── 1) Hand landmark extraction (video-only) ─────────────
        landmarks_df, vinfo = extract_landmarks_from_video(
            video_path=video_path,
            clip_duration=params["clip_duration"],
            frame_step=params["frame_step"],
            fps=fps,
        )
        n_det = landmarks_df["frame_idx"].nunique() if not landmarks_df.empty else 0
        print(
            f"  Landmarks : {len(landmarks_df)} detections in "
            f"{n_det}/{vinfo['n_processed']} frames "
            f"({vinfo['n_with_hands']} had hands)"
        )

        # ── 2) Keyboard rectification ────────────────────────────
        if not corners_valid(corners):
            print("  WARNING: invalid keyboard corners — skipping homography")
            H, dst_size = None, None
        else:
            landmarks_df, H = add_keyboard_coords_to_landmarks(
                landmarks_df, corners,
            )
            _, dst_size = compute_keyboard_homography(corners)
            print("  Homography : computed, keyboard coords added")

        # ── 3) Visualisation sanity checks ────────────────────────
        if H is not None:
            save_sanity_check_frames(
                video_path=video_path,
                landmarks_df=landmarks_df,
                H=H,
                dst_size=dst_size,
                corners=corners,
                output_dir=str(viz_dir),
                video_id=vid_id,
                n_frames=10,
            )

        # ── 4) Save per-video landmarks CSV ──────────────────────
        lm_path = lm_dir / f"{vid_id}_landmarks.csv"
        landmarks_df.to_csv(lm_path, index=False)
        print(f"  Saved      : {lm_path}")

        all_landmarks[vid_id] = landmarks_df

    # ── Summary ───────────────────────────────────────────────────
    total_det = sum(len(df) for df in all_landmarks.values())
    print(f"\nSTEP 2 COMPLETE — {total_det} total fingertip detections "
          f"across {len(all_landmarks)} videos.")
    print(f"  Visualisations : {viz_dir}")
    print(f"  Landmarks      : {lm_dir}\n")

    return all_landmarks


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Group A: Teacher labels
# ═══════════════════════════════════════════════════════════════════

def run_step3(params: dict, manifest=None, dataset=None,
              all_landmarks=None):
    """
    Generate teacher (Group A) press labels for every video.
      1. Load MIDI/TSV annotations.
      2. Generate frame-level press labels via proximity matching.
      3. Apply Gaussian temporal smoothing.
      4. Align to Group B frames (same frame indices).
      5. Export labelled CSVs + timeline plot.
    """
    import pandas as pd
    from src.data import load_manifest, PianoVAMDataset
    from src.teacher_labels import (
        generate_teacher_labels_for_video,
        plot_teacher_timeline,
    )

    print("=" * 60)
    print("STEP 3 — GROUP A: TEACHER LABELS (analysis only)")
    print("=" * 60)

    sub_dir = "smoke_test" if params["mode"] == "smoke" else "full"
    out_base = Path(params["output_dir"]) / sub_dir

    if manifest is None:
        manifest = load_manifest(str(out_base / "manifest.json"))
    if dataset is None:
        dataset = PianoVAMDataset(
            split="train", cache_dir=params["cache_dir"],
            max_samples=params["N"],
        )

    # Load Step 2 landmarks from disk if not passed
    lm_dir = out_base / "landmarks"
    if all_landmarks is None:
        all_landmarks = {}
        for _, row in manifest.iterrows():
            p = lm_dir / f"{row['video_id']}_landmarks.csv"
            if p.exists():
                all_landmarks[row["video_id"]] = pd.read_csv(p)

    labeled: dict = {}
    for _, row in manifest.iterrows():
        vid_id = row["video_id"]
        print(f"\n[{vid_id}]")

        lm_df = all_landmarks.get(vid_id)
        if lm_df is None or lm_df.empty:
            print("  No landmarks — skipping")
            continue

        # Download / cache TSV and load
        sample = dataset.get_sample_by_id(vid_id)
        if sample is None:
            # Fall-back: build sample URL manually
            from src.data import PianoVAMDataset as _D
            tsv_url = _D.BASE_URL + f"TSV/{vid_id}.tsv"
            local = dataset.download_file(tsv_url)
            tsv_df = pd.read_csv(
                local, sep="\t",
                names=["onset", "key_offset", "frame_offset",
                       "note", "velocity"],
                header=None, comment="#",
            )
        else:
            tsv_df = dataset.load_tsv_annotations(sample)

        labeled_df = generate_teacher_labels_for_video(
            lm_df, tsv_df,
            fps=row["fps"],
            clip_duration=params["clip_duration"],
        )

        # Save
        out_path = lm_dir / f"{vid_id}_labeled.csv"
        labeled_df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        labeled[vid_id] = labeled_df

    # Timeline plot (first video, first available fingertip)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        first_df = next(iter(labeled.values()), None)
        first_id = next(iter(labeled.keys()), "")
        if first_df is not None:
            fig, ax = plt.subplots(figsize=(14, 3))
            plot_teacher_timeline(first_df, first_id, ax=ax)
            plot_path = out_base / "teacher_timeline.png"
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\nTimeline plot → {plot_path}")
    except Exception as e:
        print(f"  Timeline plot skipped: {e}")

    total = sum(len(d) for d in labeled.values())
    n_press = sum(int(d["press_raw"].sum()) for d in labeled.values())
    print(f"\nSTEP 3 COMPLETE — {total} labelled samples, "
          f"{n_press} press events.\n")
    return labeled


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — CNN training
# ═══════════════════════════════════════════════════════════════════

def run_step4(params: dict, manifest=None, dataset=None,
              labeled=None):
    """
    Train a CNN on fingertip crops (pixels only, no coordinates).
      1. Extract crops from video frames.
      2. Build train / test datasets using teacher labels.
      3. Train PressNet (1 epoch in smoke test).
      4. Evaluate on TEST videos: prec, rec, F1, AUC.
      5. Save confusion matrix + ROC curve.
    """
    import torch
    import pandas as pd
    import numpy as np
    from src.data import load_manifest, PianoVAMDataset
    from src.crops import extract_crops_for_video, PressCropDataset
    from src.cnn import train_cnn, predict_cnn
    from src.eval import evaluate_predictions, save_eval_plots

    print("=" * 60)
    print("STEP 4 — CNN TRAINING (pixels → press/no-press)")
    print("=" * 60)

    sub_dir = "smoke_test" if params["mode"] == "smoke" else "full"
    out_base = Path(params["output_dir"]) / sub_dir
    lm_dir = out_base / "landmarks"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    if manifest is None:
        manifest = load_manifest(str(out_base / "manifest.json"))

    # Load labelled landmarks from Step 3
    if labeled is None:
        labeled = {}
        for _, row in manifest.iterrows():
            p = lm_dir / f"{row['video_id']}_labeled.csv"
            if p.exists():
                labeled[row["video_id"]] = pd.read_csv(p)

    # ── Extract crops + labels per split ──────────────────────────
    train_crops, train_labels = [], []
    test_crops, test_labels = [], []
    test_vid_ids = []

    for _, row in manifest.iterrows():
        vid_id = row["video_id"]
        split = row["split"]
        ldf = labeled.get(vid_id)
        if ldf is None or ldf.empty or "press_smooth" not in ldf.columns:
            print(f"  [{vid_id}] no teacher labels — skipping")
            continue

        print(f"  [{vid_id}] extracting crops ({split}) ...")
        crops, idxs = extract_crops_for_video(
            row["local_video_path"], ldf, crop_size=64,
        )
        labs = ldf.loc[idxs, "press_smooth"].values.tolist()

        if split == "train":
            train_crops.extend(crops)
            train_labels.extend(labs)
        else:
            test_crops.extend(crops)
            test_labels.extend(labs)
            test_vid_ids.extend([vid_id] * len(crops))

    print(f"\n  Train crops: {len(train_crops)}  "
          f"(pos={sum(1 for l in train_labels if l > 0.5)})")
    print(f"  Test crops : {len(test_crops)}  "
          f"(pos={sum(1 for l in test_labels if l > 0.5)})")

    if not train_crops:
        print("  No training data — skipping CNN training")
        return None, None

    # ── Compute class weight ──────────────────────────────────────
    n_pos = max(sum(1 for l in train_labels if l > 0.5), 1)
    n_neg = max(len(train_labels) - n_pos, 1)
    pos_weight = n_neg / n_pos
    print(f"  pos_weight: {pos_weight:.1f}")

    # ── Train ─────────────────────────────────────────────────────
    train_ds = PressCropDataset(train_crops, train_labels)
    model, losses = train_cnn(
        train_ds,
        epochs=params["epochs"],
        batch_size=32,
        lr=1e-3,
        device=device,
        pos_weight=pos_weight,
    )

    # ── Evaluate on TEST ──────────────────────────────────────────
    cnn_probs_test = np.array([])
    cnn_metrics = {}
    if test_crops:
        test_ds = PressCropDataset(test_crops, test_labels)
        cnn_probs_test = predict_cnn(model, test_ds, device=device)

        y_true = np.array(test_labels)
        cnn_metrics = evaluate_predictions(
            y_true, cnn_probs_test, label="CNN",
        )
        save_eval_plots(
            y_true, cnn_probs_test,
            str(out_base / "eval"), label="CNN",
        )
    else:
        print("  No test data — skipping evaluation")

    # ── Save model ────────────────────────────────────────────────
    model_path = out_base / "pressnet.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved → {model_path}")

    # ── Write CNN probs back into labelled DataFrames ─────────────
    # (needed for Step 5)
    for _, row in manifest.iterrows():
        vid_id = row["video_id"]
        ldf = labeled.get(vid_id)
        if ldf is None or ldf.empty or "press_smooth" not in ldf.columns:
            continue
        crops_v, idxs_v = extract_crops_for_video(
            row["local_video_path"], ldf, crop_size=64,
        )
        if not crops_v:
            continue
        ds_v = PressCropDataset(crops_v, [0.0] * len(crops_v))
        probs_v = predict_cnn(model, ds_v, device=device)
        ldf.loc[idxs_v, "press_prob"] = probs_v
        labeled[vid_id] = ldf
        ldf.to_csv(lm_dir / f"{vid_id}_labeled.csv", index=False)

    print("STEP 4 COMPLETE.\n")
    return model, labeled


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Temporal refinement
# ═══════════════════════════════════════════════════════════════════

def run_step5(params: dict, manifest=None, dataset=None,
              labeled=None, cnn_model=None):
    """
    Train a BiLSTM to temporally refine CNN predictions.
      1. Build [press_prob, dx, dy, speed] sequences per fingertip.
      2. Train BiLSTM on TRAIN data.
      3. Predict on TEST data.
      4. Compare CNN-only vs CNN+BiLSTM.
      5. Compute event-consistency metric.
      6. Plot timeline comparison.
    """
    import torch
    import numpy as np
    import pandas as pd
    from src.data import load_manifest
    from src.bilstm import build_sequences, train_refiner, predict_refiner
    from src.eval import evaluate_predictions, save_eval_plots, event_consistency

    print("=" * 60)
    print("STEP 5 — TEMPORAL REFINEMENT (BiLSTM)")
    print("=" * 60)

    sub_dir = "smoke_test" if params["mode"] == "smoke" else "full"
    out_base = Path(params["output_dir"]) / sub_dir
    lm_dir = out_base / "landmarks"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if manifest is None:
        manifest = load_manifest(str(out_base / "manifest.json"))

    if labeled is None:
        labeled = {}
        for _, row in manifest.iterrows():
            p = lm_dir / f"{row['video_id']}_labeled.csv"
            if p.exists():
                labeled[row["video_id"]] = pd.read_csv(p)

    # ── Build train sequences ─────────────────────────────────────
    train_feats, train_labs = [], []
    for _, row in manifest.iterrows():
        if row["split"] != "train":
            continue
        ldf = labeled.get(row["video_id"])
        if ldf is None or "press_prob" not in ldf.columns:
            continue
        f, l = build_sequences(ldf, press_prob_col="press_prob")
        train_feats.extend(f)
        train_labs.extend(l)

    print(f"  Train sequences: {len(train_feats)}")

    if not train_feats:
        print("  No training sequences — skipping BiLSTM")
        return

    # ── Train ─────────────────────────────────────────────────────
    n_pos = sum(float((l > 0.5).sum()) for l in train_labs)
    n_neg = sum(float(len(l) - (l > 0.5).sum()) for l in train_labs)
    pw = max(n_neg, 1) / max(n_pos, 1)

    bilstm, losses = train_refiner(
        (train_feats, train_labs),
        epochs=params["epochs"],
        device=device,
        pos_weight=pw,
    )

    # ── Predict on TEST ───────────────────────────────────────────
    test_true_all, cnn_prob_all, refined_all = [], [], []

    for _, row in manifest.iterrows():
        if row["split"] != "test":
            continue
        ldf = labeled.get(row["video_id"])
        if ldf is None or "press_prob" not in ldf.columns:
            continue

        refined = predict_refiner(
            bilstm, ldf, press_prob_col="press_prob", device=device,
        )
        ldf = ldf.copy()
        ldf["press_prob_refined"] = refined.values

        # Save
        ldf.to_csv(lm_dir / f"{row['video_id']}_labeled.csv", index=False)

        # Collect for evaluation
        valid = ldf.dropna(subset=["press_prob", "press_prob_refined", "press_smooth"])
        if not valid.empty:
            test_true_all.append(valid["press_smooth"].values)
            cnn_prob_all.append(valid["press_prob"].values)
            refined_all.append(valid["press_prob_refined"].values)

    # ── Compare ───────────────────────────────────────────────────
    if test_true_all:
        y_true = np.concatenate(test_true_all)
        y_cnn = np.concatenate(cnn_prob_all)
        y_ref = np.concatenate(refined_all)

        print("\n  --- CNN only ---")
        m_cnn = evaluate_predictions(y_true, y_cnn, label="CNN")
        print("  --- CNN + BiLSTM ---")
        m_ref = evaluate_predictions(y_true, y_ref, label="CNN+BiLSTM")

        save_eval_plots(
            y_true, y_ref, str(out_base / "eval"), label="CNN+BiLSTM",
        )

        # Event consistency
        ec_cnn = event_consistency((y_cnn > 0.5).astype(int))
        ec_ref = event_consistency((y_ref > 0.5).astype(int))
        print(f"\n  Event consistency (lower isolation = better):")
        print(f"    CNN only   : {ec_cnn}")
        print(f"    CNN+BiLSTM : {ec_ref}")
    else:
        print("  No test predictions — skipping comparison")

    # ── Timeline plot ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Pick first test video with valid data
        for _, row in manifest.iterrows():
            if row["split"] != "test":
                continue
            ldf = labeled.get(row["video_id"])
            if ldf is None or "press_prob_refined" not in ldf.columns:
                continue

            # Pick one fingertip track
            for fname in ["index", "middle", "thumb"]:
                sub = ldf[(ldf["hand"] == "right") & (ldf["finger_name"] == fname)]
                sub = sub.sort_values("frame_idx")
                if len(sub) > 5:
                    break
            else:
                continue

            fig, ax = plt.subplots(figsize=(14, 3))
            ax.fill_between(
                sub["time_sec"], 0, sub["press_smooth"],
                alpha=0.2, color="gray", label="teacher",
            )
            ax.plot(
                sub["time_sec"], sub["press_prob"],
                linewidth=1, color="orange", label="CNN",
            )
            ax.plot(
                sub["time_sec"], sub["press_prob_refined"],
                linewidth=1.5, color="blue", label="CNN+BiLSTM",
            )
            ax.set_ylim(-0.05, 1.15)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("press prob")
            ax.set_title(
                f"Temporal refinement — {row['video_id']} "
                f"right {fname}"
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(
                out_base / "bilstm_timeline.png", dpi=120,
            )
            plt.close(fig)
            print(f"\n  Timeline plot → {out_base / 'bilstm_timeline.png'}")
            break
    except Exception as e:
        print(f"  Timeline plot skipped: {e}")

    # Save BiLSTM model
    model_path = out_base / "bilstm_refiner.pt"
    torch.save(bilstm.state_dict(), model_path)
    print(f"  Model saved → {model_path}")
    print("STEP 5 COMPLETE.\n")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    params = resolve_params(args)

    print(f"Pipeline mode : {params['mode']}")
    print(f"Parameters    : {json.dumps({k: v for k, v in params.items() if k not in ('output_dir', 'cache_dir')}, indent=2)}")
    print()

    step = args.step
    manifest, dataset = None, None
    all_landmarks, labeled, cnn_model = None, None, None

    if step is None or step == 1:
        manifest, dataset = run_step1(params)

    if step is None or step == 2:
        all_landmarks = run_step2(params, manifest, dataset)

    if step is None or step == 3:
        labeled = run_step3(params, manifest, dataset, all_landmarks)

    if step is None or step == 4:
        cnn_model, labeled = run_step4(params, manifest, dataset, labeled)

    if step is None or step == 5:
        run_step5(params, manifest, dataset, labeled, cnn_model)


if __name__ == "__main__":
    main()
