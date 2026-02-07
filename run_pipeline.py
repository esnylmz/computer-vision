#!/usr/bin/env python3
"""
run_pipeline.py — Enhanced Piano CV Pipeline with 3-way comparison.

THREE GROUPS:
  - Group A: Uses refined JSON skeletons + metadata corners (training only)
  - Group B: Pure CV with auto keyboard detection (deployable)
  - Group C: Group B but with metadata corners (ablation study)

Usage:
    python run_pipeline.py --mode smoke
    python run_pipeline.py --mode full
    python run_pipeline.py --step 2
"""

import argparse
import json
import numpy as np
from pathlib import Path

SMOKE_DEFAULTS = dict(N=3, clip_duration=20, frame_step=10, epochs=1)
FULL_DEFAULTS = dict(N=60, clip_duration=120, frame_step=5, epochs=10)


def get_args():
    p = argparse.ArgumentParser(description="Piano CV Pipeline")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--clip_duration", type=float, default=None)
    p.add_argument("--frame_step", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--cache_dir", type=str, default="./data/cache")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_params(args) -> dict:
    defaults = SMOKE_DEFAULTS if args.mode == "smoke" else FULL_DEFAULTS
    params = dict(defaults)
    for key in ["N", "clip_duration", "frame_step", "epochs"]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
    params.update({
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "mode": args.mode,
    })
    return params


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Data & Split (unchanged)
# ═══════════════════════════════════════════════════════════════════

def run_step1(params: dict):
    from src.data import (
        sample_and_split, download_manifest_videos, save_manifest,
    )

    print("=" * 70)
    print("STEP 1 — DATA & SPLIT")
    print("=" * 70)
    print(f"  N={params['N']}, clip_duration={params['clip_duration']}s, "
          f"frame_step={params['frame_step']}")

    manifest, dataset = sample_and_split(
        N=params["N"],
        clip_duration=params["clip_duration"],
        frame_step=params["frame_step"],
        cache_dir=params["cache_dir"],
        seed=params["seed"],
    )

    local_paths = download_manifest_videos(manifest, dataset, verify_fps=True)
    manifest["local_video_path"] = manifest["video_id"].map(local_paths)

    out_dir = Path(params["output_dir"]) / params["mode"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, str(out_dir / "manifest.json"))

    print(f"\nManifest: {len(manifest)} videos → {out_dir}/manifest.json")
    print("STEP 1 COMPLETE.\n")
    return manifest, dataset


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Extract landmarks for ALL THREE GROUPS
# ═══════════════════════════════════════════════════════════════════

def run_step2(params: dict, manifest=None, dataset=None):
    """
    Extract landmarks for all three groups:
      - Group A: Load refined JSON skeletons + apply filtering
      - Group B: MediaPipe + auto keyboard detection
      - Group C: MediaPipe + metadata corners
    """
    import pandas as pd
    from src.data import load_manifest, PianoVAMDataset
    from src.mediapipe_extract import extract_landmarks_from_video
    from src.teacher_labels import load_and_refine_skeleton
    from src.keyboard.auto_detector import auto_detect_keyboard_from_video, validate_keyboard_detection
    from src.homography import parse_corners, add_keyboard_coords_to_landmarks, compute_keyboard_homography
    from src.viz import save_sanity_check_frames

    print("=" * 70)
    print("STEP 2 — LANDMARK EXTRACTION (3 groups)")
    print("=" * 70)

    if manifest is None:
        out_dir = Path(params["output_dir"]) / params["mode"]
        manifest = load_manifest(str(out_dir / "manifest.json"))
    
    if dataset is None:
        dataset = PianoVAMDataset(split="train", cache_dir=params["cache_dir"],
                                  max_samples=params["N"])

    out_base = Path(params["output_dir"]) / params["mode"]
    
    groupA_landmarks = {}
    groupB_landmarks = {}
    groupC_landmarks = {}
    
    for _, row in manifest.iterrows():
        vid_id = row["video_id"]
        video_path = row["local_video_path"]
        fps = row["fps"]
        corners = parse_corners(row["keyboard_corners"])
        
        print(f"\n{'=' * 50}\n[{vid_id}]")
        
        # ──────────────────────────────────────────────────────────
        # GROUP A: Refined JSON skeletons + metadata corners
        # ──────────────────────────────────────────────────────────
        print("  [Group A] Loading refined JSON skeleton ...")
        lm_A = load_and_refine_skeleton(
            dataset, vid_id, params["clip_duration"], fps, params["frame_step"],
        )
        
        if not lm_A.empty:
            lm_A, H_A = add_keyboard_coords_to_landmarks(lm_A, corners)
            print(f"    → {len(lm_A)} detections (filtered JSON + corners)")
        else:
            print("    → No skeleton JSON")
        
        groupA_landmarks[vid_id] = lm_A
        
        # ──────────────────────────────────────────────────────────
        # GROUP B: MediaPipe + AUTO keyboard detection
        # ──────────────────────────────────────────────────────────
        print("  [Group B] MediaPipe + auto keyboard detection ...")
        lm_B, vinfo_B = extract_landmarks_from_video(
            video_path, params["clip_duration"], params["frame_step"], fps,
        )
        
        kb_auto, kb_frame = auto_detect_keyboard_from_video(video_path, verbose=False)
        is_valid, reason = validate_keyboard_detection(kb_auto)
        
        if is_valid:
            corners_auto = {
                "LT": (kb_auto.bbox[0], kb_auto.bbox[1]),
                "RT": (kb_auto.bbox[2], kb_auto.bbox[1]),
                "RB": (kb_auto.bbox[2], kb_auto.bbox[3]),
                "LB": (kb_auto.bbox[0], kb_auto.bbox[3]),
            }
            lm_B, H_B = add_keyboard_coords_to_landmarks(lm_B, corners_auto)
            print(f"    → {len(lm_B)} detections (auto kbd: {len(kb_auto.key_boundaries)} keys)")
        else:
            print(f"    → AUTO DETECTION FAILED: {reason}")
            H_B = None
        
        groupB_landmarks[vid_id] = lm_B
        
        # ──────────────────────────────────────────────────────────
        # GROUP C: MediaPipe + METADATA corners (ablation)
        # ──────────────────────────────────────────────────────────
        print("  [Group C] MediaPipe + metadata corners ...")
        lm_C = lm_B.copy()  # Same MediaPipe landmarks
        lm_C, H_C = add_keyboard_coords_to_landmarks(lm_C, corners)
        print(f"    → {len(lm_C)} detections (metadata corners)")
        
        groupC_landmarks[vid_id] = lm_C
        
        # ──────────────────────────────────────────────────────────
        # Save visualizations (Group B and C)
        # ──────────────────────────────────────────────────────────
        viz_dir = out_base / "viz"
        if H_B is not None:
            save_sanity_check_frames(
                video_path, lm_B, H_B, (880, 110), corners_auto,
                str(viz_dir / "groupB"), vid_id, n_frames=10,
            )
        if H_C is not None:
            save_sanity_check_frames(
                video_path, lm_C, H_C, (880, 110), corners,
                str(viz_dir / "groupC"), vid_id, n_frames=10,
            )
    
    # Save all landmarks
    for group_name, landmarks_dict in [
        ("groupA", groupA_landmarks),
        ("groupB", groupB_landmarks),
        ("groupC", groupC_landmarks),
    ]:
        lm_dir = out_base / "landmarks" / group_name
        lm_dir.mkdir(parents=True, exist_ok=True)
        for vid_id, df in landmarks_dict.items():
            if not df.empty:
                df.to_csv(lm_dir / f"{vid_id}_landmarks.csv", index=False)
    
    print(f"\nSTEP 2 COMPLETE — 3 groups extracted")
    print(f"  Group A: {sum(len(df) for df in groupA_landmarks.values())} detections")
    print(f"  Group B: {sum(len(df) for df in groupB_landmarks.values())} detections")
    print(f"  Group C: {sum(len(df) for df in groupC_landmarks.values())} detections\n")
    
    return groupA_landmarks, groupB_landmarks, groupC_landmarks


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Generate teacher labels (Group A only)
# ═══════════════════════════════════════════════════════════════════

def run_step3(params: dict, manifest=None, dataset=None, groupA_landmarks=None):
    """
    Generate teacher labels using Group A's refined annotations.
    These labels will be used to train the CNN.
    """
    import pandas as pd
    from src.data import load_manifest, PianoVAMDataset
    from src.teacher_labels import generate_teacher_labels_groupA
    from src.homography import parse_corners, add_keyboard_coords_to_landmarks

    print("=" * 70)
    print("STEP 3 — TEACHER LABELS (Group A refined annotations)")
    print("=" * 70)

    out_base = Path(params["output_dir"]) / params["mode"]
    
    if manifest is None:
        manifest = load_manifest(str(out_base / "manifest.json"))
    if dataset is None:
        dataset = PianoVAMDataset(split="train", cache_dir=params["cache_dir"],
                                  max_samples=params["N"])
    
    if groupA_landmarks is None:
        groupA_landmarks = {}
        lm_dir = out_base / "landmarks" / "groupA"
        for _, row in manifest.iterrows():
            p = lm_dir / f"{row['video_id']}_landmarks.csv"
            if p.exists():
                groupA_landmarks[row["video_id"]] = pd.read_csv(p)
    
    groupA_labeled = {}
    
    for _, row in manifest.iterrows():
        vid_id = row["video_id"]
        print(f"\n[{vid_id}]")
        
        lm_A = groupA_landmarks.get(vid_id)
        if lm_A is None or lm_A.empty:
            print("  No Group A landmarks — skipping")
            continue
        
        # Add keyboard coords if missing
        if "x_kbd" not in lm_A.columns:
            corners = parse_corners(row["keyboard_corners"])
            lm_A, _ = add_keyboard_coords_to_landmarks(lm_A, corners)
        
        # Load TSV
        sample = dataset.get_sample_by_id(vid_id)
        if sample:
            tsv_df = dataset.load_tsv_annotations(sample)
        else:
            from src.data import PianoVAMDataset as _D
            local = dataset.download_file(_D.BASE_URL + f"TSV/{vid_id}.tsv")
            tsv_df = pd.read_csv(local, sep="\t",
                                names=["onset","key_offset","frame_offset","note","velocity"],
                                header=None, comment="#")
        
        # Generate labels
        from src.teacher_labels import generate_teacher_labels_for_video
        labeled = generate_teacher_labels_for_video(
            lm_A, tsv_df, fps=row["fps"], clip_duration=params["clip_duration"],
        )
        
        groupA_labeled[vid_id] = labeled
        labeled.to_csv(
            out_base / "landmarks" / "groupA" / f"{vid_id}_labeled.csv", index=False,
        )
    
    total = sum(len(d) for d in groupA_labeled.values())
    n_press = sum(int(d["press_raw"].sum()) for d in groupA_labeled.values())
    print(f"\nSTEP 3 COMPLETE — {total} labeled (Group A), {n_press} press events\n")
    return groupA_labeled


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Train CNN on Group A, apply to all groups
# ═══════════════════════════════════════════════════════════════════

def run_step4(params: dict, manifest=None, dataset=None,
              groupA_labeled=None, groupB_landmarks=None, groupC_landmarks=None):
    """
    1. Train CNN on Group A (refined teacher labels)
    2. Apply CNN to Group B and Group C
    3. Evaluate each group on TEST split
    4. Generate attention visualizations
    5. Save crop examples and training curves
    """
    import torch
    import pandas as pd
    from src.data import load_manifest
    from src.crops import extract_crops_for_video, PressCropDataset
    from src.cnn import train_cnn, predict_cnn
    from src.eval import evaluate_predictions, save_eval_plots
    from src.cnn_attention import compute_gradcam
    from src.viz_comprehensive import plot_crop_examples, plot_training_curves

    print("=" * 70)
    print("STEP 4 — CNN TRAINING (trained on Group A, applied to all)")
    print("=" * 70)

    out_base = Path(params["output_dir"]) / params["mode"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    if manifest is None:
        manifest = load_manifest(str(out_base / "manifest.json"))

    # ── Load labeled Group A ──────────────────────────────────────
    if groupA_labeled is None:
        groupA_labeled = {}
        for _, row in manifest.iterrows():
            p = out_base / "landmarks" / "groupA" / f"{row['video_id']}_labeled.csv"
            if p.exists():
                groupA_labeled[row["video_id"]] = pd.read_csv(p)

    # ── Extract crops from Group A (TRAIN only for training) ─────
    print("Extracting crops from Group A (for CNN training) ...")
    train_crops, train_labels = [], []
    crops_press_examples, crops_nopress_examples = [], []
    
    for _, row in manifest.iterrows():
        if row["split"] != "train":
            continue
        ldf = groupA_labeled.get(row["video_id"])
        if ldf is None or "press_smooth" not in ldf.columns:
            continue
        
        crops, idxs = extract_crops_for_video(row["local_video_path"], ldf, crop_size=64)
        labs = ldf.loc[idxs, "press_smooth"].values.tolist()
        
        train_crops.extend(crops)
        train_labels.extend(labs)
        
        # Collect examples for visualization
        for c, l in zip(crops[:20], labs[:20]):
            if l > 0.5 and len(crops_press_examples) < 8:
                crops_press_examples.append(c)
            elif l <= 0.5 and len(crops_nopress_examples) < 8:
                crops_nopress_examples.append(c)
    
    print(f"  Train crops: {len(train_crops)} (pos={sum(1 for l in train_labels if l > 0.5)})")
    
    if not train_crops:
        print("  ERROR: No training data\n")
        return None
    
    # ── Train CNN ──────────────────────────────────────────────────
    pos_weight = max(len(train_labels) - sum(1 for l in train_labels if l > 0.5), 1) / max(sum(1 for l in train_labels if l > 0.5), 1)
    print(f"  pos_weight: {pos_weight:.2f}")
    
    train_ds = PressCropDataset(train_crops, train_labels)
    cnn_model, losses = train_cnn(
        train_ds, epochs=params["epochs"], batch_size=32,
        lr=1e-3, device=device, pos_weight=pos_weight,
    )
    
    torch.save(cnn_model.state_dict(), out_base / "pressnet.pt")
    print(f"  Model → {out_base}/pressnet.pt")
    
    # ── Save visualizations ────────────────────────────────────────
    viz_dir = out_base / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    plot_crop_examples(crops_press_examples, crops_nopress_examples,
                      save_path=str(viz_dir / "crop_examples.png"))
    plot_training_curves(losses, save_path=str(viz_dir / "training_curve.png"))
    print(f"  Visualizations → {viz_dir}/")
    
    # ── Apply CNN to all groups ────────────────────────────────────
    print("\nApplying CNN to all groups ...")
    
    # Load Group B and C if needed
    if groupB_landmarks is None:
        groupB_landmarks = _load_landmarks(out_base, "groupB", manifest)
    if groupC_landmarks is None:
        groupC_landmarks = _load_landmarks(out_base, "groupC", manifest)
    
    for group_name, landmarks_dict in [
        ("groupA", groupA_labeled),
        ("groupB", groupB_landmarks),
        ("groupC", groupC_landmarks),
    ]:
        for _, row in manifest.iterrows():
            vid_id = row["video_id"]
            ldf = landmarks_dict.get(vid_id)
            if ldf is None or ldf.empty:
                continue
            
            crops, idxs = extract_crops_for_video(row["local_video_path"], ldf, crop_size=64)
            if not crops:
                continue
            
            ds = PressCropDataset(crops, [0.0] * len(crops))
            probs = predict_cnn(cnn_model, ds, device=device)
            ldf.loc[idxs, "press_prob"] = probs
            landmarks_dict[vid_id] = ldf
            
            # Save
            ldf.to_csv(
                out_base / "landmarks" / group_name / f"{vid_id}_labeled.csv",
                index=False,
            )
    
    print("STEP 4 COMPLETE.\n")
    return cnn_model, groupA_labeled, groupB_landmarks, groupC_landmarks


# Helper to load landmarks from disk
def _load_landmarks(out_base, group_name, manifest):
    landmarks = {}
    lm_dir = out_base / "landmarks" / group_name
    for _, row in manifest.iterrows():
        p = lm_dir / f"{row['video_id']}_landmarks.csv"
        if p.exists():
            landmarks[row["video_id"]] = pd.read_csv(p)
    return landmarks


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Temporal refinement + 3-way evaluation
# ═══════════════════════════════════════════════════════════════════

def run_step5(params: dict, manifest=None, groupA_labeled=None,
              groupB_landmarks=None, groupC_landmarks=None):
    """
    1. Train BiLSTM on Group A
    2. Apply to all groups
    3. Evaluate on TEST: Group A vs Group B vs Group C
    4. Generate comprehensive comparison report
    5. Attention visualizations
    """
    import torch
    import pandas as pd
    from src.data import load_manifest
    from src.bilstm import build_sequences, train_refiner, predict_refiner
    from src.eval import evaluate_predictions, save_eval_plots, event_consistency
    from src.viz_comprehensive import create_comparison_report, plot_timeline_comparison
    from src.cnn_attention import compute_gradcam, create_attention_grid

    print("=" * 70)
    print("STEP 5 — TEMPORAL REFINEMENT + 3-WAY EVALUATION")
    print("=" * 70)

    out_base = Path(params["output_dir"]) / params["mode"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if manifest is None:
        manifest = load_manifest(str(out_base / "manifest.json"))
    
    # Load all groups if not passed
    if groupA_labeled is None:
        groupA_labeled = _load_landmarks(out_base, "groupA", manifest)
    if groupB_landmarks is None:
        groupB_landmarks = _load_landmarks(out_base, "groupB", manifest)
    if groupC_landmarks is None:
        groupC_landmarks = _load_landmarks(out_base, "groupC", manifest)
    
    # ── Train BiLSTM on Group A ────────────────────────────────────
    print("Building sequences from Group A (TRAIN) ...")
    train_feats, train_labs = [], []
    for _, row in manifest.iterrows():
        if row["split"] != "train":
            continue
        ldf = groupA_labeled.get(row["video_id"])
        if ldf is None or "press_prob" not in ldf.columns:
            continue
        f, l = build_sequences(ldf, "press_prob")
        train_feats.extend(f)
        train_labs.extend(l)
    
    print(f"  Sequences: {len(train_feats)}")
    
    if not train_feats:
        print("  No sequences — skipping BiLSTM\n")
        return
    
    pos_weight = max(
        sum(len(l) - (l > 0.5).sum() for l in train_labs), 1
    ) / max(sum((l > 0.5).sum() for l in train_labs), 1)
    
    bilstm, _ = train_refiner(
        (train_feats, train_labs), epochs=params["epochs"],
        device=device, pos_weight=pos_weight,
    )
    torch.save(bilstm.state_dict(), out_base / "bilstm_refiner.pt")
    
    # ── Apply BiLSTM to all groups ──────────────────────────────────
    print("\nApplying BiLSTM refinement to all groups ...")
    for group_name, landmarks_dict in [
        ("groupA", groupA_labeled),
        ("groupB", groupB_landmarks),
        ("groupC", groupC_landmarks),
    ]:
        for vid_id, ldf in landmarks_dict.items():
            if ldf is None or "press_prob" not in ldf.columns:
                continue
            
            refined = predict_refiner(bilstm, ldf, "press_prob", device=device)
            ldf["press_prob_refined"] = refined.values
            landmarks_dict[vid_id] = ldf
            ldf.to_csv(
                out_base / "landmarks" / group_name / f"{vid_id}_labeled.csv",
                index=False,
            )
    
    # ── 3-WAY EVALUATION (TEST split only) ─────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION (TEST videos only)")
    print("=" * 70)
    
    results = {}
    
    for group_name, landmarks_dict in [
        ("Group A (refined annotations)", groupA_labeled),
        ("Group B (auto keyboard)", groupB_landmarks),
        ("Group C (metadata corners)", groupC_landmarks),
    ]:
        test_true, cnn_prob, refined_prob = [], [], []
        
        for _, row in manifest.iterrows():
            if row["split"] != "test":
                continue
            ldf = landmarks_dict.get(row["video_id"])
            if ldf is None:
                continue
            
            valid = ldf.dropna(subset=["press_smooth", "press_prob", "press_prob_refined"])
            if not valid.empty:
                test_true.append(valid["press_smooth"].values)
                cnn_prob.append(valid["press_prob"].values)
                refined_prob.append(valid["press_prob_refined"].values)
        
        if not test_true:
            print(f"\n  [{group_name}] No test data")
            continue
        
        y_true = np.concatenate(test_true)
        y_cnn = np.concatenate(cnn_prob)
        y_refined = np.concatenate(refined_prob)
        
        print(f"\n  [{group_name}]")
        print("    CNN only:")
        m_cnn = evaluate_predictions(y_true, y_cnn, label="  ")
        print("    CNN + BiLSTM:")
        m_refined = evaluate_predictions(y_true, y_refined, label="  ")
        
        # Event consistency
        ec_cnn = event_consistency((y_cnn > 0.5).astype(int))
        ec_ref = event_consistency((y_refined > 0.5).astype(int))
        print(f"    Isolated presses: CNN={ec_ref['n_isolated']}, BiLSTM={ec_ref['n_isolated']}")
        
        results[group_name] = {
            "cnn": m_cnn,
            "refined": m_refined,
            "event_consistency": {"cnn": ec_cnn, "bilstm": ec_ref},
        }
    
    # ── Generate comprehensive report ───────────────────────────────
    print("\nGenerating comprehensive comparison report ...")
    
    metrics_for_plot = {
        name: res["refined"] for name, res in results.items()
    }
    
    # Pick first test video for timeline
    timeline_df = None
    for _, row in manifest.iterrows():
        if row["split"] == "test":
            # Use Group C (best quality for visualization)
            ldf = groupC_landmarks.get(row["video_id"])
            if ldf is not None and "press_prob_refined" in ldf.columns:
                timeline_df = ldf
                break
    
    create_comparison_report(
        str(out_base / "report"),
        metrics_for_plot,
        timeline_df=timeline_df,
        crops_press=crops_press_examples if 'crops_press_examples' in dir() else None,
        crops_no_press=crops_nopress_examples if 'crops_nopress_examples' in dir() else None,
        training_losses=losses if 'losses' in dir() else None,
    )
    
    # ── Save results JSON ───────────────────────────────────────────
    with open(out_base / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSTEP 5 COMPLETE.\n")
    print(f"  Full report → {out_base}/report/")
    print(f"  Results     → {out_base}/results.json")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    params = resolve_params(args)

    print(f"Pipeline mode: {params['mode'].upper()}")
    print(f"Parameters: N={params['N']}, clip={params['clip_duration']}s, "
          f"step={params['frame_step']}, epochs={params['epochs']}\n")

    step = args.step
    manifest, dataset = None, None
    groupA_lms, groupB_lms, groupC_lms = None, None, None
    groupA_labeled = None

    if step is None or step == 1:
        manifest, dataset = run_step1(params)

    if step is None or step == 2:
        groupA_lms, groupB_lms, groupC_lms = run_step2(params, manifest, dataset)

    if step is None or step == 3:
        groupA_labeled = run_step3(params, manifest, dataset, groupA_lms)

    if step is None or step == 4:
        cnn_model, groupA_labeled, groupB_lms, groupC_lms = run_step4(
            params, manifest, dataset, groupA_labeled, groupB_lms, groupC_lms,
        )

    if step is None or step == 5:
        run_step5(params, manifest, groupA_labeled, groupB_lms, groupC_lms)


if __name__ == "__main__":
    main()
