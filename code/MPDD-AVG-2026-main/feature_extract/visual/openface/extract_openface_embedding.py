# -*- coding: utf-8 -*-
"""
Extract OpenFace features from original videos (no sliding window).

This version stores only the frame-level "all" feature subset and writes it
directly as:
    <save_dir>/<id>/V_1.npy
    <save_dir>/<id>/V_2.npy
    ...

Before running OpenFace for a video, the script checks whether the target
output npy already exists. If it exists and overwrite is not enabled, that
video is skipped.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4", ".AVI")
VIDEO_SLOT_PATTERN = re.compile(r"(V_(\d+))", re.IGNORECASE)


def check_openface_models(openface_exe: str) -> tuple[bool, list[str]]:
    model_dir = os.path.join(os.path.dirname(openface_exe), "model")
    required_files = [
        "patch_experts/cen_patches_0.25_of.dat",
        "main_ceclm_general.txt",
        "tris_68.txt",
    ]
    missing_files = []
    for rel in required_files:
        if not os.path.exists(os.path.join(model_dir, rel)):
            missing_files.append(rel)
    return len(missing_files) == 0, missing_files


def infer_video_slot_name(video_path: str, fallback_index: int) -> str:
    stem = Path(video_path).stem
    match = VIDEO_SLOT_PATTERN.search(stem)
    if match:
        return f"V_{int(match.group(2))}"
    return f"V_{fallback_index}"


def collect_video_files(id_folder_path: str) -> list[str]:
    video_files: list[str] = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(id_folder_path, f"*{ext}")))

    def sort_key(path: str):
        slot_match = VIDEO_SLOT_PATTERN.search(Path(path).stem)
        slot_idx = int(slot_match.group(2)) if slot_match else 10**9
        return (slot_idx, Path(path).name.lower())

    return sorted(set(video_files), key=sort_key)


def extract_features_from_video(
    video_path: str,
    id_output_folder: str,
    openface_exe: str,
    slot_name: str,
    overwrite: bool = False,
) -> bool:
    target_npy_path = os.path.join(id_output_folder, f"{slot_name}.npy")
    if os.path.exists(target_npy_path) and not overwrite:
        print(f"  [SKIP] {os.path.basename(video_path)} -> {slot_name}.npy already exists")
        return False

    feature_save_dir = os.path.join(id_output_folder, slot_name)
    os.makedirs(feature_save_dir, exist_ok=True)

    command = [openface_exe, "-f", video_path, "-out_dir", feature_save_dir]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] OpenFace failed for {os.path.basename(video_path)}, code={result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:300]}")
            return False
        csv_files = glob.glob(os.path.join(feature_save_dir, "*.csv"))
        return len(csv_files) > 0
    except Exception as e:
        print(f"  [ERROR] run OpenFace failed: {e}")
        return False


def process_original_videos(root_folder: str, output_folder: str, openface_exe: str, overwrite: bool = False):
    if not os.path.exists(openface_exe):
        print(f"[ERROR] OpenFace executable not found: {openface_exe}")
        return

    models_ok, missing = check_openface_models(openface_exe)
    if not models_ok:
        print("[WARN] OpenFace model files are incomplete.")
        print(f"[WARN] missing: {missing}")

    os.makedirs(output_folder, exist_ok=True)

    total_videos = 0
    processed_videos = 0
    skipped_videos = 0
    failed_videos = 0

    for id_folder in sorted(os.listdir(root_folder)):
        id_folder_path = os.path.join(root_folder, id_folder)
        if not os.path.isdir(id_folder_path):
            continue

        id_output_folder = os.path.join(output_folder, id_folder)
        os.makedirs(id_output_folder, exist_ok=True)

        video_files = collect_video_files(id_folder_path)
        if not video_files:
            print(f"[WARN] id={id_folder} no video files found")
            continue

        for video_index, video_file in enumerate(video_files, start=1):
            total_videos += 1
            slot_name = infer_video_slot_name(video_file, fallback_index=video_index)
            target_npy_path = os.path.join(id_output_folder, f"{slot_name}.npy")

            if os.path.exists(target_npy_path) and not overwrite:
                skipped_videos += 1
                print(f"[SKIP] id={id_folder} {slot_name}.npy already exists")
                continue

            ok = extract_features_from_video(
                video_file,
                id_output_folder,
                openface_exe,
                slot_name=slot_name,
                overwrite=overwrite,
            )
            if ok:
                processed_videos += 1
            else:
                failed_videos += 1

    print("=" * 60)
    print("OpenFace extraction done")
    print(f"total: {total_videos}")
    print(f"processed: {processed_videos}")
    print(f"skipped: {skipped_videos}")
    print(f"failed: {failed_videos}")
    print("=" * 60)


def cleanup_extracted_frame_images(root_folder: str):
    removed_count = 0
    for current_root, _, files in os.walk(root_folder):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(current_root, filename)
                try:
                    os.remove(image_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  [WARN] remove image failed: {image_path} | {e}")
    print(f"[CLEANUP] removed image frames: {removed_count}")
    return removed_count


def convert_csv_to_npy(root_folder: str, overwrite: bool = False):
    csv_files = glob.glob(os.path.join(root_folder, "**", "*.csv"), recursive=True)
    converted_count = 0
    skipped_count = 0
    failed_count = 0

    for csv_file in csv_files:
        try:
            src_path = Path(csv_file)
            slot_name = src_path.parent.name
            if not VIDEO_SLOT_PATTERN.fullmatch(slot_name):
                continue

            data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
            if data.size == 0:
                continue
            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            features = data[:, 4:].astype(np.float32)
            target_path = src_path.parent.parent / f"{slot_name}.npy"
            if target_path.exists() and not overwrite:
                skipped_count += 1
                continue

            np.save(target_path, features)
            converted_count += 1
        except Exception as e:
            print(f"  [ERROR] convert failed: {os.path.basename(csv_file)} | {e}")
            failed_count += 1

    print("=" * 60)
    print("CSV to NPY done")
    print(f"success: {converted_count}")
    print(f"skipped: {skipped_count}")
    print(f"failed: {failed_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract OpenFace features from original videos (no sliding window).")
    parser.add_argument("--root_dir", type=str, default="/home/disk2/视频数据")
    parser.add_argument("--save_dir", type=str, default="/home/disk2/yiming/video/openface")
    parser.add_argument(
        "--openface_exe",
        type=str,
        default="/home/disk2/zelin/MM2025_Challenge/0324-换滑窗/feature_extraction/OpenFace/build/bin/FeatureExtraction",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--convert_to_npy", action="store_true", default=True)
    args = parser.parse_args()

    process_original_videos(
        root_folder=args.root_dir,
        output_folder=args.save_dir,
        openface_exe=args.openface_exe,
        overwrite=args.overwrite,
    )

    if args.convert_to_npy:
        convert_csv_to_npy(args.save_dir, overwrite=args.overwrite)

    cleanup_extracted_frame_images(args.save_dir)
