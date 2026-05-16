# -*- coding: utf-8 -*-
"""
Extract DenseNet121 frame-level visual features from original videos (no sliding window).
Default behavior:
- no windowing
- frame-level output: [T, C]
- full-frame extraction (frame_sample_rate=1)
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import densenet121
from torchvision.transforms import transforms
from tqdm import tqdm


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, transform=None):
        self.frame_dir = frame_dir
        self.transform = transform
        self.frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def extract_features_from_video(
    video_path,
    feature_save_path,
    model,
    transform,
    device,
    batch_size=128,
    num_workers=8,
    frame_sample_rate=1,
):
    temp_frame_dir = f"temp_frames_{os.getpid()}_{os.path.basename(video_path)}"
    os.makedirs(temp_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] cannot open video: {video_path}")
        return False

    success, frame = cap.read()
    count = 0
    saved_count = 0
    while success:
        if count % frame_sample_rate == 0:
            frame_path = os.path.join(temp_frame_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        success, frame = cap.read()
        count += 1
    cap.release()

    print(f"  [INFO] extracted frames: {saved_count}/{count}, sample_rate=1/{frame_sample_rate}")

    if saved_count == 0:
        try:
            os.rmdir(temp_frame_dir)
        except OSError:
            pass
        return False

    dataset = FrameDataset(temp_frame_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    features = []
    with torch.no_grad():
        for images in tqdm(data_loader, desc="  extracting", leave=False):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())

    try:
        features = np.concatenate(features, axis=0)  # [T, 1000]
        print(f"  [INFO] feature shape: {features.shape}")
    except Exception as e:
        print(f"  [ERROR] concat features failed: {e}")
        return False

    try:
        os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
        np.save(feature_save_path, features)
    except Exception as e:
        print(f"  [ERROR] save feature failed: {e}")
        return False
    finally:
        for f in glob.glob(os.path.join(temp_frame_dir, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(temp_frame_dir)
        except OSError:
            pass

    return True


def process_original_videos(
    root_folder,
    output_folder,
    overwrite=False,
    batch_size=128,
    num_workers=8,
    frame_sample_rate=1,
    model_name="densenet121",
):
    if model_name != "densenet121":
        print(f"[WARN] unknown model={model_name}, fallback to densenet121")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    os.makedirs(output_folder, exist_ok=True)

    total_videos = 0
    processed_videos = 0
    skipped_videos = 0

    for id_folder in sorted(os.listdir(root_folder)):
        id_folder_path = os.path.join(root_folder, id_folder)
        if not os.path.isdir(id_folder_path):
            continue

        id_output_folder = os.path.join(output_folder, id_folder)
        os.makedirs(id_output_folder, exist_ok=True)

        video_files = []
        video_files += glob.glob(os.path.join(id_folder_path, "*.mp4"))
        video_files += glob.glob(os.path.join(id_folder_path, "*.avi"))
        video_files += glob.glob(os.path.join(id_folder_path, "*.mov"))
        video_files += glob.glob(os.path.join(id_folder_path, "*.MP4"))
        video_files += glob.glob(os.path.join(id_folder_path, "*.AVI"))

        if not video_files:
            print("  [WARN] no video files found")
            continue

        print(f"  [INFO] found {len(video_files)} video files")

        for video_file in video_files:
            video_name = os.path.basename(video_file)
            feature_save_name = f"{os.path.splitext(video_name)[0]}.npy"
            feature_save_path = os.path.join(id_output_folder, feature_save_name)

            total_videos += 1
            if not overwrite and os.path.exists(feature_save_path):
                skipped_videos += 1
                continue

            success = extract_features_from_video(
                video_file,
                feature_save_path,
                model,
                transform,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                frame_sample_rate=frame_sample_rate,
            )
            if success:
                processed_videos += 1

    print("=" * 60)
    print("Done")
    print(f"  total: {total_videos}")
    print(f"  processed: {processed_videos}")
    print(f"  skipped: {skipped_videos}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DenseNet features from original videos (no sliding window).")
    parser.add_argument("--root_dir", type=str, default="/home/disk2/zelin/MM2025_Challenge/withdisease/basicdisease/原始数据")
    parser.add_argument("--save_dir", type=str, default="/home/disk2/shangming/feature_extract/feature_frame/video/densenet")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--frame_sample_rate", type=int, default=1)
    parser.add_argument("--model", type=str, default="densenet121", choices=["densenet121"])
    args = parser.parse_args()

    process_original_videos(
        root_folder=args.root_dir,
        output_folder=args.save_dir,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_sample_rate=args.frame_sample_rate,
        model_name=args.model,
    )
