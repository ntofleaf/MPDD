# -*- coding: utf-8 -*-
"""
Extract Wav2Vec2 features from original audio files without sliding windows.

Default behavior:
- recursively scan audio files under root_dir
- keep frame-level sequence features with shape [T, C]
- preserve input subfolder structure in the output directory
- support both local model paths and public HuggingFace model ids
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


AUDIO_SUFFIXES = {".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3", ".m4a", ".M4A"}
DEFAULT_ROOT_DIR = "/home/disk2/zelin/MM2025_Challenge/withdisease/basicdisease/原始数据"
DEFAULT_SAVE_DIR = "/home/disk2/shangming/feature_extract/feature_frame/audio/wav2vec2"
DEFAULT_MODEL_ID = "facebook/wav2vec2-base-960h"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Wav2Vec2 frame-level or utterance-level features without sliding windows."
    )
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR, help="Root directory of original audio files.")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, help="Output root directory.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Local path or HuggingFace model id for Wav2Vec2Model.",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Local path or HuggingFace model id for Wav2Vec2Processor.",
    )
    parser.add_argument("--feature_level", type=str, default="FRAME", choices=["UTTERANCE", "FRAME"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resample", action="store_true", default=True)
    return parser.parse_args()


def collect_audio_files(audio_dir: str):
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file_name in files:
            if Path(file_name).suffix in AUDIO_SUFFIXES:
                audio_files.append(os.path.join(root, file_name))
    return sorted(audio_files)


def write_feature_to_npy(feature, npy_file, feature_level):
    feature = np.asarray(feature).squeeze()
    if feature_level == "UTTERANCE":
        if feature.ndim != 1:
            feature = np.mean(feature, axis=0)
    elif feature.ndim == 1:
        feature = feature[np.newaxis, :]

    np.save(npy_file, feature)


def load_audio(wav_file: str, target_sr: int = 16000, resample: bool = True):
    audio, sampling_rate = sf.read(wav_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if resample and sampling_rate != target_sr:
        audio = resampy.resample(audio, sampling_rate, target_sr, filter="kaiser_fast")
        sampling_rate = target_sr

    min_length = 3200
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)), mode="constant")

    return audio.astype(np.float32), sampling_rate


def load_model_and_processor(model_path: str, processor_path: str, device: torch.device):
    model = Wav2Vec2Model.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    model.to(device)
    model.eval()
    return model, processor


def extract(audio_files, audio_dir, feature_level, model, processor, save_dir, overwrite=False, gpu=None, resample=True):
    start_time = time.time()
    device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")

    dir_name = f"wav2vec2-{feature_level[:3].lower()}"
    out_dir = os.path.join(save_dir, dir_name)
    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f'save_dir "{out_dir}" already exists, set overwrite=True if needed.')
    os.makedirs(out_dir, exist_ok=True)

    for idx, wav_file in enumerate(audio_files, 1):
        try:
            file_name = os.path.basename(wav_file)
            audio_id = os.path.splitext(file_name)[0]

            relative_path = os.path.relpath(os.path.dirname(wav_file), audio_dir)
            output_subdir = os.path.join(out_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            print(f'Processing "{os.path.join(relative_path, file_name)}" ({idx}/{len(audio_files)})...')

            audio, sampling_rate = load_audio(wav_file, target_sr=16000, resample=resample)
            inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                features = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [T, C]

            npy_file = os.path.join(output_subdir, f"{audio_id}.npy")
            write_feature_to_npy(features, npy_file, feature_level)
            print(f"  Extracted shape: {features.shape} -> {npy_file}")

        except Exception as e:
            print(f"  [ERROR] processing failed for {wav_file}: {e}")
            continue

    end_time = time.time()
    print(f"Total time used: {end_time - start_time:.1f}s.")


def main():
    args = parse_args()

    audio_dir = args.root_dir
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"root_dir does not exist or is not a directory: {audio_dir}")

    audio_files = collect_audio_files(audio_dir)
    print(f'Find total "{len(audio_files)}" audio files.')
    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found under: {audio_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading Wav2Vec2 model on {device}...")
    model, processor = load_model_and_processor(args.model_path, args.processor_path, device)
    print("Model loaded successfully!")

    extract(
        audio_files,
        audio_dir=audio_dir,
        feature_level=args.feature_level,
        model=model,
        processor=processor,
        save_dir=args.save_dir,
        overwrite=args.overwrite,
        gpu=args.gpu,
        resample=args.resample,
    )


if __name__ == "__main__":
    main()
