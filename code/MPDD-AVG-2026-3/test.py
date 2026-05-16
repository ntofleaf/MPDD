from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import REGRESSION_TASK, MPDDElderDataset, collate_batch, load_task_maps, resolve_project_path
from metrics import evaluate_model
from models import TorchcatBaseline


PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {
    "A-V+P": "A-V-P",
    "A-V-G+P": "A-V-G+P",
    "G+P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained MPDD-AVG baseline checkpoint.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--split_csv", default="")
    parser.add_argument("--personality_npy", default="")
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    parser = build_parser(defaults)
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(f"elder_track1_test_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    return logger


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_track_task_dir(root: Path, track: str, subtrack: str, task: str, experiment_name: str) -> Path:
    subtrack_dir = SUBTRACK_LOG_DIRS.get(subtrack, subtrack.replace("+", "-"))
    return root / track / subtrack_dir / task / experiment_name


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def require_checkpoint_value(checkpoint: dict[str, Any], key: str) -> Any:
    value = checkpoint.get(key)
    if value in (None, ""):
        raise KeyError(f"Checkpoint missing required field: {key}")
    return value


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in METRIC_ARRAY_KEYS}


def remap_repo_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        return path.as_posix()
    if path.exists():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    for anchor in ("MPDD-AVG2026", "checkpoints", "logs"):
        if anchor not in path.parts:
            continue
        anchor_index = path.parts.index(anchor)
        candidate = PROJECT_ROOT.joinpath(*path.parts[anchor_index:])
        if candidate.exists() or anchor == "logs":
            return candidate.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(remap_repo_path(args.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    track = require_checkpoint_value(checkpoint, "track")
    task = require_checkpoint_value(checkpoint, "task")
    regression_label = checkpoint.get("regression_label", "")
    subtrack = require_checkpoint_value(checkpoint, "subtrack")
    encoder_type = require_checkpoint_value(checkpoint, "encoder_type")
    audio_feature = require_checkpoint_value(checkpoint, "audio_feature")
    video_feature = require_checkpoint_value(checkpoint, "video_feature")
    data_root = remap_repo_path(args.data_root or require_checkpoint_value(checkpoint, "data_root"))
    split_csv = remap_repo_path(args.split_csv or require_checkpoint_value(checkpoint, "split_csv"))
    personality_npy = remap_repo_path(args.personality_npy or require_checkpoint_value(checkpoint, "personality_npy"))
    target_t = int(require_checkpoint_value(checkpoint, "target_t"))
    experiment_name = checkpoint.get("experiment_name", checkpoint_path.parent.name)

    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    logs_root = resolve_project_path(remap_repo_path(args.logs_dir))
    log_dir = resolve_track_task_dir(logs_root, track, subtrack, task, experiment_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    task_maps = load_task_maps(split_csv, task, regression_label or "label2")
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        phq_map=task_maps.get("test_phq_map"),
        target_t=target_t,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model_kwargs = dict(require_checkpoint_value(checkpoint, "model_kwargs"))
    model = TorchcatBaseline(**model_kwargs).to(device)
    model.load_state_dict(require_checkpoint_value(checkpoint, "model_state"))
    use_regression_head = bool(model_kwargs.get("use_regression_head", False))
    is_regression_task = task == REGRESSION_TASK
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss()) if use_regression_head else nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, criterion, device, task)
    metric_summary = summarize_metrics(metrics)
    checkpoint_rel = to_project_relative_path(checkpoint_path)

    result_payload = {
        "checkpoint": checkpoint_rel,
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "regression_label": regression_label if is_regression_task else "",
        "metrics": metric_summary,
        "predictions_path": "",
    }
    result_path = log_dir / f"test_result_only_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "timestamp": timestamp,
        "mode": "test_only",
        "track": track,
        "task": task,
        "subtrack": subtrack,
        "encoder_type": encoder_type,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "checkpoint": checkpoint_rel,
        "predictions_path": "",
        "Macro-F1": f"{metrics.get('f1', 0.0):.6f}",
        "ACC": f"{metrics.get('acc', 0.0):.6f}",
        "Kappa": f"{metrics.get('kappa', 0.0):.6f}",
        "CCC": f"{metrics['ccc']:.6f}",
        "RMSE": f"{metrics['rmse']:.6f}",
        "MAE": f"{metrics['mae']:.6f}",
        "R2": f"{metrics.get('r2', ''):.6f}" if is_regression_task else "",
    }
    if is_regression_task:
        summary_row["regression_label"] = regression_label
    append_summary_row(log_dir / f"{experiment_name}_test_only.csv", summary_row)
    logger.info("Test-only metrics saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
