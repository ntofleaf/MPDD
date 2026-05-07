from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import REGRESSION_TASK, MPDDElderDataset, collate_batch, infer_input_dims, resolve_project_path
from metrics import evaluate_model
from models import TorchcatBaseline
from train_val_split import create_train_val_split


PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {
    "A-V+P": "A-V-P",
    "A-V-G+P": "A-V-G+P",
    "G+P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}
PATH_ARG_KEYS = {"config", "data_root", "split_csv", "personality_npy", "checkpoints_dir", "logs_dir"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MPDD-AVG baseline with a train/val workflow.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--track", default=defaults["track"], choices=["Track1", "Track2"])
    parser.add_argument("--task", default=defaults["task"], choices=["binary", "ternary", REGRESSION_TASK])
    parser.add_argument("--regression_label", default=defaults.get("regression_label", "label2"), choices=["label2", "label3"])
    parser.add_argument("--subtrack", default=defaults["subtrack"], choices=["A-V+P", "A-V-G+P", "G+P"])
    parser.add_argument("--encoder_type", default=defaults["encoder_type"], choices=["bilstm_mean", "hybrid_attn"])
    parser.add_argument("--audio_feature", default=defaults["audio_feature"])
    parser.add_argument("--video_feature", default=defaults["video_feature"])
    parser.add_argument("--data_root", default=defaults["data_root"])
    parser.add_argument("--split_csv", default=defaults["split_csv"])
    parser.add_argument("--personality_npy", default=defaults["personality_npy"])
    parser.add_argument("--val_ratio", type=float, default=defaults["val_ratio"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--target_t", type=int, default=defaults["target_t"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--hidden_dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument("--min_delta", type=float, default=defaults["min_delta"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--checkpoints_dir", default=defaults["checkpoints_dir"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    parser.add_argument("--experiment_name", default="")
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    parser = build_parser(defaults)
    args, unknown = parser.parse_known_args()
    return args


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"elder_track1_train_{log_file.stem}_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def resolve_track_task_dir(root: Path, track: str, subtrack: str, task: str, experiment_name: str) -> Path:
    subtrack_dir = SUBTRACK_LOG_DIRS.get(subtrack, subtrack.replace("+", "-"))
    return root / track / subtrack_dir / task / experiment_name


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def normalize_path_args(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if key in PATH_ARG_KEYS and value not in (None, ""):
            normalized[key] = to_project_relative_path(value)
        else:
            normalized[key] = value
    return normalized


def build_experiment_name(args: argparse.Namespace) -> str:
    feature_tag = "gait_only" if args.subtrack == "G+P" else f"{args.audio_feature}__{args.video_feature}"
    if args.task == REGRESSION_TASK:
        return args.experiment_name or (
            f"{args.track.lower()}_{args.task}_{args.regression_label}_{args.subtrack}_{args.encoder_type}_{feature_tag}"
        )
    return args.experiment_name or f"{args.track.lower()}_{args.task}_{args.subtrack}_{args.encoder_type}_{feature_tag}"


def get_num_classes(task: str, regression_label: str) -> int:
    if task == "binary":
        return 2
    if task == "ternary":
        return 3
    if task == REGRESSION_TASK:
        return 2 if regression_label == "label2" else 3
    raise ValueError(f"Unsupported task: {task}")


def build_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in METRIC_ARRAY_KEYS}


def get_selection_metric_name(task: str) -> str:
    return "ccc" if task == REGRESSION_TASK else "f1"


def main() -> None:
    args = parse_args()

    experiment_name = build_experiment_name(args)
    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    checkpoints_root = resolve_project_path(args.checkpoints_dir)
    logs_root = resolve_project_path(args.logs_dir)
    checkpoints_dir = resolve_track_task_dir(checkpoints_root, args.track, args.subtrack, args.task, experiment_name)
    log_dir = resolve_track_task_dir(logs_root, args.track, args.subtrack, args.task, experiment_name)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / f"result_{timestamp}.log")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    split_payload = create_train_val_split(
        split_csv=args.split_csv,
        task=args.task,
        val_ratio=args.val_ratio,
        regression_label=args.regression_label,
    )
    setup_seed(args.seed)
    use_regression_head = True
    is_regression_task = args.task == REGRESSION_TASK
    train_dataset = MPDDElderDataset(
        data_root=args.data_root,
        label_map=split_payload["train_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=split_payload.get("train_phq_map"),
        target_t=args.target_t,
    )
    val_dataset = MPDDElderDataset(
        data_root=args.data_root,
        label_map=split_payload["val_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=split_payload.get("val_phq_map"),
        target_t=args.target_t,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    input_dims = infer_input_dims(train_dataset)
    num_classes = get_num_classes(args.task, args.regression_label)
    model_kwargs = {
        "subtrack": args.subtrack,
        "num_classes": num_classes,
        "is_regression": False,
        "use_regression_head": use_regression_head,
        "audio_dim": input_dims["audio_dim"],
        "video_dim": input_dims["video_dim"],
        "gait_dim": input_dims["gait_dim"],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "encoder_type": args.encoder_type,
    }
    model = TorchcatBaseline(**model_kwargs).to(device)
    num_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info("DataParallel enabled: using %d GPUs", num_gpus)

    class_weights = build_class_weights(
        [int(sample["label"]) for sample in train_dataset.samples],
        num_classes=num_classes,
        device=device,
    )
    criterion = (nn.CrossEntropyLoss(weight=class_weights), nn.MSELoss())
    selection_metric_name = get_selection_metric_name(args.task)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    logger.info("Experiment: %s", experiment_name)
    logger.info("Device: %s", device)
    logger.info("Train/Val: %d / %d", len(train_dataset), len(val_dataset))

    history_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, Any] | None = None
    best_checkpoint_path = checkpoints_dir / f"best_model_{timestamp}.pth"
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch["label"].to(device)
            outputs = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            criterion_cls, criterion_reg = criterion
            phq9 = batch["phq9"].to(device)
            logits, reg_out = outputs
            loss = criterion_cls(logits, labels) + criterion_reg(reg_out, phq9)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)

        scheduler.step()
        train_loss = running_loss / max(1, len(train_dataset))
        val_metrics = evaluate_model(model, val_loader, criterion, device, args.task)
        history_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_ccc": round(val_metrics["ccc"], 6),
            "val_rmse": round(val_metrics["rmse"], 6),
            "val_mae": round(val_metrics["mae"], 6),
        }
        history_row["val_f1"] = round(val_metrics["f1"], 6)
        history_row["val_acc"] = round(val_metrics["acc"], 6)
        history_row["val_kappa"] = round(val_metrics["kappa"], 6)
        history_row["val_cls_loss"] = round(val_metrics["cls_loss"], 6)
        history_row["val_reg_loss"] = round(val_metrics["reg_loss"], 6)
        if is_regression_task:
            history_row["val_r2"] = round(val_metrics["r2"], 6)
        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_f1=%.6f val_acc=%.6f "
            "val_kappa=%.6f val_ccc=%.6f val_rmse=%.6f val_mae=%.6f",
            epoch, args.epochs, train_loss, val_metrics["f1"], val_metrics["acc"],
            val_metrics["kappa"], val_metrics["ccc"], val_metrics["rmse"], val_metrics["mae"],
        )
        history_rows.append(history_row)

        current_score = float(val_metrics["selection_score"])
        if current_score > best_score + args.min_delta:
            best_score = current_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_summary = summarize_metrics(val_metrics)
            epochs_without_improve = 0
            _state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "model_state": _state_dict,
                    "model_kwargs": model_kwargs,
                    "track": args.track,
                    "task": args.task,
                    "subtrack": args.subtrack,
                    "encoder_type": args.encoder_type,
                    "audio_feature": args.audio_feature,
                    "video_feature": args.video_feature,
                    "regression_label": args.regression_label if is_regression_task else "",
                    "data_root": to_project_relative_path(args.data_root),
                    "split_csv": to_project_relative_path(args.split_csv),
                    "personality_npy": to_project_relative_path(args.personality_npy),
                    "target_t": args.target_t,
                    "seed": args.seed,
                    "experiment_name": experiment_name,
                    "best_epoch": epoch,
                    "best_val_metrics": best_val_summary,
                    "metric_split": "val",
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without a valid validation checkpoint.")
    best_val_summary = summarize_metrics(best_val_metrics)

    history_path = log_dir / f"history_{timestamp}.csv"
    with open(history_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    best_checkpoint_rel = to_project_relative_path(best_checkpoint_path)
    history_rel = to_project_relative_path(history_path)
    result_payload = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "encoder_type": args.encoder_type,
        "audio_feature": args.audio_feature,
        "video_feature": args.video_feature,
        "regression_label": args.regression_label if is_regression_task else "",
        "best_epoch": best_epoch,
        "selection_metric": selection_metric_name,
        "best_val_metrics": best_val_summary,
        "checkpoint_path": best_checkpoint_rel,
        "history_path": history_rel,
        "predictions_path": "",
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "config": normalize_path_args(vars(args)),
    }
    result_path = log_dir / f"train_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "encoder_type": args.encoder_type,
        "audio_feature": args.audio_feature,
        "video_feature": args.video_feature,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint_path": best_checkpoint_rel,
        "predictions_path": "",
        "metric_split": "val",
        "selection_metric": selection_metric_name,
        "selection_score": f"{best_val_summary.get('selection_score', 0.0):.6f}",
        "Macro-F1": f"{best_val_summary.get('f1', 0.0):.6f}",
        "ACC": f"{best_val_summary.get('acc', 0.0):.6f}",
        "Kappa": f"{best_val_summary.get('kappa', 0.0):.6f}",
        "CCC": f"{best_val_summary['ccc']:.6f}",
        "RMSE": f"{best_val_summary['rmse']:.6f}",
        "MAE": f"{best_val_summary['mae']:.6f}",
        "R2": f"{best_val_summary.get('r2', 0.0):.6f}" if is_regression_task else "",
    }
    if is_regression_task:
        summary_row["regression_label"] = args.regression_label
    append_summary_row(log_dir / f"{experiment_name}.csv", summary_row)
    logger.info("Best checkpoint: %s", best_checkpoint_rel)
    logger.info("Validation metrics saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
