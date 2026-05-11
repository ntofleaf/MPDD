#!/usr/bin/env python3
"""
generate_submission.py
从 binary 和 ternary 最佳 checkpoint 对测试集做推理，
生成符合 CodaBench 格式的 submission.zip。

用法:
  cd /home/niutao/data/code/MPDD-AVG-2026-main
  python make_submission_forcodabench/generate_submission.py
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── 项目路径 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MPDDElderDataset, collate_batch, load_task_maps, resolve_project_path
from metrics import evaluate_model
from models import TorchcatBaseline

# ═══════════════════════════════════════════════════════════════════════════
# ★ 在这里修改两个最佳 checkpoint 的路径 ★
# ═══════════════════════════════════════════════════════════════════════════
BINARY_CKPT = PROJECT_ROOT / (
    "checkpoints/Track2/A-V-P/binary/"
    "track2_binary_A-V+P_bilstm_mean_mfcc__densenet/"
    "best_model_2026-05-10-15.12.19.pth"          # mfcc+densenet seed=3407  val_F1=72.1%
)
TERNARY_CKPT = PROJECT_ROOT / (
    "checkpoints/Track2/A-V-P/ternary/"
    "track2_ternary_A-V+P_bilstm_mean_mfcc__densenet/"
    "best_model_2026-05-09-23.12.49.pth"          # mfcc+densenet seed=3407  val_F1=40.4%
)

# 测试集路径
TEST_DATA_ROOT   = "/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"
TEST_SPLIT_CSV   = f"{TEST_DATA_ROOT}/split_labels_test.csv"
PERSONALITY_NPY  = "/home/niutao/data/datasets_MPDD/Train-MPDD-Young/descriptions_embeddings_with_ids.npy"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "make_submission_forcodabench" / "my_submission_ensemble"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ═══════════════════════════════════════════════════════════════════════════


def load_ckpt(ckpt_path: Path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt


def run_inference(ckpt_path: Path) -> tuple[list[int], list[int], list[float]]:
    """返回 (ids, class_preds, phq9_preds)"""
    ckpt = load_ckpt(ckpt_path)

    task          = ckpt["task"]
    subtrack      = ckpt["subtrack"]
    audio_feature = ckpt["audio_feature"]
    video_feature = ckpt["video_feature"]
    target_t      = int(ckpt["target_t"])
    model_kwargs  = dict(ckpt["model_kwargs"])
    reg_label     = ckpt.get("regression_label", "label2") or "label2"

    task_maps = load_task_maps(TEST_SPLIT_CSV, task, reg_label)
    dataset = MPDDElderDataset(
        data_root=TEST_DATA_ROOT,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=PERSONALITY_NPY,
        phq_map=task_maps.get("test_phq_map"),
        target_t=target_t,
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=collate_batch, num_workers=0
    )

    model = TorchcatBaseline(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    use_reg = bool(model_kwargs.get("use_regression_head", False))
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss()) if use_reg else nn.CrossEntropyLoss()
    metrics = evaluate_model(model, loader, criterion, DEVICE, task)

    ids         = [int(x) for x in metrics["ids"]]
    class_preds = [int(x) for x in metrics.get("class_pred", metrics["y_pred"])]
    # phq9 预测：回归头输出，clamp 到 [0, 27]
    if use_reg and "phq_pred" in metrics:
        phq_preds = [float(np.clip(x, 0, 27)) for x in metrics["phq_pred"]]
    else:
        # 没有回归头时，用类别中心值做占位 (0→3, 1→12, 2→22)
        phq_center = {0: 3.0, 1: 12.0, 2: 22.0}
        phq_preds  = [phq_center.get(c, 0.0) for c in class_preds]

    return ids, class_preds, phq_preds


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}  ({len(rows)} rows)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Binary ──────────────────────────────────────────────────────────
    print("\n=== Binary inference ===")
    b_ids, b_cls, b_phq = run_inference(BINARY_CKPT)
    binary_rows = [
        {"id": sid, "binary_pred": cls, "phq9_pred": f"{phq:.4f}"}
        for sid, cls, phq in zip(b_ids, b_cls, b_phq)
    ]
    binary_csv = OUTPUT_DIR / "binary.csv"
    write_csv(binary_csv, ["id", "binary_pred", "phq9_pred"], binary_rows)

    # ── Ternary ─────────────────────────────────────────────────────────
    print("\n=== Ternary inference ===")
    t_ids, t_cls, t_phq = run_inference(TERNARY_CKPT)
    ternary_rows = [
        {"id": sid, "ternary_pred": cls, "phq9_pred": f"{phq:.4f}"}
        for sid, cls, phq in zip(t_ids, t_cls, t_phq)
    ]
    ternary_csv = OUTPUT_DIR / "ternary.csv"
    write_csv(ternary_csv, ["id", "ternary_pred", "phq9_pred"], ternary_rows)

    # ── 验证 ID 一致性 ──────────────────────────────────────────────────
    assert sorted(b_ids) == sorted(t_ids), \
        f"binary 和 ternary 的测试 ID 不一致！\nbinary: {sorted(b_ids)}\nternary: {sorted(t_ids)}"
    print(f"\n✅ ID 校验通过，共 {len(b_ids)} 个测试样本")

    # ── 打包 submission.zip ─────────────────────────────────────────────
    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(binary_csv,  arcname="binary.csv")
        zf.write(ternary_csv, arcname="ternary.csv")
    print(f"\n✅ 提交文件已生成：{zip_path}")
    print(f"   请将此文件上传到 CodaBench。")


if __name__ == "__main__":
    main()
