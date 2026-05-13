#!/usr/bin/env python3
"""
ensemble_binary_test.py
对多个 binary checkpoint 做 logit 平均集成，生成 binary.csv，
然后和已有的 ternary.csv 合并打包成 submission.zip。

用法（在项目根目录下运行）:
  cd /home/niutao/data/code/MPDD-AVG-2026-main
  python make_submission_forcodabench/ensemble_binary_test.py

在脚本顶部的 BINARY_CKPTS 列表里填写要集成的 checkpoint 路径。
"""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MPDDElderDataset, collate_batch, load_task_maps
from models import TorchcatBaseline

# ═══════════════════════════════════════════════════════════════════════════
# ★ 填写所有要集成的 binary checkpoint（全部来自同一特征配置）★
# ═══════════════════════════════════════════════════════════════════════════
BINARY_CKPTS = [
    # seed=3407: val_F1=72.1%, val_Kappa=0.444 ✅ 最优
    "checkpoints/Track2/A-V-P/binary/track2_binary_A-V+P_bilstm_mean_mfcc__densenet/best_model_2026-05-10-15.12.19.pth",
    # seed=42: val_F1=33.8%, val_Kappa=-0.222 ❌ 差，不使用
    # "checkpoints/Track2/A-V-P/binary/track2_binary_A-V+P_bilstm_mean_mfcc__densenet/best_model_2026-05-10-15.20.15.pth",
    # seed=2024（待训练后视情况加入）
    # "checkpoints/Track2/A-V-P/binary/track2_binary_A-V+P_bilstm_mean_mfcc__densenet/best_model_XXXX.pth",
]

# 测试集路径
TEST_DATA_ROOT  = "/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young"
TEST_SPLIT_CSV  = f"{TEST_DATA_ROOT}/split_labels_test.csv"
PERSONALITY_NPY = "/home/niutao/data/datasets_MPDD/Train-MPDD-Young/descriptions_embeddings_with_ids.npy"

# 已有的 ternary.csv（上次生成的）
EXISTING_TERNARY_CSV = PROJECT_ROOT / "make_submission_forcodabench/my_submission/ternary.csv"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "make_submission_forcodabench/my_submission_ensemble"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ═══════════════════════════════════════════════════════════════════════════


def get_logits_and_phq(ckpt_path: str | Path) -> tuple[list[int], np.ndarray, list[float]]:
    """返回 (ids, logits_np [N,C], phq_preds_raw)
    phq_preds_raw: 已反变换到原始 PHQ-9 尺度 [0, 27]
    """
    ckpt_path = PROJECT_ROOT / ckpt_path
    print(f"  Loading: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    task          = ckpt["task"]
    subtrack      = ckpt["subtrack"]
    audio_feature = ckpt["audio_feature"]
    video_feature = ckpt["video_feature"]
    target_t      = int(ckpt["target_t"])
    model_kwargs  = dict(ckpt["model_kwargs"])
    reg_label     = ckpt.get("regression_label", "label2") or "label2"
    # 读取是否使用 log1p 归一化
    # 旧 checkpoint 无此字段，默认 False（直接输出原始 PHQ-9）
    phq_log1p = bool(ckpt.get("phq_log1p_normalized", False))

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
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_batch, num_workers=0)

    model = TorchcatBaseline(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    use_reg = bool(model_kwargs.get("use_regression_head", False))

    all_ids, all_logits, all_phq = [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                audio=batch["audio"].to(DEVICE) if "audio" in batch else None,
                video=batch["video"].to(DEVICE) if "video" in batch else None,
                gait=batch["gait"].to(DEVICE) if "gait" in batch else None,
                personality=batch["personality"].to(DEVICE),
                pair_mask=batch["pair_mask"].to(DEVICE) if "pair_mask" in batch else None,
            )
            if use_reg:
                logits, reg_out = outputs
                phq_raw = reg_out.cpu().numpy()        # log1p 或原始，取决于训练配置
                if phq_log1p:
                    # 新 checkpoint：输出是 log1p 空间，需要 expm1() 反变换
                    # expm1(x) = exp(x) - 1，与 log1p(x) = log(1+x) 互为逆运算
                    # 例：模型输出 1.792 → expm1(1.792) ≈ 5.0 → 原始 PHQ-9=5
                    phq_batch = np.expm1(phq_raw).tolist()
                else:
                    # 旧 checkpoint：输出已在原始 PHQ-9 尺度，直接使用
                    phq_batch = phq_raw.tolist()
            else:
                logits = outputs
                # 类别中心作为 PHQ-9 占位
                preds = logits.argmax(dim=-1).cpu().numpy()
                phq_center = {0: 3.0, 1: 12.0, 2: 22.0}
                phq_batch = [phq_center.get(int(p), 0.0) for p in preds]

            probs = F.softmax(logits, dim=-1)
            all_ids.extend(batch["pid"].cpu().numpy().tolist())
            all_logits.append(probs.cpu().numpy())
            all_phq.extend(phq_batch)

    return (
        [int(x) for x in all_ids],
        np.concatenate(all_logits, axis=0),   # [N, num_classes]
        # clip 到 [0, 27]，确保 PHQ-9 在合法范围内
        [float(np.clip(x, 0, 27)) for x in all_phq],
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}  ({len(rows)} rows)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Ensemble binary inference ({len(BINARY_CKPTS)} checkpoints) ===")
    ids_ref, acc_logits, acc_phq = None, None, None

    for ckpt_path in BINARY_CKPTS:
        ids, logits, phq = get_logits_and_phq(ckpt_path)
        if ids_ref is None:
            ids_ref = ids
            acc_logits = logits
            acc_phq = np.array(phq)
        else:
            assert ids == ids_ref, "ID 顺序不一致，请检查 checkpoint 是否使用同一测试集"
            acc_logits = acc_logits + logits
            acc_phq = acc_phq + np.array(phq)

    n = len(BINARY_CKPTS)
    avg_logits = acc_logits / n          # 平均 softmax 概率
    avg_phq    = acc_phq / n             # 平均 PHQ-9

    binary_preds = avg_logits.argmax(axis=-1).tolist()
    phq_preds    = [float(np.clip(x, 0, 27)) for x in avg_phq]

    # 写 binary.csv
    binary_rows = [
        {"id": sid, "binary_pred": int(cls), "phq9_pred": f"{phq:.4f}"}
        for sid, cls, phq in zip(ids_ref, binary_preds, phq_preds)
    ]
    binary_csv = OUTPUT_DIR / "binary.csv"
    write_csv(binary_csv, ["id", "binary_pred", "phq9_pred"], binary_rows)

    print(f"\n预测分布: 正常(0)={binary_preds.count(0)}, 抑郁(1)={binary_preds.count(1)}")

    # 复制现有 ternary.csv
    import shutil
    ternary_csv = OUTPUT_DIR / "ternary.csv"
    shutil.copy(EXISTING_TERNARY_CSV, ternary_csv)
    print(f"Copied ternary.csv from: {EXISTING_TERNARY_CSV}")

    # 打包
    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(binary_csv,  arcname="binary.csv")
        zf.write(ternary_csv, arcname="ternary.csv")

    print(f"\n✅ 集成提交文件已生成：{zip_path}")
    print(f"   ({n} 个 checkpoint 的平均预测)")


if __name__ == "__main__":
    main()
