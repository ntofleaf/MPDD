#!/usr/bin/env python3
"""
infer_and_pack.py
对 binary + ternary 两个 checkpoint 分别做测试集推理，
然后将 binary.csv + ternary.csv 打包成符合 CodaBench 格式的 submission.zip。

文件在 ZIP 根目录下（无子目录前缀）：
    binary.csv
    ternary.csv

用法（在项目根目录下运行）:
    python make_submission_forcodabench/infer_and_pack.py \\
        --binary_ckpt  checkpoints/Track2/A-V-P/binary/.../best_model_XXX.pth \\
        --ternary_ckpt checkpoints/Track2/A-V-P/ternary/.../best_model_YYY.pth \\
        --test_root    /home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young \\
        --personality  /home/niutao/data/datasets_MPDD/Train-MPDD-Young/descriptions_embeddings_with_ids.npy \\
        --output_dir   make_submission_forcodabench/my_submissions/run_latest
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── 项目路径 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MPDDElderDataset, collate_batch, load_task_maps
from metrics import evaluate_model
from models import TorchcatBaseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-set inference + CodaBench packaging")
    p.add_argument("--binary_ckpt",  required=True, help="binary 任务 checkpoint 路径")
    p.add_argument("--ternary_ckpt", required=True, help="ternary 任务 checkpoint 路径")
    p.add_argument("--test_root",    required=True, help="测试集数据根目录（含 split_labels_test.csv）")
    p.add_argument("--personality",  required=True, help="descriptions_embeddings_with_ids.npy 路径")
    p.add_argument("--output_dir",   required=True, help="输出目录（binary.csv / ternary.csv / submission.zip 都在这里）")
    return p.parse_args()


def run_inference(
    ckpt_path: Path,
    test_root: str,
    personality_npy: str,
) -> tuple[list[int], list[int], list[float]]:
    """对测试集做推理。
    返回 (ids, class_preds, phq9_preds)
      ids        : 测试样本 ID 列表
      class_preds: 分类预测（binary=0/1，ternary=0/1/2）
      phq9_preds : PHQ-9 预测值，已反变换到原始尺度 [0, 27]
    """
    ckpt_path = PROJECT_ROOT / ckpt_path
    print(f"\n  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    task          = ckpt["task"]
    subtrack      = ckpt["subtrack"]
    audio_feature = ckpt["audio_feature"]
    video_feature = ckpt["video_feature"]
    target_t      = int(ckpt["target_t"])
    model_kwargs  = dict(ckpt["model_kwargs"])
    reg_label     = ckpt.get("regression_label", "label2") or "label2"
    phq_log1p     = bool(ckpt.get("phq_log1p_normalized", False))

    test_split_csv = str(Path(test_root) / "split_labels_test.csv")
    task_maps = load_task_maps(test_split_csv, task, reg_label)

    dataset = MPDDElderDataset(
        data_root=test_root,
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
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_batch, num_workers=0)

    model = TorchcatBaseline(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # [Fix #3] 同时支持旧式回归头（use_regression_head=True）
    # 和新式纯回归模型（is_regression=True, use_regression_head=False）
    use_reg      = bool(model_kwargs.get("use_regression_head", False))
    is_pure_reg  = bool(model_kwargs.get("is_regression", False)) and not use_reg
    task_in_ckpt = ckpt.get("task", "")

    # 构建 criterion：纯回归模型用 CCCLoss，旧式联合模型用 (CE, MSE)，分类模型用 CE
    if is_pure_reg:
        from metrics import CCCLoss
        criterion = CCCLoss()
    elif use_reg:
        criterion = (nn.CrossEntropyLoss(), nn.MSELoss())
    else:
        criterion = nn.CrossEntropyLoss()

    metrics     = evaluate_model(model, loader, criterion, DEVICE, task_in_ckpt)

    ids         = [int(x) for x in metrics["ids"]]
    class_preds = [int(x) for x in metrics.get("class_pred", metrics["y_pred"])]

    # PHQ-9 反变换：use_reg（旧式联合）或 is_pure_reg（新式纯回归）都需要处理
    has_reg_output = (use_reg or is_pure_reg) and "phq_pred" in metrics
    if has_reg_output:
        if phq_log1p:
            phq_preds = [float(np.clip(np.expm1(x), 0, 27)) for x in metrics["phq_pred"]]
            _r = [min(phq_preds), max(phq_preds)]
            print(f"  [log1p 模式] expm1 反变换完成，PHQ-9 范围: [{_r[0]:.2f}, {_r[1]:.2f}]")
        else:
            phq_preds = [float(np.clip(x, 0, 27)) for x in metrics["phq_pred"]]
            _r = [min(phq_preds), max(phq_preds)]
            print(f"  [原始模式] 直接 clip，PHQ-9 范围: [{_r[0]:.2f}, {_r[1]:.2f}]")
    else:
        # 无回归头的纯分类模型：用类别中心值占位
        phq_center  = {0: 3.0, 1: 12.0, 2: 22.0}
        phq_preds   = [phq_center.get(c, 0.0) for c in class_preds]
        print(f"  [无回归头] 使用类别中心值占位")

    return ids, class_preds, phq_preds


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Binary 推理 ────────────────────────────────────────────────────────
    print("\n=== [1/2] Binary 推理 ===")
    b_ids, b_cls, b_phq = run_inference(args.binary_ckpt, args.test_root, args.personality)
    binary_rows = [
        {"id": sid, "binary_pred": int(cls), "phq9_pred": f"{phq:.4f}"}
        for sid, cls, phq in zip(b_ids, b_cls, b_phq)
    ]
    binary_csv = output_dir / "binary.csv"
    write_csv(binary_csv, ["id", "binary_pred", "phq9_pred"], binary_rows)

    cls_dist = {0: b_cls.count(0), 1: b_cls.count(1)}
    print(f"  预测分布: 正常(0)={cls_dist[0]}, 抑郁(1)={cls_dist[1]}")

    # ── Ternary 推理 ───────────────────────────────────────────────────────
    print("\n=== [2/2] Ternary 推理 ===")
    t_ids, t_cls, t_phq = run_inference(args.ternary_ckpt, args.test_root, args.personality)
    ternary_rows = [
        {"id": sid, "ternary_pred": int(cls), "phq9_pred": f"{phq:.4f}"}
        for sid, cls, phq in zip(t_ids, t_cls, t_phq)
    ]
    ternary_csv = output_dir / "ternary.csv"
    write_csv(ternary_csv, ["id", "ternary_pred", "phq9_pred"], ternary_rows)

    t_dist = {k: t_cls.count(k) for k in [0, 1, 2]}
    print(f"  预测分布: 轻度(0)={t_dist[0]}, 中度(1)={t_dist[1]}, 重度(2)={t_dist[2]}")

    # ── ID 一致性校验 ──────────────────────────────────────────────────────
    if sorted(b_ids) != sorted(t_ids):
        print(f"\n[WARNING] binary 和 ternary 的测试 ID 不一致，请检查 checkpoint！")
        print(f"  binary  IDs: {sorted(b_ids)}")
        print(f"  ternary IDs: {sorted(t_ids)}")
    else:
        print(f"\n✅ ID 校验通过，共 {len(b_ids)} 个测试样本")

    # ── 打包 submission.zip（根目录无子文件夹）────────────────────────────
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(binary_csv,  arcname="binary.csv")   # 根目录，无前缀
        zf.write(ternary_csv, arcname="ternary.csv")  # 根目录，无前缀
    print(f"\n✅ 提交文件已生成：{zip_path}")
    print(f"   ZIP 内容:")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            print(f"     {info.filename}  ({info.file_size} bytes)")
    print(f"\n   上传到 CodaBench 即可。")


if __name__ == "__main__":
    main()
