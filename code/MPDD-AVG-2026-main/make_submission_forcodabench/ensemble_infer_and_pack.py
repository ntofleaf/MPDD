#!/usr/bin/env python3
"""
ensemble_infer_and_pack.py
──────────────────────────────────────────────────────────────────────────────
用途
────
对三类任务（binary / ternary / regression）各自的多个 checkpoint 做集成推理，
将结果拼接成符合 CodaBench 格式的 submission.zip。

集成策略
────────
• 分类（binary / ternary）：对每个 checkpoint 的 softmax 概率做算术平均，
  再对平均概率取 argmax 得到最终类别。
• 回归（PHQ-9）：直接对每个 regression checkpoint 输出的 PHQ-9 值做算术平均。

提交文件内容
────────────
  binary.csv   = binary_pred（来自 binary 集成）  + phq9_pred（来自 regression 集成）
  ternary.csv  = ternary_pred（来自 ternary 集成）+ phq9_pred（来自 regression 集成）

用法（在项目根目录下运行）
────────────────────────
  python make_submission_forcodabench/ensemble_infer_and_pack.py \
      --binary_ckpts   ckpt1.pth ckpt2.pth ckpt3.pth \
      --ternary_ckpts  ckpt4.pth ckpt5.pth ckpt6.pth \
      --reg_ckpts      ckpt7.pth ckpt8.pth ckpt9.pth \
      --test_root      /data/Test-MPDD-Young/Young         \
      --personality    /data/Train-MPDD-Young/descriptions_embeddings_with_ids.npy \
      --output_dir     make_submission_forcodabench/my_submissions/run_latest
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─── 将项目根目录加入搜索路径，使 dataset / models 等模块可被导入 ───────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MPDDElderDataset, collate_batch, load_task_maps
from models import TorchcatBaseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  命令行参数解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多 checkpoint 集成推理，输出 CodaBench 格式的 submission.zip"
    )
    # ── 分类 checkpoints ──
    p.add_argument("--binary_ckpts",  nargs="+", required=True,
                   help="binary 任务的 checkpoint 路径列表（空格分隔，≥1 个）")
    p.add_argument("--ternary_ckpts", nargs="+", required=True,
                   help="ternary 任务的 checkpoint 路径列表（空格分隔，≥1 个）")
    # ── 回归 checkpoints ──
    p.add_argument("--reg_ckpts",     nargs="+", required=True,
                   help="regression 任务的 checkpoint 路径列表（空格分隔，≥1 个）")
    # ── 数据路径 ──
    p.add_argument("--test_root",    required=True,
                   help="测试集根目录（含 split_labels_test.csv）")
    p.add_argument("--personality",  required=True,
                   help="descriptions_embeddings_with_ids.npy 路径")
    # ── 输出 ──
    p.add_argument("--output_dir",   required=True,
                   help="输出目录，binary.csv / ternary.csv / submission.zip 均保存到此处")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  单个 checkpoint 的前向推理
# ═══════════════════════════════════════════════════════════════════════════════

def infer_one_ckpt(
    ckpt_path: str | Path,
    test_root: str,
    personality_npy: str,
) -> tuple[list[int], np.ndarray, list[float]]:
    """
    对一个 checkpoint 做完整的测试集推理。

    返回
    ────
    ids        : 测试样本 ID 列表（与 split_labels_test.csv 中的 ID 对应）
    probs_np   : softmax 概率矩阵，shape=[N, num_classes]，float32
    phq9_preds : PHQ-9 预测值列表（已从 log1p 空间还原到原始 0~27 尺度）
                 若该 checkpoint 没有回归头，则全填 0.0（占位）
    """
    ckpt_path = PROJECT_ROOT / ckpt_path
    print(f"    → Loading: {ckpt_path.name}")
    # weights_only=False 是因为 checkpoint 里存有自定义对象（如 OrderedDict）
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── 从 checkpoint 中读取训练时的配置 ─────────────────────────────────────
    task          = ckpt["task"]            # binary / ternary / regression
    subtrack      = ckpt["subtrack"]        # A-V+P / A-V-G+P / G+P
    audio_feature = ckpt["audio_feature"]
    video_feature = ckpt["video_feature"]
    target_t      = int(ckpt["target_t"])
    model_kwargs  = dict(ckpt["model_kwargs"])
    reg_label     = ckpt.get("regression_label", "label2") or "label2"
    # 是否在训练时对 PHQ-9 做了 log1p 归一化
    # 若为 True，推理时需 expm1() 还原；若为 False（旧 checkpoint），直接使用
    phq_log1p = bool(ckpt.get("phq_log1p_normalized", False))

    # ── 构建测试集 Dataset / DataLoader ──────────────────────────────────────
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
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
    )

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = TorchcatBaseline(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    use_reg = bool(model_kwargs.get("use_regression_head", False))

    # ── 前向推理，收集 softmax 概率 和 PHQ-9 预测 ────────────────────────────
    all_ids: list[int] = []
    all_probs: list[np.ndarray] = []   # 每个 batch 的 softmax 概率
    all_phq: list[float] = []          # 每个样本的 PHQ-9 预测

    # 纯回归模式标记：is_regression=True 时 classifier 直接输出 1 个标量
    #                 use_regression_head=False 时无辅助回归头（新模式）
    is_pure_reg_model = bool(model_kwargs.get("is_regression", False)) and not use_reg

    with torch.no_grad():
        for batch in loader:
            # 前向传播（根据 subtrack 决定是否传入各模态）
            outputs = model(
                audio=batch["audio"].to(DEVICE)       if "audio"    in batch else None,
                video=batch["video"].to(DEVICE)       if "video"    in batch else None,
                gait=batch["gait"].to(DEVICE)         if "gait"     in batch else None,
                personality=batch["personality"].to(DEVICE),
                pair_mask=batch["pair_mask"].to(DEVICE) if "pair_mask" in batch else None,
            )

            # ── 分离分类 logits 和回归输出 ────────────────────────────────
            if is_pure_reg_model:
                # 纯回归模式（新）：outputs 是 [batch_size, 1] 或 [batch_size] 的标量
                # classifier(is_regression=True) 输出 Linear(hidden, 1)
                reg_np = outputs.squeeze(-1).cpu().float().numpy().flatten()  # [batch_size]
                if phq_log1p:
                    phq_batch = [float(np.clip(np.expm1(v), 0, 27)) for v in reg_np]
                else:
                    phq_batch = [float(np.clip(v, 0, 27)) for v in reg_np]
                # 纯回归模型没有分类 logits，用均匀分布占位（不参与分类集成）
                num_cls = model_kwargs.get("num_classes", 2)
                logits = torch.zeros(len(reg_np), num_cls)
            elif use_reg:
                # 联合模式（旧）：模型输出 (logits, reg_out) 元组
                logits, reg_out = outputs
                reg_np = reg_out.cpu().numpy().flatten()   # [batch_size]
                if phq_log1p:
                    # expm1 是 log1p 的逆变换：log1p(x)=log(1+x)，expm1(y)=e^y-1
                    # 训练时目标 = log1p(PHQ-9)，推理时输出 = 模型预测的 log1p 值
                    # 还原到原始 PHQ-9：expm1(模型输出)，再 clip 到合法范围 [0, 27]
                    phq_batch = [float(np.clip(np.expm1(v), 0, 27)) for v in reg_np]
                else:
                    # 旧 checkpoint（训练时未做 log1p），直接 clip
                    phq_batch = [float(np.clip(v, 0, 27)) for v in reg_np]
            else:
                # 无回归头的分类模型：PHQ-9 用 0.0 占位，后续会被 regression 集成覆盖
                logits = outputs
                phq_batch = [0.0] * logits.shape[0]

            # ── 计算 softmax 概率 ──────────────────────────────────────────
            # 注意：这里在 softmax 之前做集成（概率平均）比在 logit 层面平均更稳定
            probs = F.softmax(logits, dim=-1).cpu().numpy()  # [batch_size, num_classes]

            all_ids.extend(batch["pid"].cpu().numpy().tolist())
            all_probs.append(probs)
            all_phq.extend(phq_batch)

    # ── 拼接所有 batch 的结果 ─────────────────────────────────────────────────
    probs_np = np.concatenate(all_probs, axis=0)   # [N, num_classes]
    ids = [int(x) for x in all_ids]
    return ids, probs_np, all_phq


# ═══════════════════════════════════════════════════════════════════════════════
#  多 checkpoint 集成：概率平均
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_cls(
    ckpt_paths: list[str | Path],
    test_root: str,
    personality_npy: str,
    task_label: str,
) -> tuple[list[int], list[int]]:
    """
    对 binary 或 ternary 的多个 checkpoint 做集成分类推理。

    集成策略：softmax 概率算术平均 → argmax

    参数
    ────
    ckpt_paths   : 多个同任务 checkpoint 的路径列表
    task_label   : 仅用于日志打印（"binary" / "ternary"）

    返回
    ────
    ids    : 测试样本 ID 列表
    preds  : 集成后的分类预测（int 列表）
    """
    print(f"\n  [集成分类 - {task_label}] {len(ckpt_paths)} 个 checkpoint")
    ids_ref: list[int] | None = None
    acc_probs: np.ndarray | None = None

    for ckpt_path in ckpt_paths:
        ids, probs, _ = infer_one_ckpt(ckpt_path, test_root, personality_npy)

        if ids_ref is None:
            # 第一个 checkpoint：初始化累积变量
            ids_ref = ids
            acc_probs = probs.copy()
        else:
            # 后续 checkpoint：累加概率（最后除以 N 得到平均值）
            assert ids == ids_ref, (
                f"ID 顺序不一致！请确认所有 {task_label} checkpoint 使用同一测试集。"
            )
            acc_probs = acc_probs + probs

    # 平均概率
    avg_probs = acc_probs / len(ckpt_paths)   # [N, num_classes]
    preds = avg_probs.argmax(axis=-1).tolist()  # [N]，取概率最大的类别

    # 打印类别分布，方便调试
    unique, counts = np.unique(preds, return_counts=True)
    dist_str = ", ".join(f"class={k}: {v}" for k, v in zip(unique, counts))
    print(f"    集成预测分布: {dist_str}")

    return ids_ref, [int(p) for p in preds]


def ensemble_reg(
    ckpt_paths: list[str | Path],
    test_root: str,
    personality_npy: str,
    min_ccc_threshold: float = 0.05,
) -> tuple[list[int], list[float]]:
    """
    对 regression 的多个 checkpoint 做集成回归推理。

    集成策略：
      1. 先读取每个 checkpoint 保存的 val phq9_ccc 指标
      2. 过滤掉退化 checkpoint（phq9_ccc < min_ccc_threshold）
      3. 对过滤后的 checkpoint 做 PHQ-9 预测值算术平均

    为什么需要过滤：
      若某个 seed 的训练完全退化（val CCC ≈ 0），该 checkpoint 预测值
      会集中在训练集均值附近（std ≈ 0），直接平均会拉低整体预测方差，
      导致最终提交值过度压缩。

    参数
    ────
    min_ccc_threshold : 过滤阈值，val phq9_ccc 低于此值的 checkpoint 被丢弃（默认 0.05）
    """
    print(f"\n  [集成回归 - PHQ-9] {len(ckpt_paths)} 个 checkpoint，过滤阈值 CCC>{min_ccc_threshold}")

    # ── Step 1: 按 val CCC 过滤，只保留有效 checkpoint ──────────────────────
    good_ckpts: list[str | Path] = []
    for ckpt_path in ckpt_paths:
        p = PROJECT_ROOT / ckpt_path
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m = ckpt.get("best_val_metrics", {})
        # 优先用 phq9_ccc（原始 PHQ-9 空间），没有则退用 ccc（log1p 空间）
        val_ccc = float(m.get("phq9_ccc", m.get("ccc", 0.0)))
        pred_std = float(m.get("phq9_pred_std", 0.0))
        status = "OK" if val_ccc >= min_ccc_threshold else "SKIP"
        print(f"    [{status}] {p.name}  val_ccc={val_ccc:.4f}  pred_std={pred_std:.2f}")
        if val_ccc >= min_ccc_threshold:
            good_ckpts.append(ckpt_path)

    if not good_ckpts:
        # 所有 checkpoint 都退化时，退回使用全部（避免空集成崩溃）
        print(f"    [警告] 所有 checkpoint 的 CCC 均低于阈值，退回使用全部 {len(ckpt_paths)} 个")
        good_ckpts = list(ckpt_paths)
    else:
        print(f"    → 实际集成 {len(good_ckpts)}/{len(ckpt_paths)} 个 checkpoint")

    # ── Step 2: 对过滤后的 checkpoint 做推理 + 平均 ─────────────────────────
    ids_ref: list[int] | None = None
    acc_phq: np.ndarray | None = None

    for ckpt_path in good_ckpts:
        ids, _, phq = infer_one_ckpt(ckpt_path, test_root, personality_npy)

        if ids_ref is None:
            ids_ref = ids
            acc_phq = np.array(phq, dtype=np.float32)
        else:
            assert ids == ids_ref, (
                "ID 顺序不一致！请确认所有 regression checkpoint 使用同一测试集。"
            )
            acc_phq = acc_phq + np.array(phq, dtype=np.float32)

    avg_phq = acc_phq / len(good_ckpts)
    # 再次 clip，确保集成后的平均值也在合法范围内
    phq_preds = [float(np.clip(v, 0, 27)) for v in avg_phq]

    print(f"    PHQ-9 范围: [{min(phq_preds):.2f}, {max(phq_preds):.2f}]  "
          f"均值: {np.mean(phq_preds):.2f}  std: {np.std(phq_preds):.2f}")
    return ids_ref, phq_preds


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV 写入 & ZIP 打包
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """将一个字典列表写入 CSV 文件（UTF-8，LF 换行）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} 行)")


def pack_zip(zip_path: Path, csv_files: dict[str, Path]) -> None:
    """
    将多个 CSV 文件打包成 ZIP。

    关键：arcname 只用文件名（无目录前缀），确保 CodaBench 评分脚本能直接找到文件。
    错误的打包方式（使用 zip -r submission/ 目录）会产生带前缀的路径，导致评分失败。
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src_path in csv_files.items():
            zf.write(src_path, arcname=arcname)   # arcname 不含目录，直接放根目录

    print(f"\n✅ 提交文件已生成：{zip_path}")
    print("   ZIP 内部结构（必须无子目录）：")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            print(f"     {info.filename}  ({info.file_size} bytes)")


# ═══════════════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  集成推理配置")
    print(f"  binary  checkpoints : {len(args.binary_ckpts)} 个")
    print(f"  ternary checkpoints : {len(args.ternary_ckpts)} 个")
    print(f"  reg     checkpoints : {len(args.reg_ckpts)} 个")
    print(f"  测试集根目录        : {args.test_root}")
    print(f"  输出目录            : {output_dir}")
    print("=" * 60)

    # ── Step 1: 集成 binary 分类 ─────────────────────────────────────────────
    print("\n>>> Step 1/3: Binary 集成分类推理")
    b_ids, b_cls = ensemble_cls(
        args.binary_ckpts, args.test_root, args.personality, "binary"
    )

    # ── Step 2: 集成 ternary 分类 ────────────────────────────────────────────
    print("\n>>> Step 2/3: Ternary 集成分类推理")
    t_ids, t_cls = ensemble_cls(
        args.ternary_ckpts, args.test_root, args.personality, "ternary"
    )

    # ── Step 3: 集成 regression（PHQ-9）─────────────────────────────────────
    print("\n>>> Step 3/3: Regression 集成回归推理（PHQ-9）")
    r_ids, r_phq = ensemble_reg(
        args.reg_ckpts, args.test_root, args.personality
    )

    # ── ID 一致性校验 ─────────────────────────────────────────────────────────
    # binary / ternary / regression 三套模型应该对同一批测试样本做推理
    assert sorted(b_ids) == sorted(t_ids) == sorted(r_ids), (
        "binary / ternary / regression 的测试集 ID 不一致！"
        f"\nbinary IDs: {sorted(b_ids)}"
        f"\nternary IDs: {sorted(t_ids)}"
        f"\nregression IDs: {sorted(r_ids)}"
    )
    print(f"\n✅ ID 校验通过，共 {len(b_ids)} 个测试样本")

    # ── 构建 PHQ-9 查找表（以 ID 为 key，方便按顺序填入两个 CSV）────────────
    # regression 预测的顺序可能与 binary 顺序不同（视 Dataset 迭代顺序而定）
    phq_map = {sid: phq for sid, phq in zip(r_ids, r_phq)}

    # ── 写 binary.csv ─────────────────────────────────────────────────────────
    binary_rows = [
        {
            "id":          sid,
            "binary_pred": int(cls),
            "phq9_pred":   f"{phq_map[sid]:.4f}",   # PHQ-9 来自 regression 集成
        }
        for sid, cls in zip(b_ids, b_cls)
    ]
    binary_csv = output_dir / "binary.csv"
    write_csv(binary_csv, ["id", "binary_pred", "phq9_pred"], binary_rows)

    # ── 写 ternary.csv ────────────────────────────────────────────────────────
    ternary_rows = [
        {
            "id":           sid,
            "ternary_pred": int(cls),
            "phq9_pred":    f"{phq_map[sid]:.4f}",  # PHQ-9 来自 regression 集成
        }
        for sid, cls in zip(t_ids, t_cls)
    ]
    ternary_csv = output_dir / "ternary.csv"
    write_csv(ternary_csv, ["id", "ternary_pred", "phq9_pred"], ternary_rows)

    # ── 打包 submission.zip ───────────────────────────────────────────────────
    zip_path = output_dir / "submission.zip"
    pack_zip(zip_path, {"binary.csv": binary_csv, "ternary.csv": ternary_csv})

    print(f"\n   上传此文件到 CodaBench：")
    print(f"   → {zip_path}")


if __name__ == "__main__":
    main()
