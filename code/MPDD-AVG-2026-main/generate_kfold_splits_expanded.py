"""
generate_kfold_splits_expanded.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
将原始训练集 + 测试集伪标签合并，生成「三池」5折交叉验证分割文件。

三池设计（val_size=9, n_folds=5 时）：
  Pool A（轮换池）: 45个 ← 从原始训练集分层抽取，参与 val 轮换
                           StratifiedKFold(5) → 9 val / 36 train 每折
  Pool B（固定训练）: 43个 ← 原始训练集其余样本，永远在 train
  Pool C（测试固定）: 22个 ← 测试集伪标签，永远在 train，特征来自 test 目录

每折结果：
  train = 36(Pool A) + 43(Pool B) + 22(Pool C) = 101
  val   = 9

输出 fold CSV 格式（相比原版新增 data_source 列）：
  ID, split, label2, label3, phq9_score, data_source
  data_source = "train"（来自原始训练集）/ "test"（来自测试集）

用法示例
--------
# Binary
python generate_kfold_splits_expanded.py \\
    --train_csv /home/niutao/data/datasets_MPDD/Train-MPDD-Young/split_labels_train.csv \\
    --test_label_csv /home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young/labels_binary.csv \\
    --task binary \\
    --val_size 9 \\
    --out_dir kfold_splits/Track2_Young_binary_expanded

# Ternary
python generate_kfold_splits_expanded.py \\
    --train_csv /home/niutao/data/datasets_MPDD/Train-MPDD-Young/split_labels_train.csv \\
    --test_label_csv /home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young/labels_3class.csv \\
    --task ternary \\
    --val_size 9 \\
    --out_dir kfold_splits/Track2_Young_ternary_expanded
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from dataset import REGRESSION_TASK, get_phq9_target, resolve_project_path
from train_val_split import _load_train_rows


# ─── 辅助：构建分层标签 ────────────────────────────────────────────────────

def build_strat_labels(rows: list[dict], task: str) -> list[int]:
    if task == "binary":
        return [int(float(r["label2"])) for r in rows]
    elif task == "ternary":
        return [int(float(r["label3"])) for r in rows]
    else:
        raise ValueError(f"task must be 'binary' or 'ternary', got: {task}")


# ─── 辅助：从测试标签 CSV 加载行（标准化列名与训练集一致）─────────────────

def load_test_label_rows(test_label_csv: str | Path) -> list[dict]:
    """
    读取测试集伪标签 CSV（labels_binary.csv 或 labels_3class.csv）。
    标准化为与训练集 CSV 相同的列名结构（缺失列填 "-1"）。
    """
    csv_path = Path(test_label_csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    if not raw_rows:
        raise ValueError(f"测试标签 CSV 为空: {csv_path}")

    # 目标列名（与训练集一致）
    target_fields = ["ID", "split", "label2", "label3", "phq9_score"]
    normalized = []
    for r in raw_rows:
        row: dict[str, str] = {}
        row["ID"]        = str(int(float(r["ID"])))
        row["split"]     = "train"   # 加入训练池时统一标记为 train
        row["label2"]    = str(int(float(r["label2"])))    if "label2"    in r else "-1"
        row["label3"]    = str(int(float(r["label3"])))    if "label3"    in r else "-1"
        row["phq9_score"]= str(float(r.get("phq9_score", r.get("PHQ-9", "0"))))
        normalized.append(row)
    return normalized


# ─── 辅助：分层抽取 n_select 个样本 ──────────────────────────────────────

def stratified_select(
    rows: list[dict],
    strat_labels: list[int],
    n_select: int,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    从 rows 中分层抽取 n_select 个，返回 (selected_idx, remaining_idx)。
    使用 StratifiedShuffleSplit 确保类别比例与原始分布一致。
    """
    n_total = len(rows)
    if n_select >= n_total:
        raise ValueError(
            f"n_select ({n_select}) >= n_total ({n_total})，"
            f"无法从 Pool 中保留固定训练样本。"
        )
    train_size = n_select / n_total
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=seed
    )
    sel_arr, rem_arr = next(sss.split(range(n_total), strat_labels))
    return sel_arr.tolist(), rem_arr.tolist()


# ─── 主函数 ───────────────────────────────────────────────────────────────

def generate_kfold_splits_expanded(
    train_csv:      str | Path,
    test_label_csv: str | Path,
    task:           str,
    out_dir:        str | Path,
    n_folds:        int = 5,
    val_size:       int = 9,
    seed:           int = 42,
) -> list[Path]:
    """
    生成扩展版 n_folds 个 fold CSV（三池设计）。

    参数
    ----
    train_csv      : 原始训练集 CSV（split=train 的行）
    test_label_csv : 测试集伪标签 CSV（labels_binary.csv 或 labels_3class.csv）
    task           : "binary" 或 "ternary"
    out_dir        : 输出目录
    n_folds        : Fold 数（默认 5）
    val_size       : 每 fold 验证集样本数（默认 9）
    seed           : 随机种子（固定，勿修改）
    """
    if task not in ("binary", "ternary"):
        raise ValueError(f"task 必须是 'binary' 或 'ternary'，got: {task}")

    # ── 1. 加载数据 ────────────────────────────────────────────────────────
    train_rows = _load_train_rows(train_csv)       # 原始训练集（88行）
    test_rows  = load_test_label_rows(test_label_csv)  # 测试集伪标签（22行）

    print(f"原始训练集样本数: {len(train_rows)}")
    print(f"测试集伪标签样本数: {len(test_rows)}")

    # ── 2. 构建分层标签（用于 Pool A 抽取 & KFold 分层）──────────────────
    strat_labels = build_strat_labels(train_rows, task)
    label_counts = Counter(strat_labels)
    print(f"训练集标签分布: {dict(sorted(label_counts.items()))}")

    # ── 3. 从原始训练集中分层抽取 Pool A（轮换池） ────────────────────────
    n_pool_a = val_size * n_folds   # 9 × 5 = 45
    print(f"\n三池划分:")
    print(f"  Pool A（轮换池）: {n_pool_a} 样本（val_size={val_size} × n_folds={n_folds}）")
    print(f"  Pool B（固定训练）: {len(train_rows) - n_pool_a} 样本")
    print(f"  Pool C（测试固定）: {len(test_rows)} 样本")

    pool_a_idx, pool_b_idx = stratified_select(
        train_rows, strat_labels, n_select=n_pool_a, seed=seed
    )
    pool_a = [train_rows[i] for i in pool_a_idx]   # 45
    pool_b = [train_rows[i] for i in pool_b_idx]   # 43

    pool_a_strat = build_strat_labels(pool_a, task)
    pool_a_label_counts = Counter(pool_a_strat)
    print(f"  Pool A 标签分布: {dict(sorted(pool_a_label_counts.items()))}")

    # ── 4. 对 Pool A 做 StratifiedKFold ───────────────────────────────────
    min_bin = min(pool_a_label_counts.values())
    if min_bin < n_folds:
        print(
            f"  [警告] Pool A 最小类别样本量 ({min_bin}) < n_folds ({n_folds})，"
            f"StratifiedKFold 可能退化。"
        )

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pool_a_ids = list(range(len(pool_a)))  # 用下标作为 split 的输入

    out_path = resolve_project_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 标准列名 + data_source 列
    fieldnames = ["ID", "split", "label2", "label3", "phq9_score", "data_source"]
    generated: list[Path] = []

    for fold_idx, (a_train_idx, a_val_idx) in enumerate(
        kf.split(pool_a_ids, pool_a_strat)
    ):
        fold_path = out_path / f"fold_{fold_idx}.csv"

        with open(fold_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Pool A val（9条）
            for i in a_val_idx:
                row = dict(pool_a[i])
                row["split"] = "val"
                row["data_source"] = "train"
                writer.writerow({k: row.get(k, "") for k in fieldnames})

            # Pool A train（36条）
            for i in a_train_idx:
                row = dict(pool_a[i])
                row["split"] = "train"
                row["data_source"] = "train"
                writer.writerow({k: row.get(k, "") for k in fieldnames})

            # Pool B（43条，固定训练）
            for row in pool_b:
                row = dict(row)
                row["split"] = "train"
                row["data_source"] = "train"
                writer.writerow({k: row.get(k, "") for k in fieldnames})

            # Pool C（22条，测试集伪标签）
            for row in test_rows:
                row = dict(row)
                row["split"] = "train"
                row["data_source"] = "test"
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        n_val   = len(a_val_idx)
        n_train = len(a_train_idx) + len(pool_b) + len(test_rows)
        print(f"  Fold {fold_idx}: train={n_train}, val={n_val}  →  {fold_path}")
        generated.append(fold_path)

    # ── 5. 写 meta.json ────────────────────────────────────────────────────
    meta = {
        "task":           task,
        "n_folds":        n_folds,
        "val_size":       val_size,
        "seed":           seed,
        "train_csv":      str(Path(train_csv).resolve()),
        "test_label_csv": str(Path(test_label_csv).resolve()),
        "total_train_original": len(train_rows),
        "total_test_added":     len(test_rows),
        "total_samples":        len(train_rows) + len(test_rows),
        "pool_a_size":    len(pool_a),
        "pool_b_size":    len(pool_b),
        "pool_c_size":    len(test_rows),
        "per_fold": {
            "train": len(pool_a) - val_size + len(pool_b) + len(test_rows),
            "val":   val_size,
        },
        "train_label_counts": dict(sorted(label_counts.items())),
        "pool_a_label_counts": dict(sorted(pool_a_label_counts.items())),
    }
    meta_path = out_path / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Meta 信息已写入: {meta_path}")

    return generated


# ─── CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成扩展版 5折 CV 分割（三池设计，val_size=9）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train_csv", required=True,
        help="原始训练集 CSV 路径（split_labels_train.csv）",
    )
    parser.add_argument(
        "--test_label_csv", required=True,
        help="测试集伪标签 CSV（labels_binary.csv 或 labels_3class.csv）",
    )
    parser.add_argument(
        "--task", required=True, choices=["binary", "ternary"],
        help="任务类型（决定分层抽样依据）",
    )
    parser.add_argument(
        "--out_dir", default="kfold_splits/expanded",
        help="输出目录（默认 kfold_splits/expanded/）",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Fold 数（默认 5）",
    )
    parser.add_argument(
        "--val_size", type=int, default=9,
        help="每 fold 验证集样本数（默认 9）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（固定，生成后勿修改）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(f"  生成扩展版五折 CV 分割文件（三池设计）")
    print(f"  train_csv      : {args.train_csv}")
    print(f"  test_label_csv : {args.test_label_csv}")
    print(f"  task           : {args.task}")
    print(f"  n_folds        : {args.n_folds}")
    print(f"  val_size       : {args.val_size}")
    print(f"  seed           : {args.seed}  (固定，勿修改)")
    print(f"  out_dir        : {args.out_dir}")
    print("=" * 60)

    generated = generate_kfold_splits_expanded(
        train_csv=args.train_csv,
        test_label_csv=args.test_label_csv,
        task=args.task,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        val_size=args.val_size,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print(f"  完成！共生成 {len(generated)} 个 fold 文件：")
    for p in generated:
        print(f"    {p}")
    print("\n  ⚠  此目录已固定，后续直接用 --fold_csv 读取，勿重新生成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
