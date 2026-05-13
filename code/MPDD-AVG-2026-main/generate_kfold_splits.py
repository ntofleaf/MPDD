"""
generate_kfold_splits.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【只运行一次】将训练集按 StratifiedKFold 预先划分为 N 个固定 fold CSV，
写入 --out_dir 目录（默认 kfold_splits/），每个文件命名 fold_0.csv ~ fold_{N-1}.csv。

文件格式与原始 split_labels_train.csv 完全一致，仅 "split" 列的值
被替换为 "train"（训练折）或 "val"（验证折）。

【重要】生成后请勿修改或重新生成，否则 fold 分割发生变化，
已训练好的 checkpoint 与 fold 对应关系会错乱。

用法示例
--------
# Elder Track1 — binary
python generate_kfold_splits.py \\
    --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv \\
    --task binary \\
    --out_dir kfold_splits/Track1_Elder_binary

# Young Track2 — binary
python generate_kfold_splits.py \\
    --split_csv /home/niutao/data/datasets_MPDD/Train-MPDD-Young/split_labels_train.csv \\
    --task binary \\
    --out_dir kfold_splits/Track2_Young_binary

# Young Track2 — regression
python generate_kfold_splits.py \\
    --split_csv /home/niutao/data/datasets_MPDD/Train-MPDD-Young/split_labels_train.csv \\
    --task regression \\
    --out_dir kfold_splits/Track2_Young_regression
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dataset import REGRESSION_TASK, get_phq9_target, resolve_project_path
from train_val_split import _load_train_rows


def build_strat_labels(
    rows: list[dict[str, str]],
    task: str,
    n_folds: int,
) -> list[int]:
    """
    构建用于分层抽样的标签列表。

    - 分类任务（binary / ternary）：直接使用 label2 / label3 列。
    - 回归任务（regression）：
        把 PHQ-9 分数分成 n_folds 个等频箱（quantile-based），
        保证每个箱在每个 fold 中都有代表，避免 val 集 PHQ-9 分布极端偏斜。
    """
    if task == REGRESSION_TASK:
        phq_vals = np.array([get_phq9_target(r) for r in rows], dtype=np.float64)
        # 等频分箱：使用 n_folds 个分位数作为箱边界
        quantiles = np.quantile(phq_vals, np.linspace(0.0, 1.0, n_folds + 1)[1:-1])
        strat = [int(np.searchsorted(quantiles, v)) for v in phq_vals]
    elif task == "binary":
        strat = [int(float(r["label2"])) for r in rows]
    elif task == "ternary":
        strat = [int(float(r["label3"])) for r in rows]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return strat


def generate_kfold_splits(
    split_csv: str | Path,
    task: str,
    out_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
) -> list[Path]:
    """
    生成 n_folds 个 fold CSV 文件，返回文件路径列表。

    参数
    ----
    split_csv : 原始训练集 CSV（含 split=train/val 的行）
    task      : "binary" / "ternary" / "regression"
    out_dir   : 输出目录
    n_folds   : Fold 数（默认 5）
    seed      : 固定随机种子（确保可复现，勿修改）
    """
    rows = _load_train_rows(split_csv)
    ids = [int(r["ID"]) for r in rows]
    strat = build_strat_labels(rows, task, n_folds)

    # 检查分层标签分布
    bin_counts = Counter(strat)
    print(f"分层标签分布: {dict(sorted(bin_counts.items()))}")
    min_bin = min(bin_counts.values())
    if min_bin < n_folds:
        print(
            f"[警告] 最小分箱样本量 ({min_bin}) < n_folds ({n_folds})，"
            f"StratifiedKFold 可能退化为普通 KFold。"
        )

    out_path = resolve_project_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fieldnames = list(rows[0].keys())
    generated: list[Path] = []

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(ids, strat)):
        train_set = {ids[i] for i in train_indices}
        val_set   = {ids[i] for i in val_indices}
        fold_path = out_path / f"fold_{fold_idx}.csv"

        with open(fold_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                pid = int(row["ID"])
                new_row = dict(row)
                new_row["split"] = "train" if pid in train_set else "val"
                writer.writerow(new_row)

        n_train = len(train_set)
        n_val   = len(val_set)
        print(f"  Fold {fold_idx}: train={n_train}, val={n_val}  →  {fold_path}")
        generated.append(fold_path)

    # 写一个 meta.json，记录生成参数，方便溯源
    meta = {
        "task":      task,
        "n_folds":   n_folds,
        "seed":      seed,
        "split_csv": str(resolve_project_path(split_csv)),
        "total_samples": len(rows),
        "strat_bin_counts": dict(sorted(bin_counts.items())),
    }
    meta_path = out_path / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Meta 信息已写入: {meta_path}")

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一次性生成 N 个固定 fold CSV（StratifiedKFold）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split_csv", required=True,
        help="原始 split_labels_train.csv 的路径（绝对路径或相对于项目根目录）",
    )
    parser.add_argument(
        "--task", required=True,
        choices=["binary", "ternary", REGRESSION_TASK],
        help="任务类型，决定分层依据",
    )
    parser.add_argument(
        "--out_dir", default="kfold_splits",
        help="输出目录（默认 kfold_splits/）",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Fold 数（默认 5）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，生成后勿修改（默认 42）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(f"  生成五折 CV 分割文件")
    print(f"  split_csv : {args.split_csv}")
    print(f"  task      : {args.task}")
    print(f"  n_folds   : {args.n_folds}")
    print(f"  seed      : {args.seed}  (固定，勿修改)")
    print(f"  out_dir   : {args.out_dir}")
    print("=" * 60)

    generated = generate_kfold_splits(
        split_csv=args.split_csv,
        task=args.task,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
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
