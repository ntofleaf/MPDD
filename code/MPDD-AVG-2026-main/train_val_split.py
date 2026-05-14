from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from dataset import REGRESSION_TASK, get_phq9_target, get_task_label, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parent
POOL_SPLITS = {"", "train", "val"}


def _load_train_rows(split_csv: str | Path) -> list[dict[str, str]]:
    csv_path = resolve_project_path(split_csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Split CSV is empty: {csv_path}")

    train_rows = [row for row in rows if row.get("split", "train").strip().lower() in POOL_SPLITS]
    if not train_rows:
        raise ValueError(f"No train rows found in split CSV: {csv_path}")
    return train_rows


def create_train_val_split(
    split_csv: str | Path,
    task: str,
    val_ratio: float = 0.1,
    regression_label: str = "label2",
    seed: int | None = None,
) -> dict[str, Any]:
    rows = _load_train_rows(split_csv)
    sample_ids = [int(row["ID"]) for row in rows]
    sample_labels = [get_task_label(row, task, regression_label) for row in rows]

    if len(sample_ids) < 2:
        raise ValueError("At least two train samples are required to create a train/val split.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    # ── 分层策略：回归任务按 PHQ-9 分箱，分类任务按类别标签 ──────────────────
    # 【修复原因】
    # 回归任务的 sample_labels 是 PHQ-9 分数（float），若按类别计数（Counter）
    # 判断能否分层，大多数 PHQ-9 值出现 1~2 次，落入"不可分层"分支，退用纯随机
    # ShuffleSplit。18 个 val 样本的 PHQ-9 分布完全随机，可能与训练集分布差异
    # 很大，导致某些 seed 下 val CCC 为负。
    #
    # 【修复方案】
    # 对回归任务单独处理：把 PHQ-9 分成 N_REG_BINS 个等频箱（每箱样本量均等），
    # 再按箱标签做 StratifiedShuffleSplit。这确保 val 集覆盖 PHQ-9 的全部范围，
    # 不会出现 val 集全是低分或全是高分的极端情况。
    N_REG_BINS = 4   # PHQ-9 分 4 箱，每箱约 22 个样本
    if task == REGRESSION_TASK:
        # 回归任务：按真实 PHQ-9 分数分箱做分层抽样
        # 注意：sample_labels 对回归任务是 label2(0/1)，不是 PHQ-9！
        # 必须单独读取 PHQ-9 原始分数来做分箱。
        import numpy as _np
        from dataset import get_phq9_target as _get_phq9
        phq_vals = _np.array([_get_phq9(row) for row in rows])
        try:
            sorted_vals = _np.sort(phq_vals)
            quantiles = [sorted_vals[int(len(sorted_vals) * q / N_REG_BINS)]
                         for q in range(1, N_REG_BINS)]
            strat_labels = [int(_np.searchsorted(quantiles, v)) for v in phq_vals]
        except Exception:
            strat_labels = [int(float(l)) for l in sample_labels]
        bin_counts = Counter(strat_labels)
        if min(bin_counts.values()) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=1.0 - val_ratio,
                                              random_state=seed)
            train_indices, val_indices = next(splitter.split(sample_ids, strat_labels))
        else:
            splitter = ShuffleSplit(n_splits=1, train_size=1.0 - val_ratio,
                                    random_state=seed)
            train_indices, val_indices = next(splitter.split(sample_ids))
    else:
        # 分类任务：按类别标签做分层抽样（原逻辑，增加 random_state）
        label_counts = Counter(int(label) for label in sample_labels)
        splitter: StratifiedShuffleSplit | ShuffleSplit
        if label_counts and min(label_counts.values()) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=1.0 - val_ratio,
                                              random_state=seed)
            train_indices, val_indices = next(splitter.split(sample_ids, sample_labels))
        else:
            splitter = ShuffleSplit(n_splits=1, train_size=1.0 - val_ratio,
                                    random_state=seed)
            train_indices, val_indices = next(splitter.split(sample_ids))

    train_id_split = [sample_ids[index] for index in train_indices]
    val_id_split = [sample_ids[index] for index in val_indices]

    train_id_split = sorted(int(item) for item in train_id_split)
    val_id_split = sorted(int(item) for item in val_id_split)
    train_id_set = set(train_id_split)
    val_id_set = set(val_id_split)

    source_split_map = {int(row["ID"]): "train" for row in rows}
    train_map = {int(row["ID"]): get_task_label(row, task, regression_label) for row in rows if int(row["ID"]) in train_id_set}
    val_map = {int(row["ID"]): get_task_label(row, task, regression_label) for row in rows if int(row["ID"]) in val_id_set}

    payload = {
        "train_ids": train_id_split,
        "val_ids": val_id_split,
        "train_map": train_map,
        "val_map": val_map,
        "source_split_map": source_split_map,
        "rows": rows,
        "split_label": regression_label if task == REGRESSION_TASK else ("label2" if task == "binary" else "label3"),
        "train_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in train_id_set},
        "val_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in val_id_set},
    }
    return payload


def save_split_preview(
    rows: list[dict[str, str]],
    train_ids: list[int],
    val_ids: list[int],
    save_path: str | Path,
) -> Path:
    save_path = resolve_project_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    train_set = set(train_ids)
    val_set = set(val_ids)
    fieldnames = list(rows[0].keys())

    with open(save_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            person_id = int(row["ID"])
            split_name = row.get("split", "train").strip().lower()
            new_row = dict(row)
            if split_name in POOL_SPLITS:
                if person_id in train_set:
                    new_row["split"] = "train"
                elif person_id in val_set:
                    new_row["split"] = "val"
            writer.writerow(new_row)
    return save_path


def load_fold_split(
    fold_csv: str | Path,
    task: str,
    regression_label: str = "label2",
) -> dict[str, Any]:
    """
    从预先生成的 fold CSV（split 列 = "train" 或 "val"）直接加载分割。

    不做任何随机操作，完全确定性——无论运行多少次结果都相同。
    适用于与 generate_kfold_splits.py 配合使用的五折交叉验证流程。

    参数
    ----
    fold_csv         : generate_kfold_splits.py 生成的 fold_k.csv 路径
    task             : "binary" / "ternary" / "regression"
    regression_label : 回归任务使用的标签列（"label2" 或 "label3"）

    返回
    ----
    与 create_train_val_split() 完全相同结构的 dict，
    可直接传给 train.py 的 split_payload 变量。
    """
    csv_path = resolve_project_path(fold_csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Fold CSV 为空: {csv_path}")

    train_rows = [r for r in rows if r.get("split", "").strip().lower() == "train"]
    val_rows   = [r for r in rows if r.get("split", "").strip().lower() == "val"]

    if not train_rows:
        raise ValueError(f"fold CSV 中没有 split=train 的行: {csv_path}")
    if not val_rows:
        raise ValueError(f"fold CSV 中没有 split=val 的行: {csv_path}")

    # source_split_map：Dataset 用来找特征文件所在的物理目录（"train" 或 "test"）
    # 优先读取 fold CSV 中的 data_source 列（由 generate_kfold_splits_expanded.py 写入）；
    # 旧格式 fold CSV 无此列时自动 fallback 到 "train"，保持向后兼容。
    source_split_map = {
        int(r["ID"]): r.get("data_source", "train").strip().lower()
        for r in rows
    }

    train_map = {
        int(r["ID"]): get_task_label(r, task, regression_label)
        for r in train_rows
    }
    val_map = {
        int(r["ID"]): get_task_label(r, task, regression_label)
        for r in val_rows
    }
    train_phq_map = {int(r["ID"]): get_phq9_target(r) for r in train_rows}
    val_phq_map   = {int(r["ID"]): get_phq9_target(r) for r in val_rows}

    split_label = (
        regression_label if task == REGRESSION_TASK
        else ("label2" if task == "binary" else "label3")
    )

    return {
        "train_ids":        sorted(train_map.keys()),
        "val_ids":          sorted(val_map.keys()),
        "train_map":        train_map,
        "val_map":          val_map,
        "source_split_map": source_split_map,
        "rows":             rows,
        "split_label":      split_label,
        "train_phq_map":    train_phq_map,
        "val_phq_map":      val_phq_map,
    }


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a train/val split from official MPDD-AVG train IDs.")
    parser.add_argument("--task", required=True, choices=["binary", "ternary", REGRESSION_TASK])
    parser.add_argument("--regression_label", default="label2", choices=["label2", "label3"])
    parser.add_argument("--split_csv", default="MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_payload = create_train_val_split(
        split_csv=args.split_csv,
        task=args.task,
        val_ratio=args.val_ratio,
        regression_label=args.regression_label,
    )
    if args.save_path:
        preview_path = save_split_preview(
            rows=split_payload["rows"],
            train_ids=split_payload["train_ids"],
            val_ids=split_payload["val_ids"],
            save_path=args.save_path,
        )
        print(json.dumps({"save_path": to_project_relative_path(preview_path)}, ensure_ascii=False))
        return

    summary = {
        "task": args.task,
        "regression_label": args.regression_label if args.task == REGRESSION_TASK else "",
        "val_ratio": args.val_ratio,
        "train_count": len(split_payload["train_ids"]),
        "val_count": len(split_payload["val_ids"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
