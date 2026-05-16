"""Generate frozen, balance-optimized 5-fold splits for each task.

Same balance scoring as ``generate_stratified_splits.py``, but instead of one
train/val split it produces K = 5 folds. The chosen fold assignment is the one
whose **per-fold** distribution most closely matches the overall distribution
across all balance metrics (class ratio, PHQ mean, age mean, gender ratio,
disease distribution, no class missing in any fold).

Output: one CSV per (track, task) with an extra ``fold_id`` column in {0..4},
plus the original ``split`` column kept as ``train`` (the official train split
on disk is what we are partitioning).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Reuse helpers
from generate_stratified_splits import (
    PROJECT_ROOT,
    _read_csv_rows,
    _pid,
    _get_phq,
    _parse_descriptions,
    _stratification_key,
    _normalized_mean_gap,
    _ratio_diff,
    _tv_distance,
)


def _score_kfold(
    fold_assignment: np.ndarray,
    *,
    n_splits: int,
    strat_keys: list[int],
    phq_values: list[float],
    ages: list,
    genders: list,
    diseases: list,
) -> tuple[float, dict[str, float]]:
    """Score a full K-fold assignment: sum of per-fold gaps vs the whole dataset.

    For each fold, treat that fold as ``val`` and the rest as ``train``, and
    compute the same composite gap used in the single-split scoring.
    """
    all_class_gap: list[float] = []
    all_phq_gap: list[float] = []
    all_age_gap: list[float] = []
    all_gender_gap: list[float] = []
    all_disease_gap: list[float] = []
    missing_total = 0

    def take(arr, indices):
        return [arr[i] for i in indices]

    age_overall = [a for a in ages if a is not None]

    for k in range(n_splits):
        val_idx = np.where(fold_assignment == k)[0]
        train_idx = np.where(fold_assignment != k)[0]
        if len(val_idx) == 0 or len(train_idx) == 0:
            return 1e9, {"reason": "empty fold"}

        train_strat = take(strat_keys, train_idx)
        val_strat = take(strat_keys, val_idx)
        all_class_gap.append(_ratio_diff(train_strat, val_strat))

        all_phq_gap.append(_normalized_mean_gap(
            take(phq_values, train_idx),
            take(phq_values, val_idx),
            phq_values,
        ))

        train_ages = [a for a in take(ages, train_idx) if a is not None]
        val_ages = [a for a in take(ages, val_idx) if a is not None]
        all_age_gap.append(
            _normalized_mean_gap(train_ages, val_ages, age_overall) if age_overall else 0.0
        )

        tr_g = [g for g in take(genders, train_idx) if g is not None]
        va_g = [g for g in take(genders, val_idx) if g is not None]
        all_gender_gap.append(_ratio_diff(tr_g, va_g) if tr_g and va_g else 0.0)

        tr_d = [d for d in take(diseases, train_idx) if d is not None]
        va_d = [d for d in take(diseases, val_idx) if d is not None]
        all_disease_gap.append(_tv_distance(tr_d, va_d) if tr_d and va_d else 0.0)

        # Missing-class penalty: any class in the dataset that is absent from val
        missing_total += len(set(strat_keys) - set(val_strat))

    components = {
        "class_gap_mean": float(np.mean(all_class_gap)),
        "class_gap_max":  float(np.max(all_class_gap)),
        "phq_mean_gap_mean":  float(np.mean(all_phq_gap)),
        "age_mean_gap_mean":  float(np.mean(all_age_gap)),
        "gender_gap_mean":    float(np.mean(all_gender_gap)),
        "disease_gap_mean":   float(np.mean(all_disease_gap)),
        "missing_classes_total": float(missing_total),
    }
    score = (
        2.0 * components["class_gap_mean"]
        + 1.5 * components["class_gap_max"]
        + 1.0 * components["phq_mean_gap_mean"]
        + 0.5 * components["age_mean_gap_mean"]
        + 0.5 * components["gender_gap_mean"]
        + 1.0 * components["disease_gap_mean"]
        + 5.0 * components["missing_classes_total"]
    )
    return score, components


def _generate_kfold_for_task(
    rows: list[dict[str, str]],
    descriptions: dict[int, dict[str, object]],
    task: str,
    n_splits: int,
    n_candidates: int,
    base_seed: int,
) -> dict:
    pids = [_pid(r) for r in rows]
    # Auto-merge sparse PHQ bins for regression: each remaining class must allow
    # at least one sample per fold, i.e. count >= n_splits.
    val_ratio_for_merge = 1.0 / n_splits
    strat_keys = _stratification_key(rows, task, val_ratio=val_ratio_for_merge)
    phq_values = [_get_phq(r) for r in rows]
    ages = [descriptions.get(p, {}).get("age") for p in pids]
    genders = [descriptions.get(p, {}).get("gender") for p in pids]
    diseases = [descriptions.get(p, {}).get("disease") for p in pids]

    counts = Counter(strat_keys)
    if min(counts.values()) < n_splits:
        # If even after merging we still have a class with <K samples, we can
        # technically still run KFold but some folds will miss that class.
        # The missing-class penalty will dominate the score, so we tolerate it.
        print(
            f"  [warn] task={task}: class counts {dict(counts)} smaller than n_splits={n_splits};"
            f" some folds may miss minority classes."
        )

    best_score = None
    best_assignment: np.ndarray | None = None
    best_components: dict[str, float] = {}
    best_seed = -1

    indices = np.arange(len(pids))
    for trial in range(n_candidates):
        seed = base_seed + trial
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        assignment = np.full(len(pids), -1, dtype=np.int32)
        try:
            for fold_idx, (_, val_idx) in enumerate(kfold.split(indices, strat_keys)):
                assignment[val_idx] = fold_idx
        except ValueError:
            continue
        score, components = _score_kfold(
            assignment, n_splits=n_splits,
            strat_keys=strat_keys, phq_values=phq_values,
            ages=ages, genders=genders, diseases=diseases,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_components = components
            best_seed = seed

    assert best_assignment is not None

    fold_stats = []
    for k in range(n_splits):
        val_idx = np.where(best_assignment == k)[0]
        train_idx = np.where(best_assignment != k)[0]
        fold_stats.append({
            "fold": k,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "val_class_counts": dict(Counter(strat_keys[i] for i in val_idx)),
            "val_phq_mean": float(np.mean([phq_values[i] for i in val_idx])),
            "val_age_mean": (
                float(np.mean([ages[i] for i in val_idx if ages[i] is not None]))
                if any(ages[i] is not None for i in val_idx) else None
            ),
        })

    return {
        "task": task,
        "n_splits": n_splits,
        "best_seed": best_seed,
        "score": best_score,
        "components": best_components,
        "fold_assignment": [int(x) for x in best_assignment.tolist()],
        "fold_stats": fold_stats,
    }


def _write_kfold_csv(
    rows: list[dict[str, str]],
    fold_assignment: list[int],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if "fold_id" not in fieldnames:
        fieldnames.append("fold_id")
    with open(save_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, fold_id in zip(rows, fold_assignment):
            new_row = dict(row)
            new_row["split"] = "train"  # full pool is officially train
            new_row["fold_id"] = int(fold_id)
            writer.writerow(new_row)


def _write_per_fold_csvs(
    rows: list[dict[str, str]],
    fold_assignment: list[int],
    out_dir: Path,
    task: str,
    n_splits: int,
) -> list[Path]:
    """Also emit fold_{k}_frozen.csv where split=val for fold k, else train.

    This lets train.py be invoked unchanged with --split_csv pointing here.
    """
    paths = []
    fieldnames = list(rows[0].keys())
    if "fold_id" in fieldnames:
        fieldnames.remove("fold_id")
    for k in range(n_splits):
        path = out_dir / f"split_labels_train_{task}_fold{k}_frozen.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, fold_id in zip(rows, fold_assignment):
                new_row = {k_: row[k_] for k_ in fieldnames}
                new_row["split"] = "val" if fold_id == k else "train"
                writer.writerow(new_row)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["Track1", "Track2"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_candidates", type=int, default=400)
    parser.add_argument("--base_seed", type=int, default=2000)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        sub = "Elder" if args.track == "Track1" else "Young"
        data_root = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-trainval" / sub
    data_root = data_root.resolve()
    split_csv = data_root / "split_labels_train.csv"
    desc_csv = data_root / "descriptions.csv"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_root

    rows = _read_csv_rows(split_csv)
    descriptions = _parse_descriptions(desc_csv) if desc_csv.exists() else {}

    summary: dict[str, object] = {
        "track": args.track,
        "data_root": str(data_root),
        "n_samples": len(rows),
        "n_splits": args.n_splits,
        "n_candidates_per_task": args.n_candidates,
        "tasks": {},
    }

    for task in ("binary", "ternary", "regression"):
        result = _generate_kfold_for_task(
            rows=rows, descriptions=descriptions,
            task=task,
            n_splits=args.n_splits,
            n_candidates=args.n_candidates,
            base_seed=args.base_seed,
        )
        # Index CSV (all rows with fold_id)
        index_path = out_dir / f"split_labels_train_{task}_kfold.csv"
        _write_kfold_csv(rows, result["fold_assignment"], index_path)
        per_fold_paths = _write_per_fold_csvs(
            rows, result["fold_assignment"], out_dir, task, args.n_splits,
        )
        result["kfold_index_csv"] = str(index_path)
        result["per_fold_csvs"] = [str(p) for p in per_fold_paths]
        summary["tasks"][task] = result
        print(
            f"[{args.track}/{task}] score={result['score']:.4f}  "
            f"seed={result['best_seed']}  index={index_path.name}"
        )
        for fs in result["fold_stats"]:
            print(
                f"  fold {fs['fold']}: n_val={fs['n_val']:2d}  "
                f"class={fs['val_class_counts']}  phq_mean={fs['val_phq_mean']:.2f}"
            )
        print("  components:", json.dumps(result["components"], ensure_ascii=False))

    summary_path = out_dir / "kfold_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
