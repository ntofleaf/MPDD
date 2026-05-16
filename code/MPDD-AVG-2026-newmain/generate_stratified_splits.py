"""Generate frozen, balance-optimized train/val splits for each task.

For each (track, task) we:
  1. Build a stratification key
       binary    -> label2
       ternary   -> label3
       regression-> PHQ-9 bin in {0-4, 5-9, 10-14, 15+}
  2. Sample K candidate splits via StratifiedShuffleSplit with different seeds.
  3. Score each candidate by a composite distribution-gap metric covering:
       - class ratio (the stratification key)
       - PHQ-9 mean
       - age mean (parsed from descriptions.csv)
       - gender ratio (Young only)
       - disease distribution (Elder only)
  4. Keep the lowest-score candidate and write a frozen CSV whose ``split`` column
     already encodes train/val membership.

Run:
    python generate_stratified_splits.py --track Track1 --val_ratio 0.1 --n_candidates 400
    python generate_stratified_splits.py --track Track2 --val_ratio 0.1 --n_candidates 400
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent
PHQ_BIN_EDGES = (5, 10, 15)  # produces bins [0,4],[5,9],[10,14],[15,inf]
PHQ_COLS = ("PHQ-9", "phq9_score", "PHQ9", "phq9")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _pid(row: dict[str, str]) -> int:
    return int(str(row["ID"]).strip())


def _get_phq(row: dict[str, str]) -> float:
    for col in PHQ_COLS:
        if col in row and str(row[col]).strip() != "":
            return float(row[col])
    raise KeyError(f"No PHQ-9 column in row: {list(row.keys())}")


def _phq_bin(phq: float) -> int:
    bin_idx = 0
    for edge in PHQ_BIN_EDGES:
        if phq >= edge:
            bin_idx += 1
    return bin_idx  # 0..3


_AGE_RE = re.compile(r"(\d{1,3})\s*-?\s*year[s\-]*\s*old|(\d{1,3})\s+years\s+old", re.IGNORECASE)
_GENDER_RE = re.compile(r"\b(female|male)\b", re.IGNORECASE)
_DISEASE_RE = re.compile(
    r"the patient has\s+([a-z ]+?)\s+diseases?\.", re.IGNORECASE
)


def _parse_descriptions(desc_csv: Path) -> dict[int, dict[str, object]]:
    """Return {pid: {age: int|None, gender: str|None, disease: str|None}}"""
    out: dict[int, dict[str, object]] = {}
    for row in _read_csv_rows(desc_csv):
        try:
            pid = int(str(row["ID"]).strip())
        except (KeyError, ValueError):
            continue
        text = row.get("Descriptions", "") or ""
        age_match = _AGE_RE.search(text)
        age: int | None = None
        if age_match:
            age = int(age_match.group(1) or age_match.group(2))
        gender_match = _GENDER_RE.search(text)
        gender = gender_match.group(1).lower() if gender_match else None
        disease_match = _DISEASE_RE.search(text)
        disease = disease_match.group(1).strip().lower() if disease_match else None
        out[pid] = {"age": age, "gender": gender, "disease": disease}
    return out


def _stratification_key(rows: list[dict[str, str]], task: str, val_ratio: float = 0.1) -> list[int]:
    if task == "binary":
        return [int(float(r["label2"])) for r in rows]
    if task == "ternary":
        return [int(float(r["label3"])) for r in rows]
    if task == "regression":
        keys = [_phq_bin(_get_phq(r)) for r in rows]
        # Auto-merge sparse tail bins so each remaining bin can contribute >=1 to val.
        min_required = max(2, int(np.ceil(1.0 / val_ratio)))
        while True:
            counts = Counter(keys)
            top = max(counts)
            if counts[top] >= min_required or top == 0:
                break
            print(
                f"  [regression] merging PHQ bin {top} (count={counts[top]}) into bin {top-1} "
                f"(min_required={min_required} for val_ratio={val_ratio})"
            )
            keys = [k - 1 if k == top else k for k in keys]
        return keys
    raise ValueError(f"Unknown task={task}")


def _ratio_diff(values_a: list, values_b: list) -> float:
    """Max absolute frequency gap between two categorical lists."""
    cats = set(values_a) | set(values_b)
    if not cats:
        return 0.0
    ca, cb = Counter(values_a), Counter(values_b)
    na, nb = max(len(values_a), 1), max(len(values_b), 1)
    return max(abs(ca[c] / na - cb[c] / nb) for c in cats)


def _tv_distance(values_a: list, values_b: list) -> float:
    """Total-variation distance between two categorical distributions."""
    cats = set(values_a) | set(values_b)
    if not cats:
        return 0.0
    ca, cb = Counter(values_a), Counter(values_b)
    na, nb = max(len(values_a), 1), max(len(values_b), 1)
    return 0.5 * sum(abs(ca[c] / na - cb[c] / nb) for c in cats)


def _normalized_mean_gap(values_a: list[float], values_b: list[float], overall: list[float]) -> float:
    if not values_a or not values_b:
        return 0.0
    sd = float(np.std(overall)) if len(overall) > 1 else 1.0
    sd = sd if sd > 1e-8 else 1.0
    return abs(float(np.mean(values_a)) - float(np.mean(values_b))) / sd


def _score_candidate(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    strat_keys: list[int],
    phq_values: list[float],
    ages: list[int | None],
    genders: list[str | None],
    diseases: list[str | None],
) -> tuple[float, dict[str, float]]:
    def pick(arr, idx):
        return [arr[i] for i in idx]

    train_strat = pick(strat_keys, train_idx)
    val_strat = pick(strat_keys, val_idx)
    class_gap = _ratio_diff(train_strat, val_strat)

    train_phq = pick(phq_values, train_idx)
    val_phq = pick(phq_values, val_idx)
    phq_gap = _normalized_mean_gap(train_phq, val_phq, phq_values)

    train_ages = [a for a in pick(ages, train_idx) if a is not None]
    val_ages = [a for a in pick(ages, val_idx) if a is not None]
    overall_ages = [a for a in ages if a is not None]
    age_gap = _normalized_mean_gap(train_ages, val_ages, overall_ages) if overall_ages else 0.0

    train_gender = [g for g in pick(genders, train_idx) if g is not None]
    val_gender = [g for g in pick(genders, val_idx) if g is not None]
    gender_gap = _ratio_diff(train_gender, val_gender) if train_gender and val_gender else 0.0

    train_dis = [d for d in pick(diseases, train_idx) if d is not None]
    val_dis = [d for d in pick(diseases, val_idx) if d is not None]
    disease_gap = _tv_distance(train_dis, val_dis) if train_dis and val_dis else 0.0

    # Penalize missing classes in val (any class with 0 val samples) heavily.
    val_classes = set(val_strat)
    missing_classes = len(set(strat_keys) - val_classes)

    components = {
        "class_gap": class_gap,
        "phq_mean_gap": phq_gap,
        "age_mean_gap": age_gap,
        "gender_gap": gender_gap,
        "disease_gap": disease_gap,
        "missing_classes": float(missing_classes),
    }
    # Weights chosen so each is roughly O(0.1-0.3); class & missing are dominant.
    score = (
        2.0 * class_gap
        + 1.0 * phq_gap
        + 0.5 * age_gap
        + 0.5 * gender_gap
        + 1.0 * disease_gap
        + 10.0 * missing_classes
    )
    return score, components


def _generate_for_task(
    rows: list[dict[str, str]],
    descriptions: dict[int, dict[str, object]],
    task: str,
    val_ratio: float,
    n_candidates: int,
    base_seed: int,
) -> dict:
    pids = [_pid(r) for r in rows]
    strat_keys = _stratification_key(rows, task, val_ratio=val_ratio)
    phq_values = [_get_phq(r) for r in rows]
    ages = [descriptions.get(p, {}).get("age") for p in pids]
    genders = [descriptions.get(p, {}).get("gender") for p in pids]
    diseases = [descriptions.get(p, {}).get("disease") for p in pids]

    counts = Counter(strat_keys)
    if min(counts.values()) < 2:
        raise ValueError(
            f"Cannot stratify task={task}: a class has <2 samples ({dict(counts)})"
        )

    best_score = None
    best_train: np.ndarray | None = None
    best_val: np.ndarray | None = None
    best_components: dict[str, float] = {}
    best_seed = -1

    indices = np.arange(len(pids))
    for trial in range(n_candidates):
        seed = base_seed + trial
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=seed
            )
            train_idx, val_idx = next(splitter.split(indices, strat_keys))
        except ValueError:
            continue
        score, components = _score_candidate(
            train_idx, val_idx,
            strat_keys=strat_keys,
            phq_values=phq_values,
            ages=ages,
            genders=genders,
            diseases=diseases,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_train = train_idx
            best_val = val_idx
            best_components = components
            best_seed = seed

    assert best_train is not None and best_val is not None
    train_pids = sorted(int(pids[i]) for i in best_train)
    val_pids = sorted(int(pids[i]) for i in best_val)

    # Stats for the chosen split
    def _stats(idx_list):
        sel = [i for i, p in enumerate(pids) if p in idx_list]
        return {
            "n": len(sel),
            "class_counts": dict(Counter(strat_keys[i] for i in sel)),
            "phq_mean": float(np.mean([phq_values[i] for i in sel])),
            "age_mean": (
                float(np.mean([ages[i] for i in sel if ages[i] is not None]))
                if any(ages[i] is not None for i in sel) else None
            ),
            "gender_counts": dict(Counter(g for g in (genders[i] for i in sel) if g)),
            "disease_counts": dict(Counter(d for d in (diseases[i] for i in sel) if d)),
        }

    return {
        "task": task,
        "best_seed": best_seed,
        "score": best_score,
        "components": best_components,
        "train_ids": train_pids,
        "val_ids": val_pids,
        "train_stats": _stats(set(train_pids)),
        "val_stats": _stats(set(val_pids)),
    }


def _write_frozen_csv(
    rows: list[dict[str, str]],
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    save_path: Path,
) -> None:
    train_set = {int(x) for x in train_ids}
    val_set = {int(x) for x in val_ids}
    fieldnames = list(rows[0].keys())
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pid = _pid(row)
            new_row = dict(row)
            if pid in val_set:
                new_row["split"] = "val"
            elif pid in train_set:
                new_row["split"] = "train"
            writer.writerow(new_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["Track1", "Track2"])
    parser.add_argument("--data_root", default=None, help="Override data root.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--n_candidates", type=int, default=400)
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--out_dir", default=None,
                        help="Where to write frozen split CSVs. Defaults to data_root.")
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
        "n_candidates_per_task": args.n_candidates,
        "tasks": {},
    }

    for task in ("binary", "ternary", "regression"):
        result = _generate_for_task(
            rows=rows,
            descriptions=descriptions,
            task=task,
            val_ratio=args.val_ratio,
            n_candidates=args.n_candidates,
            base_seed=args.base_seed,
        )
        out_path = out_dir / f"split_labels_train_{task}_frozen.csv"
        _write_frozen_csv(rows, result["train_ids"], result["val_ids"], out_path)
        result["frozen_csv"] = str(out_path)
        summary["tasks"][task] = result
        print(
            f"[{args.track}/{task}] score={result['score']:.4f}  "
            f"seed={result['best_seed']}  "
            f"train={result['train_stats']['n']} val={result['val_stats']['n']}  "
            f"-> {out_path.name}"
        )
        print("  train:", json.dumps(result["train_stats"], ensure_ascii=False))
        print("  val  :", json.dumps(result["val_stats"], ensure_ascii=False))
        print("  components:", json.dumps(result["components"], ensure_ascii=False))

    summary_path = out_dir / "stratified_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
