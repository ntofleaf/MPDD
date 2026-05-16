"""Reproducible training pipeline for MPDD-AVG experiments.

The whole point of this script is to enforce a fair, reproducible evaluation:

  - **Frozen splits.** Train/val (and 5-fold) partitions come from the CSVs
    produced by ``generate_stratified_splits.py`` / ``generate_kfold_splits.py``.
    They are written once and never re-randomized.
  - **Fixed training procedure.** Every knob other than the feature combination,
    seed, and fold is pinned in ``FIXED_HPARAMS`` at the top of this file. To
    change one, edit this dict in code (and commit the change) so the diff
    makes it obvious what moved.
  - **Single experiment variable.** A "combo" is a 4-tuple
    ``(subtrack, audio_feature, video_feature, encoder)``. The pipeline lets
    you sweep combos, seeds, and folds — but *only* those.
  - **Aggregation.** When you run multiple seeds and/or folds, the pipeline
    summarises mean/std/min/max of every metric *and* flags class collapse
    (Kappa ≤ 0 and the model predicts a single class on val).

Modes:
  single          one (combo, seed, frozen val) run
  multi_seed      one combo, N seeds, evaluated on the frozen val
  kfold           one combo, 5 folds with a fixed seed
  kfold_multiseed one combo, 5 folds × N seeds (full re-training stability test)
  grid            sweep combos × (seeds or folds), full report

Each underlying training call delegates to ``train.py`` so we reuse the
existing, tested model + loss + early-stopping code. We only orchestrate
configuration, splits, and reporting.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# FIXED training hyperparameters — change here in code, not via CLI flags.
# Anything not listed here is whatever train.py defaults to.
# ---------------------------------------------------------------------------
FIXED_HPARAMS: dict[str, str] = {
    "epochs": "140",
    "batch_size": "4",
    "lr": "8e-5",
    "weight_decay": "1e-5",
    "hidden_dim": "128",
    "dropout": "0.45",
    "patience": "35",
    "min_delta": "1e-4",
    "target_t": "128",
    "val_ratio": "0.1",   # ignored when frozen split CSV has val rows
}

DEFAULT_SEEDS = (42, 3407, 2026, 7, 123)


# ---------------------------------------------------------------------------
@dataclass
class Combo:
    track: str       # Track1 | Track2
    task: str        # binary | ternary | regression
    subtrack: str    # G+P | A-V+P | A-V-G+P
    audio_feature: str
    video_feature: str
    encoder: str = "bilstm_mean"

    @property
    def slug(self) -> str:
        return (
            f"{self.subtrack.replace('+', 'p').replace('-', '_')}"
            f"__{self.audio_feature}__{self.video_feature}__{self.encoder}"
        )

    def data_root(self) -> Path:
        sub = "Elder" if self.track == "Track1" else "Young"
        return PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-trainval" / sub


@dataclass
class RunSpec:
    combo: Combo
    seed: int
    split_csv: Path
    run_tag: str            # e.g. "frozen_val_seed42" or "fold3_seed42"
    experiment_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.experiment_name = f"pipeline__{self.combo.slug}__{self.run_tag}"


# ---------------------------------------------------------------------------
def _find_frozen_single(combo: Combo) -> Path:
    return combo.data_root() / f"split_labels_train_{combo.task}_frozen.csv"


def _find_frozen_fold(combo: Combo, fold: int) -> Path:
    return combo.data_root() / f"split_labels_train_{combo.task}_fold{fold}_frozen.csv"


def _personality_npy(combo: Combo) -> Path:
    return combo.data_root() / "descriptions_embeddings_with_ids.npy"


def _build_train_cmd(run: RunSpec) -> list[str]:
    combo = run.combo
    cmd = [
        "python", str(PROJECT_ROOT / "train.py"),
        "--track", combo.track,
        "--task", combo.task,
        "--subtrack", combo.subtrack,
        "--encoder_type", combo.encoder,
        "--audio_feature", combo.audio_feature,
        "--video_feature", combo.video_feature,
        "--experiment_name", run.experiment_name,
        "--data_root", str(combo.data_root().relative_to(PROJECT_ROOT)),
        "--split_csv", (
            str(run.split_csv.relative_to(PROJECT_ROOT))
            if run.split_csv.is_relative_to(PROJECT_ROOT) else str(run.split_csv)
        ),
        "--personality_npy", str(_personality_npy(combo).relative_to(PROJECT_ROOT)),
        "--seed", str(run.seed),
        "--device", "cuda",
    ]
    for key, value in FIXED_HPARAMS.items():
        cmd.extend([f"--{key}", value])
    return cmd


_SUBTRACK_LOG_DIRS = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P", "G+P": "G-P"}


def _result_path_for(run: RunSpec) -> Path | None:
    """Locate the train_result_*.json that train.py wrote for this run."""
    sub_dir = _SUBTRACK_LOG_DIRS.get(run.combo.subtrack, run.combo.subtrack.replace("+", "-"))
    log_dir = PROJECT_ROOT / "logs" / run.combo.track / sub_dir / run.combo.task / run.experiment_name
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob("train_result_*.json"))
    return candidates[-1] if candidates else None


def _run_one(run: RunSpec, gpu: int, log_root: Path, verbose: bool) -> dict:
    cmd = _build_train_cmd(run)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{run.experiment_name}__gpu{gpu}.log"
    t0 = time.time()
    if verbose:
        print(f"  [start] gpu={gpu} {run.experiment_name}", flush=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    result_path = _result_path_for(run)
    record: dict = {
        "experiment_name": run.experiment_name,
        "run_tag": run.run_tag,
        "seed": run.seed,
        "split_csv": str(run.split_csv),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_log": str(log_path),
        "result_path": str(result_path) if result_path else None,
    }
    if result_path and result_path.exists():
        with open(result_path, "r", encoding="utf-8") as handle:
            record["train_result"] = json.load(handle)
    if verbose:
        if proc.returncode != 0:
            print(f"  [FAIL ] gpu={gpu} ({elapsed:.1f}s) -> tail {log_path}", flush=True)
        else:
            best = (record.get("train_result") or {}).get("best_val_metrics") or {}
            f1 = best.get("f1") if isinstance(best.get("f1"), (int, float)) else best.get("macro_f1")
            f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else "N/A"
            print(f"  [done ] gpu={gpu} ({elapsed:.1f}s) {run.experiment_name}  f1={f1_str}", flush=True)
    return record


# ---------------------------------------------------------------------------
def _execute_runs(runs: list[RunSpec], gpus: list[int], log_root: Path, verbose: bool) -> list[dict]:
    if not gpus:
        gpus = [0]

    task_queue: queue.Queue[RunSpec] = queue.Queue()
    for run in runs:
        task_queue.put(run)
    results: list[dict] = []
    results_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                run = task_queue.get_nowait()
            except queue.Empty:
                return
            record = _run_one(run, gpu_id, log_root, verbose)
            with results_lock:
                results.append(record)
            task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Preserve a deterministic order (matches the input run list)
    by_name = {r["experiment_name"]: r for r in results}
    return [by_name[r.experiment_name] for r in runs if r.experiment_name in by_name]


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def _per_class_recall(cm: list[list[int]]) -> list[float]:
    recalls = []
    for i, row in enumerate(cm):
        total = sum(row)
        recalls.append(row[i] / total if total else 0.0)
    return recalls


def _collapse_flag(cm: list[list[int]], kappa: float | None) -> bool:
    """True if the model essentially predicts one class on val."""
    if kappa is not None and kappa > 0.05:
        return False
    columns_with_preds = sum(1 for c in range(len(cm)) if sum(cm[r][c] for r in range(len(cm))) > 0)
    return columns_with_preds <= 1


def _summarize_runs(runs: list[dict]) -> dict:
    """Mean ± std of every metric across a list of completed runs."""
    metric_keys = ["acc", "f1", "kappa", "ccc", "rmse", "mae"]
    by_metric: dict[str, list[float]] = {k: [] for k in metric_keys}
    per_class_recalls: list[list[float]] = []
    collapse_count = 0
    n_classes = 0

    for r in runs:
        tr = r.get("train_result") or {}
        b = tr.get("best_val_metrics") or {}
        cm = b.get("confusion_matrix")
        if cm:
            recalls = _per_class_recall(cm)
            per_class_recalls.append(recalls)
            n_classes = max(n_classes, len(recalls))
            if _collapse_flag(cm, b.get("kappa")):
                collapse_count += 1
        for k in metric_keys:
            v = b.get(k) if k != "f1" else (b.get("f1") or b.get("macro_f1"))
            if isinstance(v, (int, float)):
                by_metric[k].append(float(v))

    summary = {"n_runs": len(runs), "collapse_count": collapse_count, "metrics": {}}
    for k, values in by_metric.items():
        if values:
            summary["metrics"][k] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min":  float(np.min(values)),
                "max":  float(np.max(values)),
            }
    if per_class_recalls:
        arr = np.array(per_class_recalls)
        summary["per_class_recall_mean"] = arr.mean(axis=0).tolist()
        summary["per_class_recall_min"] = arr.min(axis=0).tolist()
    return summary


def _print_combo_summary(label: str, summary: dict) -> None:
    print(f"\n  === {label} ===")
    print(f"    runs={summary['n_runs']}  collapse={summary['collapse_count']}")
    for k, s in summary.get("metrics", {}).items():
        print(f"    {k:>6}: mean={s['mean']:+.3f}  std={s['std']:.3f}  range=[{s['min']:+.3f}, {s['max']:+.3f}]")
    if "per_class_recall_mean" in summary:
        rec = " ".join(f"{x:.2f}" for x in summary["per_class_recall_mean"])
        print(f"    recall_per_class_mean: [{rec}]")
        rec_min = " ".join(f"{x:.2f}" for x in summary["per_class_recall_min"])
        print(f"    recall_per_class_min:  [{rec_min}]  <-- worst-case across runs")


# ---------------------------------------------------------------------------
def _expand_runs(args: argparse.Namespace) -> list[RunSpec]:
    seeds: list[int] = list(args.seeds) if args.seeds else [args.seed]
    combos: list[Combo] = [
        Combo(args.track, args.task, st, af, vf, args.encoder)
        for st, af, vf in itertools.product(args.grid_subtracks, args.grid_audios, args.grid_videos)
    ]
    runs: list[RunSpec] = []
    if args.mode == "single":
        c = combos[0]
        split = _find_frozen_single(c)
        if not split.exists():
            raise FileNotFoundError(f"Frozen single split not found: {split}")
        runs.append(RunSpec(c, seeds[0], split, run_tag=f"frozen_val_seed{seeds[0]}"))
    elif args.mode == "multi_seed":
        c = combos[0]
        split = _find_frozen_single(c)
        if not split.exists():
            raise FileNotFoundError(f"Frozen single split not found: {split}")
        for s in seeds:
            runs.append(RunSpec(c, s, split, run_tag=f"frozen_val_seed{s}"))
    elif args.mode == "kfold":
        c = combos[0]
        for f in range(args.n_splits):
            fold_csv = _find_frozen_fold(c, f)
            if not fold_csv.exists():
                raise FileNotFoundError(f"Frozen fold split not found: {fold_csv}")
            runs.append(RunSpec(c, seeds[0], fold_csv, run_tag=f"fold{f}_seed{seeds[0]}"))
    elif args.mode == "kfold_multiseed":
        c = combos[0]
        for f in range(args.n_splits):
            fold_csv = _find_frozen_fold(c, f)
            if not fold_csv.exists():
                raise FileNotFoundError(f"Frozen fold split not found: {fold_csv}")
            for s in seeds:
                runs.append(RunSpec(c, s, fold_csv, run_tag=f"fold{f}_seed{s}"))
    elif args.mode == "grid":
        for c in combos:
            if args.eval == "frozen_val":
                split = _find_frozen_single(c)
                for s in seeds:
                    runs.append(RunSpec(c, s, split, run_tag=f"frozen_val_seed{s}"))
            else:  # kfold
                for f in range(args.n_splits):
                    fold_csv = _find_frozen_fold(c, f)
                    for s in seeds:
                        runs.append(RunSpec(c, s, fold_csv, run_tag=f"fold{f}_seed{s}"))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Deduplicate: G+P subtrack ignores audio/video features at the dataset level
    # (need_av=False), so iterating over audio/video grids would launch identical
    # models. Collapse them to one representative run.
    seen: set[tuple] = set()
    deduped: list[RunSpec] = []
    for r in runs:
        if r.combo.subtrack == "G+P":
            key = (r.combo.track, r.combo.task, "G+P", r.combo.encoder,
                   r.seed, str(r.split_csv))
        else:
            key = (r.combo.track, r.combo.task, r.combo.subtrack,
                   r.combo.audio_feature, r.combo.video_feature, r.combo.encoder,
                   r.seed, str(r.split_csv))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _group_by_combo_and_split_kind(runs: list[dict], specs: list[RunSpec]) -> dict[str, dict]:
    """Group runs by combo slug; if kfold mode, also keep per-fold buckets."""
    by_name = {s.experiment_name: s for s in specs}
    grouped: dict[str, dict] = {}
    for record in runs:
        spec = by_name[record["experiment_name"]]
        slug = spec.combo.slug
        grouped.setdefault(slug, {"combo": spec.combo, "runs_all": [], "runs_by_fold": {}})
        grouped[slug]["runs_all"].append(record)
        fold_tag = spec.run_tag.split("_seed")[0]  # "fold3" or "frozen_val"
        grouped[slug]["runs_by_fold"].setdefault(fold_tag, []).append(record)
    return grouped


def _write_grid_summary_csv(grouped: dict[str, dict], out_path: Path) -> None:
    fieldnames = [
        "combo", "track", "task", "subtrack", "audio", "video", "encoder",
        "n_runs", "collapse_count",
        "f1_mean", "f1_std", "acc_mean", "kappa_mean", "ccc_mean", "rmse_mean", "mae_mean",
        "recall_min_class_mean", "recall_min_class_min",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for slug, bucket in sorted(grouped.items(), key=lambda kv: -((_summarize_runs(kv[1]["runs_all"]).get("metrics", {}).get("f1", {}) or {}).get("mean", -1))):
            c: Combo = bucket["combo"]
            s = _summarize_runs(bucket["runs_all"])
            m = s.get("metrics", {})
            row = {
                "combo": slug,
                "track": c.track, "task": c.task, "subtrack": c.subtrack,
                "audio": c.audio_feature, "video": c.video_feature, "encoder": c.encoder,
                "n_runs": s["n_runs"], "collapse_count": s["collapse_count"],
                "f1_mean":   round(m.get("f1", {}).get("mean", float("nan")), 4),
                "f1_std":    round(m.get("f1", {}).get("std",  float("nan")), 4),
                "acc_mean":  round(m.get("acc", {}).get("mean", float("nan")), 4),
                "kappa_mean":round(m.get("kappa", {}).get("mean", float("nan")), 4),
                "ccc_mean":  round(m.get("ccc", {}).get("mean", float("nan")), 4),
                "rmse_mean": round(m.get("rmse", {}).get("mean", float("nan")), 4),
                "mae_mean":  round(m.get("mae", {}).get("mean", float("nan")), 4),
                "recall_min_class_mean": round(min(s.get("per_class_recall_mean", [0.0])), 4),
                "recall_min_class_min":  round(min(s.get("per_class_recall_min",  [0.0])), 4),
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", required=True, choices=["Track1", "Track2"])
    parser.add_argument("--task", required=True, choices=["binary", "ternary", "regression"])
    parser.add_argument("--mode", required=True,
                        choices=["single", "multi_seed", "kfold", "kfold_multiseed", "grid"])

    # Combo args (single-combo modes)
    parser.add_argument("--subtrack", default="A-V-G+P", choices=["G+P", "A-V+P", "A-V-G+P"])
    parser.add_argument("--audio_feature", default="wav2vec")
    parser.add_argument("--video_feature", default="resnet")
    parser.add_argument("--encoder", default="bilstm_mean", choices=["bilstm_mean", "hybrid_attn"])

    # Grid args
    parser.add_argument("--grid_subtracks", default=None,
                        help="Comma-separated subtracks for grid mode. Default = [--subtrack].")
    parser.add_argument("--grid_audios", default=None,
                        help="Comma-separated audio features. Default = [--audio_feature].")
    parser.add_argument("--grid_videos", default=None,
                        help="Comma-separated video features. Default = [--video_feature].")
    parser.add_argument("--eval", choices=["frozen_val", "kfold"], default="frozen_val",
                        help="Inner evaluation protocol when --mode grid.")

    # Seeds / folds
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (overrides --seed).")
    parser.add_argument("--n_splits", type=int, default=5)

    # Execution
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for parallel workers.")
    parser.add_argument("--output_dir", default="experiments/_pipeline_runs",
                        help="Where this script writes its own logs + aggregate JSON.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Resolve list-style args
    args.seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    args.grid_subtracks = (args.grid_subtracks.split(",") if args.grid_subtracks else [args.subtrack])
    args.grid_audios = (args.grid_audios.split(",") if args.grid_audios else [args.audio_feature])
    args.grid_videos = (args.grid_videos.split(",") if args.grid_videos else [args.video_feature])
    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip() != ""]

    runs = _expand_runs(args)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    log_root = output_dir / "stdout_logs"

    print(f"[pipeline] mode={args.mode} task={args.task} track={args.track}")
    print(f"[pipeline] fixed hyperparams: {FIXED_HPARAMS}")
    print(f"[pipeline] runs={len(runs)} gpus={gpus}")
    for r in runs:
        print(f"  - {r.experiment_name}  (split={r.split_csv.name}, seed={r.seed})")

    records = _execute_runs(runs, gpus, log_root, verbose=not args.quiet)

    grouped = _group_by_combo_and_split_kind(records, runs)
    for slug, bucket in grouped.items():
        s = _summarize_runs(bucket["runs_all"])
        _print_combo_summary(slug, s)
        if len(bucket["runs_by_fold"]) > 1:
            print(f"    per-fold breakdown:")
            for fold_tag, fold_runs in sorted(bucket["runs_by_fold"].items()):
                fs = _summarize_runs(fold_runs)
                m = fs.get("metrics", {})
                f1 = m.get("f1", {})
                kp = m.get("kappa", {})
                cc = m.get("ccc", {})
                print(
                    f"      {fold_tag:>10}: n={fs['n_runs']} collapse={fs['collapse_count']} "
                    f"f1={f1.get('mean', float('nan')):+.3f}±{f1.get('std', 0):.3f} "
                    f"kappa={kp.get('mean', float('nan')):+.3f} "
                    f"ccc={cc.get('mean', float('nan')):+.3f}"
                )

    # Write aggregate JSON + grid CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S")
    agg_path = output_dir / f"aggregate_{args.track}_{args.task}_{args.mode}_{timestamp}.json"
    aggregate = {
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
        "fixed_hparams": FIXED_HPARAMS,
        "groups": {
            slug: {
                "combo": vars(bucket["combo"]),
                "summary_all": _summarize_runs(bucket["runs_all"]),
                "summary_by_fold": {
                    ft: _summarize_runs(rs) for ft, rs in bucket["runs_by_fold"].items()
                },
                "run_records": bucket["runs_all"],
            }
            for slug, bucket in grouped.items()
        },
    }
    with open(agg_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\n[pipeline] aggregate -> {agg_path}")

    if args.mode == "grid":
        csv_path = output_dir / f"grid_summary_{args.track}_{args.task}_{timestamp}.csv"
        _write_grid_summary_csv(grouped, csv_path)
        print(f"[pipeline] grid CSV -> {csv_path}")


if __name__ == "__main__":
    main()
