"""Auto-pack 6 codabench submission zips after the grid finishes.

What it produces:
    submission_Track1_G+P.zip
    submission_Track1_A-V+P.zip
    submission_Track1_A-V-G+P.zip
    submission_Track2_G+P.zip
    submission_Track2_A-V+P.zip
    submission_Track2_A-V-G+P.zip

Each zip contains:
    binary.csv   (id, binary_pred, phq9_pred)
    ternary.csv  (id, ternary_pred, phq9_pred)

How best checkpoint is chosen per (track, subtrack, task):
    1. Scan logs/<track>/<subtrack-dir>/<task>/pipeline__*/train_result_*.json
    2. Group by (track, subtrack, task) — i.e. ignore audio/video/encoder/seed
    3. Pick the single run with the highest val Macro-F1
       (for regression we'd switch to CCC, but submission format doesn't ask for it)
    4. Load its checkpoint, do test-set inference

PHQ-9 predictions are produced in log1p space inside the model (because
dataset.py:normalize_phq_target wraps with log1p), so we invert with expm1
and clip to [0, 27] before writing to the submission CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MPDDElderDataset, load_task_maps, REGRESSION_TASK
from models.torchcat_baseline import TorchcatBaseline
from train import collate_batch, evaluate_model, summarize_metrics  # reuse plumbing


PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_LOG_DIRS = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P", "G+P": "G-P"}
ALL_SUBTRACKS = ("G+P", "A-V+P", "A-V-G+P")
ALL_TRACKS = ("Track1", "Track2")
# Tasks that produce a label prediction column.
CLASSIFICATION_TASKS = ("binary", "ternary")
# Tasks whose checkpoints we'll consider as PHQ-9 source. We don't train a
# dedicated regression task -- instead we look at the regression head that
# every binary/ternary checkpoint already has, and pick the one with the
# highest val CCC across all binary+ternary runs for this (track, subtrack).
PHQ_SOURCE_TASKS = ("binary", "ternary")


def _find_train_results(
    experiment_prefix: str = "pipeline__",
    newer_than_ts: float | None = None,
    require_full_run: bool = True,
) -> list[dict]:
    """Walk logs/ and load every train_result_*.json that came from this pipeline.

    Filters:
      experiment_prefix : only experiments whose dir name starts with this
      newer_than_ts     : drop results whose mtime is older than this UNIX ts
                          (used to exclude smoke / older grid runs)
      require_full_run  : drop results whose run config used <20 epochs
                          (smoke runs typically use 3-8 epochs and produce
                           degenerate checkpoints)
    """
    out: list[dict] = []
    for path in (PROJECT_ROOT / "logs").rglob("train_result_*.json"):
        parts = path.parts
        # logs/<track>/<sub>/<task>/<experiment_name>/train_result_*.json
        if "logs" not in parts:
            continue
        i = parts.index("logs")
        if len(parts) - i < 6:
            continue
        experiment_name = parts[i + 4]
        if not experiment_name.startswith(experiment_prefix):
            continue
        if newer_than_ts is not None and path.stat().st_mtime < newer_than_ts:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if require_full_run:
            epochs_cfg = (data.get("config") or {}).get("epochs")
            try:
                if epochs_cfg is not None and int(epochs_cfg) < 20:
                    continue
            except (TypeError, ValueError):
                pass
        data["_result_path"] = str(path)
        out.append(data)
    return out


def _best_run_per_group(
    runs: list[dict],
    track: str,
    subtrack: str,
    task: str,
) -> dict | None:
    """Pick the single best run for (track, subtrack, task) by val Macro-F1."""
    candidates = [
        r for r in runs
        if r.get("track") == track
        and r.get("subtrack") == subtrack
        and r.get("task") == task
    ]
    if not candidates:
        return None

    def score(r: dict) -> float:
        m = r.get("best_val_metrics") or {}
        v = m.get("f1") if isinstance(m.get("f1"), (int, float)) else m.get("macro_f1")
        return float(v) if isinstance(v, (int, float)) else -1e9

    return max(candidates, key=score)


def _best_phq_source(
    runs: list[dict],
    track: str,
    subtrack: str,
    candidate_tasks: tuple[str, ...] = PHQ_SOURCE_TASKS,
) -> dict | None:
    """For phq9_pred: pick the single ckpt with the highest val CCC across all
    runs in `candidate_tasks` for this (track, subtrack). The regression head
    is shared between binary and ternary checkpoints, so either may win."""
    candidates = [
        r for r in runs
        if r.get("track") == track
        and r.get("subtrack") == subtrack
        and r.get("task") in candidate_tasks
    ]
    if not candidates:
        return None

    def score(r: dict) -> float:
        m = r.get("best_val_metrics") or {}
        v = m.get("ccc")
        return float(v) if isinstance(v, (int, float)) else -1e9

    return max(candidates, key=score)


def _resolve_checkpoint_path(run: dict) -> Path:
    ckpt = run.get("checkpoint_path") or ""
    p = Path(ckpt)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TorchcatBaseline, dict]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_kwargs = dict(checkpoint["model_kwargs"])
    model = TorchcatBaseline(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _build_test_dataset(checkpoint: dict, track: str, target_t: int) -> MPDDElderDataset:
    sub = "Elder" if track == "Track1" else "Young"
    data_root = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub
    split_csv = data_root / "split_labels_test.csv"
    personality = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-trainval" / sub / "descriptions_embeddings_with_ids.npy"
    task = checkpoint["task"]
    task_maps = load_task_maps(split_csv, task, checkpoint.get("regression_label") or "label2")
    return MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=checkpoint["subtrack"],
        task=task,
        audio_feature=checkpoint["audio_feature"],
        video_feature=checkpoint["video_feature"],
        personality_npy=personality,
        phq_map=task_maps.get("test_phq_map"),
        target_t=target_t,
    )


def _infer_test(run: dict, device: torch.device) -> dict[int, dict[str, float]]:
    """Return {id: {class_pred, phq9_pred}} on the test set for this run's checkpoint."""
    ckpt_path = _resolve_checkpoint_path(run)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, checkpoint = _load_model_from_checkpoint(ckpt_path, device)
    target_t = int(checkpoint.get("target_t", 128))
    test_ds = _build_test_dataset(checkpoint, run["track"], target_t)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)

    import torch.nn as nn
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss())
    metrics = evaluate_model(model, loader, criterion, device, checkpoint["task"])

    ids = metrics["ids"]
    cls = metrics["class_pred"]
    phq_log1p = metrics["phq_pred"]
    out: dict[int, dict[str, float]] = {}
    for i in range(len(ids)):
        pid = int(ids[i])
        pp_log = float(phq_log1p[i])
        pp = float(np.clip(np.expm1(pp_log), 0.0, 27.0))
        out[pid] = {"class_pred": int(cls[i]), "phq9_pred": pp}
    return out


def _write_submission_csv(
    path: Path,
    class_rows_by_id: dict[int, dict[str, float]],
    phq_rows_by_id: dict[int, dict[str, float]],
    task: str,
    test_ids_order: list[int],
) -> None:
    """Write a submission CSV where the label column comes from `class_rows_by_id`
    (trained on `task`) and the phq9_pred column comes from `phq_rows_by_id`
    (trained on the regression task)."""
    label_col = "binary_pred" if task == "binary" else "ternary_pred"
    fields = ["id", label_col, "phq9_pred"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pid in test_ids_order:
            cls = class_rows_by_id.get(pid)
            phq = phq_rows_by_id.get(pid)
            if cls is None:
                raise KeyError(f"No classification prediction for id={pid} in task={task}.")
            if phq is None:
                raise KeyError(f"No regression prediction for id={pid}.")
            writer.writerow({
                "id": pid,
                label_col: int(cls["class_pred"]),
                "phq9_pred": f"{phq['phq9_pred']:.4f}",
            })


def _read_test_ids(track: str) -> list[int]:
    sub = "Elder" if track == "Track1" else "Young"
    csv_path = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub / "split_labels_test.csv"
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Young CSV has whitespace in column names — strip everything.
    ids: list[int] = []
    for row in rows:
        for k, v in row.items():
            if k.strip().upper() == "ID":
                ids.append(int(str(v).strip()))
                break
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="sub/submissions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefix", default="pipeline__",
                        help="Only consider train_result.json from experiments with this name prefix.")
    parser.add_argument("--newer_than", default=None,
                        help="Drop results older than this UNIX timestamp OR file mtime (e.g. "
                             "'experiments/grid_baseline'). Excludes smoke runs and older grids.")
    parser.add_argument("--include_short_runs", action="store_true",
                        help="If set, also consider runs with <20 epochs (default: filter them out).")
    args = parser.parse_args()

    newer_than_ts: float | None = None
    if args.newer_than:
        ref = Path(args.newer_than)
        if ref.exists():
            newer_than_ts = ref.stat().st_mtime
            print(f"[pack] filter: keep results newer than {args.newer_than} (mtime={newer_than_ts:.0f})")
        else:
            try:
                newer_than_ts = float(args.newer_than)
                print(f"[pack] filter: keep results newer than ts={newer_than_ts:.0f}")
            except ValueError:
                raise SystemExit(f"--newer_than must be an existing path or a UNIX timestamp; got {args.newer_than!r}")

    device = torch.device(args.device)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pack] scanning train_result_*.json with prefix '{args.prefix}'")
    runs = _find_train_results(
        experiment_prefix=args.prefix,
        newer_than_ts=newer_than_ts,
        require_full_run=not args.include_short_runs,
    )
    print(f"[pack] found {len(runs)} run records (after filters)")

    test_ids: dict[str, list[int]] = {t: _read_test_ids(t) for t in ALL_TRACKS}
    summary: dict[str, Any] = {"submissions": [], "selected_checkpoints": []}

    for track in ALL_TRACKS:
        for subtrack in ALL_SUBTRACKS:
            zip_name = f"submission_{track}_{subtrack}.zip"
            zip_path = out_dir / zip_name
            per_pack_dir = out_dir / f"_workdir_{track}_{subtrack}"
            per_pack_dir.mkdir(parents=True, exist_ok=True)

            pack_record: dict[str, Any] = {
                "track": track, "subtrack": subtrack, "zip": str(zip_path),
                "tasks": {},
            }

            # ----- Step 1: pick the highest val-CCC run across binary+ternary
            # to source phq9_pred. Both tasks already train a PHQ head.
            best_phq = _best_phq_source(runs, track, subtrack)
            if best_phq is None:
                print(f"[pack] WARN no binary/ternary run for {track}/{subtrack} — skipping pack")
                continue
            phq_ccc = (best_phq.get("best_val_metrics") or {}).get("ccc", 0.0)
            print(f"[pack] {track}/{subtrack}/phq_source: best ccc={phq_ccc:+.3f}  "
                  f"task={best_phq.get('task')}  audio={best_phq.get('audio_feature')}  "
                  f"video={best_phq.get('video_feature')}  "
                  f"seed={(best_phq.get('config') or {}).get('seed')}")
            phq_preds = _infer_test(best_phq, device)
            pack_record["tasks"]["phq_source"] = {
                "checkpoint": best_phq.get("checkpoint_path", ""),
                "source_task": best_phq.get("task"),
                "val_ccc": phq_ccc,
                "val_f1_of_source": (best_phq.get("best_val_metrics") or {}).get("f1"),
                "audio_feature": best_phq.get("audio_feature"),
                "video_feature": best_phq.get("video_feature"),
                "encoder": best_phq.get("encoder_type"),
                "seed": (best_phq.get("config") or {}).get("seed"),
                "test_n_predictions": len(phq_preds),
                "used_for": "phq9_pred column in both binary.csv and ternary.csv",
            }
            summary["selected_checkpoints"].append({
                "track": track, "subtrack": subtrack, "task": "phq_source",
                **pack_record["tasks"]["phq_source"],
            })

            # ----- Step 2: pick best binary + ternary by val Macro-F1, predict labels.
            csv_paths: list[Path] = []
            ok = True
            for task in CLASSIFICATION_TASKS:
                best = _best_run_per_group(runs, track, subtrack, task)
                if best is None:
                    print(f"[pack] WARN no run for {track}/{subtrack}/{task} — skipping pack")
                    ok = False
                    continue
                ckpt_rel = best.get("checkpoint_path", "")
                f1 = (best.get("best_val_metrics") or {}).get("f1", 0.0)
                print(f"[pack] {track}/{subtrack}/{task}: best ckpt val_f1={f1:.3f}  "
                      f"audio={best.get('audio_feature')}  video={best.get('video_feature')}  "
                      f"seed={(best.get('config') or {}).get('seed')}")
                cls_preds = _infer_test(best, device)
                csv_path = per_pack_dir / f"{task}.csv"
                _write_submission_csv(csv_path, cls_preds, phq_preds, task, test_ids[track])
                csv_paths.append(csv_path)
                pack_record["tasks"][task] = {
                    "checkpoint": ckpt_rel,
                    "val_f1": f1,
                    "audio_feature": best.get("audio_feature"),
                    "video_feature": best.get("video_feature"),
                    "encoder": best.get("encoder_type"),
                    "seed": (best.get("config") or {}).get("seed"),
                    "test_n_predictions": len(cls_preds),
                    "csv_path": str(csv_path),
                }
                summary["selected_checkpoints"].append({
                    "track": track, "subtrack": subtrack, "task": task,
                    **pack_record["tasks"][task],
                })

            if not ok or len(csv_paths) != len(CLASSIFICATION_TASKS):
                print(f"[pack] skipping {zip_name} (missing classification task)")
                continue
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for cp in csv_paths:
                    zf.write(cp, arcname=cp.name)
            size_kb = zip_path.stat().st_size / 1024.0
            print(f"[pack] -> {zip_path}  ({size_kb:.1f} KB)")
            pack_record["size_kb"] = round(size_kb, 1)
            summary["submissions"].append(pack_record)

    summary_path = out_dir / "submissions_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\n[pack] summary -> {summary_path}")
    print(f"[pack] {len(summary['submissions'])} submission zips written to {out_dir}")


if __name__ == "__main__":
    main()
