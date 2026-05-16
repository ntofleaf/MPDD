"""5-fold validated, fold-ensembled submission packer.

For each (track, subtrack, task) cell:
  1. Scan all kfold runs (run_tag like 'fold{k}_seed42').
  2. Group by combo (audio, video, encoder). A 'complete' combo has 5 folds.
  3. Score each complete combo:
        classification: mean_f1 - std_f1 - 0.5 * collapse_count
        regression src: mean_ccc - 0.5 * std_ccc  (across binary+ternary kfold)
  4. Pick the best combo. Load its 5 fold ckpts. Run test inference for each,
     average logits, then argmax for class_pred and mean for phq_pred.
  5. Write binary.csv + ternary.csv + zip per (track, subtrack).

Output layout:
    submissions_v2/
      submission_<track>_<subtrack>.zip
      submissions_summary.json
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
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MPDDElderDataset, load_task_maps, REGRESSION_TASK
from models.torchcat_baseline import TorchcatBaseline
from train import collate_batch

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_SUBTRACKS = ("G+P", "A-V+P", "A-V-G+P")
ALL_TRACKS = ("Track1", "Track2")
CLASSIFICATION_TASKS = ("binary", "ternary")
PHQ_SOURCE_TASKS = ("binary", "ternary")


# ---------------------------------------------------------------------------
def _is_collapse(b: dict) -> bool:
    """Mirror the collapse heuristic used in run_pipeline.py."""
    kappa = b.get("kappa")
    cm = b.get("confusion_matrix") or [[0]]
    cols_with = sum(1 for c in range(len(cm)) if sum(cm[r][c] for r in range(len(cm))) > 0)
    if isinstance(kappa, (int, float)) and kappa > 0.05:
        return False
    return cols_with <= 1


def _find_kfold_runs(prefix: str = "pipeline__") -> list[dict]:
    """Load every kfold train_result_*.json. When the same experiment_name has
    multiple train_result files (e.g. the run was re-launched), keep the one
    with the newest mtime — that one matches the surviving checkpoint."""
    by_exp: dict[str, tuple[float, dict]] = {}
    for path in (PROJECT_ROOT / "logs").rglob("train_result_*.json"):
        parts = path.parts
        if "logs" not in parts:
            continue
        i = parts.index("logs")
        if len(parts) - i < 6:
            continue
        experiment_name = parts[i + 4]
        if not experiment_name.startswith(prefix):
            continue
        # We only want kfold runs (experiment names include '__foldN_seed')
        if "__fold" not in experiment_name:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if (data.get("config") or {}).get("epochs"):
            try:
                if int((data.get("config") or {}).get("epochs")) < 20:
                    continue
            except (TypeError, ValueError):
                pass
        # parse fold id from experiment_name
        try:
            tag_part = experiment_name.split("__fold")[-1]
            fold_id = int(tag_part.split("_")[0])
        except (IndexError, ValueError):
            continue
        data["_fold_id"] = fold_id
        data["_result_path"] = str(path)
        mtime = path.stat().st_mtime
        # experiment_name only encodes subtrack/audio/video/encoder/fold/seed
        # -- not track or task. Same slug therefore appears under multiple
        # logs/<track>/<sub>/<task>/<exp>/... directories. Use (track, task,
        # experiment_name) as the dedup key so cross-cell collisions don't
        # erase records.
        key = (data.get("track"), data.get("task"), experiment_name)
        prev = by_exp.get(key)
        if prev is None or mtime > prev[0]:
            by_exp[key] = (mtime, data)
    return [v[1] for v in by_exp.values()]


def _combo_key(run: dict) -> tuple[str, str, str]:
    return (
        run.get("audio_feature", ""),
        run.get("video_feature", ""),
        run.get("encoder_type", ""),
    )


def _group_by_combo(runs: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        out[_combo_key(r)].append(r)
    return out


def _summarize_combo(runs: list[dict]) -> dict[str, Any]:
    f1s, kappas, cccs, recalls = [], [], [], []
    collapses = 0
    for r in runs:
        b = r.get("best_val_metrics") or {}
        f1 = b.get("f1") if isinstance(b.get("f1"), (int, float)) else b.get("macro_f1")
        if isinstance(f1, (int, float)):
            f1s.append(float(f1))
        if isinstance(b.get("kappa"), (int, float)):
            kappas.append(float(b["kappa"]))
        if isinstance(b.get("ccc"), (int, float)):
            cccs.append(float(b["ccc"]))
        cm = b.get("confusion_matrix") or [[0]]
        per_class_recall = []
        for i_, row in enumerate(cm):
            tot = sum(row)
            per_class_recall.append(row[i_] / tot if tot else 0.0)
        if per_class_recall:
            recalls.append(min(per_class_recall))
        if _is_collapse(b):
            collapses += 1

    def stats(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)

    f1_mean, f1_std = stats(f1s)
    kappa_mean, _ = stats(kappas)
    ccc_mean, ccc_std = stats(cccs)
    recall_min_mean, _ = stats(recalls)
    return {
        "n_folds": len(runs),
        "f1_mean": f1_mean, "f1_std": f1_std,
        "kappa_mean": kappa_mean,
        "ccc_mean": ccc_mean, "ccc_std": ccc_std,
        "recall_min_mean": recall_min_mean,
        "collapse_count": collapses,
    }


def _classification_score(summary: dict) -> float:
    return summary["f1_mean"] - summary["f1_std"] - 0.5 * summary["collapse_count"]


def _regression_score(summary: dict) -> float:
    return summary["ccc_mean"] - 0.5 * summary["ccc_std"]


# ---------------------------------------------------------------------------
def _resolve_checkpoint_path(run: dict) -> Path:
    ckpt = run.get("checkpoint_path") or ""
    p = Path(ckpt)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _build_test_dataset(track: str, subtrack: str, task: str, audio_feature: str,
                        video_feature: str, regression_label: str, target_t: int) -> MPDDElderDataset:
    sub = "Elder" if track == "Track1" else "Young"
    data_root = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub
    split_csv = data_root / "split_labels_test.csv"
    personality = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-trainval" / sub / "descriptions_embeddings_with_ids.npy"
    task_maps = load_task_maps(split_csv, task, regression_label or "label2")
    return MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality,
        phq_map=task_maps.get("test_phq_map"),
        target_t=target_t,
    )


@torch.no_grad()
def _forward_logits(model: TorchcatBaseline, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cls_logits, phq_pred_log1p) for the batch."""
    outputs = model(
        audio=batch["audio"].to(device) if "audio" in batch else None,
        video=batch["video"].to(device) if "video" in batch else None,
        gait=batch["gait"].to(device) if "gait" in batch else None,
        personality=batch["personality"].to(device),
        pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
    )
    if isinstance(outputs, tuple):
        logits, reg_out = outputs
    else:
        logits, reg_out = outputs, None
    return logits, reg_out


def _ensemble_infer_test(
    fold_runs: list[dict],
    track: str, subtrack: str, task: str,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Average logits across the fold ckpts and return per-id predictions."""
    if not fold_runs:
        raise ValueError("need at least one fold run for ensemble inference")

    # Build dataset once (consistent across folds since features are deterministic).
    first = fold_runs[0]
    ckpt0 = torch.load(str(_resolve_checkpoint_path(first)), map_location="cpu", weights_only=False)
    target_t = int(ckpt0.get("target_t", 128))
    regression_label = ckpt0.get("regression_label") or "label2"
    audio_feature = ckpt0["audio_feature"]
    video_feature = ckpt0["video_feature"]
    ds = _build_test_dataset(
        track, subtrack, task,
        audio_feature, video_feature, regression_label, target_t,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)

    # Sum logits and phq predictions across folds, then divide.
    sum_logits: list[torch.Tensor] = []
    sum_phq: list[torch.Tensor] = []
    ids_list: list[int] = []

    for fold_run in fold_runs:
        ckpt_path = _resolve_checkpoint_path(fold_run)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = TorchcatBaseline(**ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        fold_logits: list[torch.Tensor] = []
        fold_phq: list[torch.Tensor] = []
        fold_ids: list[int] = []
        for batch in loader:
            logits, reg_out = _forward_logits(model, batch, device)
            fold_logits.append(torch.softmax(logits, dim=-1).cpu())
            if reg_out is not None:
                fold_phq.append(reg_out.cpu().view(-1))
            fold_ids.extend(int(x) for x in batch["pid"].cpu().tolist())
        full_probs = torch.cat(fold_logits, dim=0)
        full_phq = torch.cat(fold_phq, dim=0) if fold_phq else torch.zeros(full_probs.shape[0])
        if not sum_logits:
            sum_logits.append(full_probs)
            sum_phq.append(full_phq)
            ids_list = fold_ids
        else:
            sum_logits[0] = sum_logits[0] + full_probs
            sum_phq[0] = sum_phq[0] + full_phq
        del model, ckpt
        torch.cuda.empty_cache()

    avg_probs = sum_logits[0] / len(fold_runs)
    avg_phq_log = sum_phq[0] / len(fold_runs)
    class_pred = avg_probs.argmax(dim=-1).tolist()
    phq_pred = torch.clamp(torch.expm1(avg_phq_log), min=0.0, max=27.0).tolist()

    out: dict[int, dict[str, float]] = {}
    for i, pid in enumerate(ids_list):
        out[int(pid)] = {
            "class_pred": int(class_pred[i]),
            "phq9_pred": float(phq_pred[i]),
            "prob": [float(p) for p in avg_probs[i].tolist()],
        }
    return out


# ---------------------------------------------------------------------------
def _write_submission_csv(
    path: Path,
    class_rows_by_id: dict[int, dict[str, float]],
    phq_rows_by_id: dict[int, dict[str, float]],
    task: str,
    test_ids_order: list[int],
) -> None:
    label_col = "binary_pred" if task == "binary" else "ternary_pred"
    fields = ["id", label_col, "phq9_pred"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pid in test_ids_order:
            cls = class_rows_by_id.get(pid)
            phq = phq_rows_by_id.get(pid)
            if cls is None or phq is None:
                raise KeyError(f"Missing prediction for id={pid}")
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
    ids: list[int] = []
    for row in rows:
        for k, v in row.items():
            if k is None:
                continue
            if k.strip().upper() == "ID":
                ids.append(int(str(v).strip()))
                break
    return sorted(ids)


# ---------------------------------------------------------------------------
def _pick_best_classification(runs: list[dict]) -> tuple[tuple, dict] | None:
    """Group by combo, require ≥3 folds, score and pick best."""
    grouped = _group_by_combo(runs)
    candidates: list[tuple[tuple, dict, list[dict]]] = []
    for combo, fold_runs in grouped.items():
        if len(fold_runs) < 3:
            continue
        summary = _summarize_combo(fold_runs)
        candidates.append((combo, summary, fold_runs))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -_classification_score(c[1]))
    combo, summary, fold_runs = candidates[0]
    summary["score"] = _classification_score(summary)
    return combo, {"summary": summary, "fold_runs": fold_runs}


def _pick_best_phq_source(runs_by_task: dict[str, list[dict]]) -> tuple[tuple, dict, str] | None:
    """Look across binary+ternary kfold runs and pick best by mean CCC."""
    pooled: list[tuple[str, tuple, list[dict]]] = []
    for task, runs in runs_by_task.items():
        grouped = _group_by_combo(runs)
        for combo, fold_runs in grouped.items():
            if len(fold_runs) < 3:
                continue
            pooled.append((task, combo, fold_runs))
    if not pooled:
        return None
    scored = []
    for task, combo, fold_runs in pooled:
        summary = _summarize_combo(fold_runs)
        scored.append((task, combo, summary, fold_runs, _regression_score(summary)))
    scored.sort(key=lambda x: -x[4])
    task, combo, summary, fold_runs, score = scored[0]
    summary["score"] = score
    return combo, {"summary": summary, "fold_runs": fold_runs}, task


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="sub/submissions_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[pack-v2] scanning kfold train_result_*.json")
    all_runs = _find_kfold_runs()
    print(f"[pack-v2] loaded {len(all_runs)} kfold run records")

    # Group runs by (track, subtrack, task)
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in all_runs:
        by_cell[(r.get("track"), r.get("subtrack"), r.get("task"))].append(r)

    test_ids = {t: _read_test_ids(t) for t in ALL_TRACKS}
    summary: dict[str, Any] = {"submissions": [], "selected_combos": []}

    for track in ALL_TRACKS:
        for subtrack in ALL_SUBTRACKS:
            zip_name = f"submission_{track}_{subtrack}.zip"
            zip_path = out_dir / zip_name
            workdir = out_dir / f"_workdir_{track}_{subtrack}"
            workdir.mkdir(parents=True, exist_ok=True)
            pack_record: dict[str, Any] = {
                "track": track, "subtrack": subtrack, "zip": str(zip_path),
                "tasks": {},
            }

            # ----- Pick PHQ source (best mean CCC across binary+ternary kfold)
            phq_runs = {tk: by_cell.get((track, subtrack, tk), []) for tk in PHQ_SOURCE_TASKS}
            phq_pick = _pick_best_phq_source(phq_runs)
            if phq_pick is None:
                print(f"[pack-v2] WARN no kfold runs for {track}/{subtrack} PHQ -- skip")
                continue
            phq_combo, phq_pack, phq_task = phq_pick
            ps = phq_pack["summary"]
            print(f"[pack-v2] {track}/{subtrack}/phq_source: combo={phq_combo} task={phq_task}  "
                  f"mean_ccc={ps['ccc_mean']:+.3f}±{ps['ccc_std']:.3f}  "
                  f"folds={ps['n_folds']}")
            phq_preds = _ensemble_infer_test(phq_pack["fold_runs"], track, subtrack, phq_task, device)
            pack_record["tasks"]["phq_source"] = {
                "combo": phq_combo, "source_task": phq_task,
                "summary": ps,
                "n_folds_ensembled": ps["n_folds"],
                "used_for": "phq9_pred in both binary.csv and ternary.csv",
            }
            summary["selected_combos"].append({
                "track": track, "subtrack": subtrack, "task": "phq_source",
                "combo": phq_combo, "source_task": phq_task, **ps,
            })

            # ----- Pick binary + ternary best combos
            csv_paths: list[Path] = []
            ok = True
            for task in CLASSIFICATION_TASKS:
                cell_runs = by_cell.get((track, subtrack, task), [])
                pick = _pick_best_classification(cell_runs)
                if pick is None:
                    print(f"[pack-v2] WARN no kfold runs for {track}/{subtrack}/{task} -- skip")
                    ok = False
                    continue
                combo, pack = pick
                s = pack["summary"]
                print(f"[pack-v2] {track}/{subtrack}/{task}: combo={combo}  "
                      f"score={s['score']:+.3f}  f1={s['f1_mean']:.3f}±{s['f1_std']:.3f}  "
                      f"kappa={s['kappa_mean']:+.3f}  collapse={s['collapse_count']}/{s['n_folds']}")
                cls_preds = _ensemble_infer_test(pack["fold_runs"], track, subtrack, task, device)
                csv_path = workdir / f"{task}.csv"
                _write_submission_csv(csv_path, cls_preds, phq_preds, task, test_ids[track])
                csv_paths.append(csv_path)
                pack_record["tasks"][task] = {
                    "combo": combo, "summary": s,
                    "n_folds_ensembled": s["n_folds"],
                    "csv_path": str(csv_path),
                }
                summary["selected_combos"].append({
                    "track": track, "subtrack": subtrack, "task": task,
                    "combo": combo, **s,
                })

            if not ok or len(csv_paths) != len(CLASSIFICATION_TASKS):
                print(f"[pack-v2] skipping {zip_name} (missing task)")
                continue
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for cp in csv_paths:
                    zf.write(cp, arcname=cp.name)
            size_kb = zip_path.stat().st_size / 1024.0
            print(f"[pack-v2] -> {zip_path}  ({size_kb:.1f} KB)")
            pack_record["size_kb"] = round(size_kb, 1)
            summary["submissions"].append(pack_record)

    summary_path = out_dir / "submissions_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    print(f"\n[pack-v2] {len(summary['submissions'])} submission zips -> {out_dir}")
    print(f"[pack-v2] summary -> {summary_path}")


if __name__ == "__main__":
    main()
