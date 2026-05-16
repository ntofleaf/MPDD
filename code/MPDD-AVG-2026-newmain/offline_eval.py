#!/usr/bin/env python3
"""Offline evaluation of trained checkpoints against the held-out test set
with real labels (Test-MPDD-{Elder,Young}/labels_{binary,3class}.csv).

Outputs one JSON per ckpt with:
  - val metrics (from ckpt)
  - test metrics (computed here): f1, kappa, acc, ccc, rmse, mae, recall_per_class
  - per-sample probs + phq pred (log1p space) for ensembling later

Usage:
  CUDA_VISIBLE_DEVICES=0 python offline_eval.py --ckpts ckpts.txt --out_dir offline_eval
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    MPDDElderDataset, collate_batch, load_task_maps, normalize_phq_target,
    resolve_project_path, get_label_column, _strip_row_keys,
)
from metrics import classification_metrics, joint_regression_metrics
from models import TorchcatBaseline

PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_DIR = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P", "G+P": "G-P"}


def real_test_labels(track: str, task: str) -> tuple[dict[int, int], dict[int, float]]:
    """Load real test labels from Test-MPDD-{Elder,Young}/labels_{binary,3class}.csv."""
    sub = "Elder" if track == "Track1" else "Young"
    fname = "labels_binary.csv" if task == "binary" else "labels_3class.csv"
    label_col = "label2" if task == "binary" else "label3"
    path = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub / fname
    with open(path, "r", encoding="utf-8-sig") as fh:
        rows = _strip_row_keys(list(csv.DictReader(fh)))
    lmap, pmap = {}, {}
    for r in rows:
        pid = int(r["ID"])
        lmap[pid] = int(float(r[label_col]))
        for k in ("PHQ-9", "phq9_score", "PHQ9", "phq9"):
            if k in r and r[k]:
                pmap[pid] = float(r[k])
                break
    return lmap, pmap


def run_inference(model, loader, device, has_reg_head: bool):
    """Return ids, true_cls, pred_cls, probs(N,C), phq_pred(log1p), phq_true(log1p)."""
    model.eval()
    all_ids, all_true, all_pred, all_probs = [], [], [], []
    all_phq_pred, all_phq_true = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            out = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if has_reg_head:
                logits, reg_out = out
                phq_true = batch["phq9"].to(device)
                all_phq_pred.extend(reg_out.cpu().numpy().tolist())
                all_phq_true.extend(phq_true.cpu().numpy().tolist())
            else:
                logits = out
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_probs.extend(probs.tolist())
            all_pred.extend(preds.tolist())
            all_true.extend(labels.cpu().numpy().tolist())
            all_ids.extend(batch["pid"].cpu().numpy().tolist())
    return {
        "ids": all_ids,
        "y_true": all_true,
        "y_pred": all_pred,
        "probs": all_probs,
        "phq_pred_log1p": all_phq_pred,
        "phq_true_log1p": all_phq_true,
    }


def evaluate_ckpt(ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    track = ckpt["track"]
    task = ckpt["task"]
    subtrack = ckpt["subtrack"]
    sub_dir = SUBTRACK_DIR.get(subtrack, subtrack.replace("+", "-"))
    # Build dataset using REAL test labels
    real_lmap, real_pmap = real_test_labels(track, task)
    sub = "Elder" if track == "Track1" else "Young"
    data_root = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub
    # personality npy lives only in trainval (contains both train + test IDs).
    personality_npy = resolve_project_path(ckpt["personality_npy"])
    source_split = {pid: "test" for pid in real_lmap}
    ds = MPDDElderDataset(
        data_root=data_root,
        label_map=real_lmap,
        source_split_map=source_split,
        subtrack=subtrack,
        task=task,
        audio_feature=ckpt["audio_feature"],
        video_feature=ckpt["video_feature"],
        personality_npy=personality_npy,
        phq_map=real_pmap,
        target_t=int(ckpt["target_t"]),
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)
    model_kwargs = dict(ckpt["model_kwargs"])
    model = TorchcatBaseline(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    has_reg = bool(model_kwargs.get("use_regression_head", False))
    res = run_inference(model, loader, device, has_reg)
    # metrics
    y_true = np.asarray(res["y_true"], dtype=np.int64)
    y_pred = np.asarray(res["y_pred"], dtype=np.int64)
    if has_reg and res["phq_pred_log1p"]:
        phq_true = np.asarray(res["phq_true_log1p"], dtype=np.float64)
        phq_pred = np.asarray(res["phq_pred_log1p"], dtype=np.float64)
        test_metrics = joint_regression_metrics(y_true, y_pred, phq_true, phq_pred)
    else:
        test_metrics = classification_metrics(y_true, y_pred)
    # strip array fields from metrics summary
    summary = {k: v for k, v in test_metrics.items()
               if k not in {"ids", "y_true", "y_pred", "class_true", "class_pred",
                            "phq_true", "phq_pred"}}
    return {
        "ckpt": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "track": track, "task": task, "subtrack": subtrack,
        "audio_feature": ckpt["audio_feature"],
        "video_feature": ckpt["video_feature"],
        "encoder_type": ckpt["encoder_type"],
        "seed": ckpt.get("seed"),
        "experiment_name": ckpt.get("experiment_name", ""),
        "val": ckpt.get("best_val_metrics", {}),
        "metric_split": ckpt.get("metric_split", ""),
        "test": summary,
        "ids": res["ids"],
        "probs": res["probs"],
        "phq_pred_log1p": res["phq_pred_log1p"],
        "phq_true_log1p": res["phq_true_log1p"],
        "y_true": res["y_true"],
        "y_pred": res["y_pred"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="text file: one ckpt path per line")
    ap.add_argument("--out_dir", default="offline_eval/results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p.strip()) for p in Path(args.ckpts).read_text().splitlines() if p.strip()]
    device = torch.device(args.device)
    print(f"[offline_eval] {len(paths)} ckpts | device={device}", flush=True)
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        out_path = out_dir / (p.parent.name + "__" + p.stem + ".json")
        if out_path.exists():
            print(f"  [{i}/{len(paths)}] SKIP {p.name} (cached)", flush=True)
            continue
        try:
            result = evaluate_ckpt(p, device)
            with open(out_path, "w") as fh:
                json.dump(result, fh)
            tm = result["test"]
            vm = result["val"]
            print(f"  [{i}/{len(paths)}] {p.parent.parent.parent.name}/{p.parent.parent.name}/{p.parent.parent.name} "
                  f"val_f1={vm.get('f1',0):.3f} test_f1={tm.get('f1',0):.3f} "
                  f"test_kappa={tm.get('kappa',0):.3f} test_ccc={tm.get('ccc',0):.3f} "
                  f"| {p.parent.name[:60]}",
                  flush=True)
        except Exception as e:
            print(f"  [{i}/{len(paths)}] FAIL {p}: {type(e).__name__}: {e}", flush=True)
    print(f"[offline_eval] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
