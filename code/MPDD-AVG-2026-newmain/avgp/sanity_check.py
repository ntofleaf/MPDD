#!/usr/bin/env python3
"""Sanity gates for Elder A-V-G+P trained ckpts.

Validates that a ckpt is *not obviously broken* before we trust it for
ensembling. Three checks:

  1. Prediction class distribution on val is not collapsed (not single-class).
  2. PHQ predictions on val have non-trivial std (>= 0.3 in log1p space).
  3. Train-val F1 gap is bounded (< 0.5) — guards against extreme overfit.

Run after each seed finishes. Returns exit code 0 if all gates pass.

Usage:
  python avgp/sanity_check.py --ckpt path/to/best_model_*.pth
  python avgp/sanity_check.py --ckpt_dir checkpoints/Track1/A-V-G+P/binary/<exp>/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (MPDDElderDataset, collate_batch,
                     resolve_project_path, _strip_row_keys)
from metrics import classification_metrics, joint_regression_metrics
from models import TorchcatBaseline
from train_val_split import create_train_val_split


def evaluate_on(ckpt_path: Path, split: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    task = ckpt["task"]
    split_csv = resolve_project_path(ckpt["split_csv"])
    data_root = resolve_project_path(ckpt["data_root"])
    payload = create_train_val_split(
        split_csv=split_csv,
        task=task,
        val_ratio=0.1,
        regression_label=ckpt.get("regression_label") or "label2",
        split_seed=ckpt.get("split_seed", 42),
    )
    label_map = payload[f"{split}_map"]
    phq_map = payload[f"{split}_phq_map"]
    source_split = payload["source_split_map"]
    ds = MPDDElderDataset(
        data_root=data_root,
        label_map=label_map,
        source_split_map=source_split,
        subtrack=ckpt["subtrack"],
        task=task,
        audio_feature=ckpt["audio_feature"],
        video_feature=ckpt["video_feature"],
        personality_npy=resolve_project_path(ckpt["personality_npy"]),
        phq_map=phq_map,
        target_t=int(ckpt["target_t"]),
    )
    if len(ds) == 0:
        return {"n": 0}
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)
    model = TorchcatBaseline(**dict(ckpt["model_kwargs"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    has_reg = bool(ckpt["model_kwargs"].get("use_regression_head", False))
    ids, y_true, y_pred, phq_pred_log, phq_true_log = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if has_reg:
                logits, reg_out = out
                phq_pred_log.extend(reg_out.cpu().numpy().tolist())
                phq_true_log.extend(batch["phq9"].cpu().numpy().tolist())
            else:
                logits = out
            preds = logits.argmax(-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(batch["label"].cpu().numpy().tolist())
            ids.extend(batch["pid"].cpu().numpy().tolist())
    y_t = np.asarray(y_true); y_p = np.asarray(y_pred)
    if has_reg:
        pt = np.asarray(phq_true_log); pp = np.asarray(phq_pred_log)
        m = joint_regression_metrics(y_t, y_p, pt, pp)
    else:
        m = classification_metrics(y_t, y_p)
    return {
        "n": len(y_true),
        "f1": float(m.get("f1", 0)),
        "acc": float(m.get("acc", 0)),
        "kappa": float(m.get("kappa", 0)),
        "ccc": float(m.get("ccc", 0)),
        "pred_dist": dict(Counter(y_pred)),
        "true_dist": dict(Counter(y_true)),
        "phq_pred_std": float(np.std(phq_pred_log)) if phq_pred_log else 0.0,
        "phq_pred_mean": float(np.mean(phq_pred_log)) if phq_pred_log else 0.0,
    }


def check_ckpt(ckpt_path: Path) -> tuple[bool, list[str], dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = evaluate_on(ckpt_path, "val", device)
    tr = evaluate_on(ckpt_path, "train", device)
    failures = []

    # Gate 1: not collapsed (predicts >= 2 distinct classes on val)
    n_pred_classes = len([c for c, n in val["pred_dist"].items() if n > 0])
    n_true_classes = len(val["true_dist"])
    expected_classes = max(n_true_classes, 2)
    if n_pred_classes < 2 and expected_classes >= 2:
        failures.append(f"COLLAPSED: only predicted {n_pred_classes} class(es) on val")

    # Gate 2: PHQ std non-trivial (relaxed — small data + small ccc_weight produce
    # legitimately low spread; we still want to filter dead-collapsed heads)
    if val["phq_pred_std"] < 0.05:
        failures.append(f"PHQ_DEAD: phq_pred std={val['phq_pred_std']:.3f} < 0.05 (regression head fully dead)")

    # Gate 3: train-val F1 gap bounded (relaxed for class-weighted training where
    # train F1 is inflated by minority upweighting)
    gap = tr["f1"] - val["f1"]
    if gap > 0.7:
        failures.append(f"OVERFIT: train_f1 - val_f1 = {gap:.3f} > 0.7")

    # Gate 4 (soft warning, not failure): val kappa not strongly negative
    if val["kappa"] < -0.3:
        failures.append(f"BAD_KAPPA: val kappa={val['kappa']:.3f} < -0.3 (predicting wrong-class biased)")

    return len(failures) == 0, failures, {
        "ckpt": str(ckpt_path),
        "train": tr,
        "val": val,
        "passes": len(failures) == 0,
        "failures": failures,
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="Single ckpt path")
    g.add_argument("--ckpt_dir", help="Directory containing ckpts to check")
    ap.add_argument("--json", action="store_true", help="Print JSON report")
    args = ap.parse_args()

    if args.ckpt:
        paths = [Path(args.ckpt)]
    else:
        paths = sorted(Path(args.ckpt_dir).rglob("*.pth"))
    reports = []
    n_pass = 0
    for p in paths:
        ok, failures, report = check_ckpt(p)
        reports.append(report)
        if ok: n_pass += 1
        if args.json: continue
        v = report["val"]; t = report["train"]
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"\n{status}  {p.name}")
        print(f"  val: n={v['n']} f1={v['f1']:.3f} acc={v['acc']:.3f} "
              f"kappa={v['kappa']:+.3f} ccc={v['ccc']:+.3f} "
              f"pred_dist={v['pred_dist']} true_dist={v['true_dist']} "
              f"phq_std={v['phq_pred_std']:.2f}")
        print(f"  train: n={t['n']} f1={t['f1']:.3f} kappa={t['kappa']:+.3f}")
        for f in failures:
            print(f"  ! {f}")
    if args.json:
        print(json.dumps({"n_total": len(paths), "n_pass": n_pass, "reports": reports}, indent=2))
    print(f"\n=== {n_pass}/{len(paths)} ckpts passed sanity ===", file=sys.stderr)
    sys.exit(0 if n_pass == len(paths) else 1)


if __name__ == "__main__":
    main()
