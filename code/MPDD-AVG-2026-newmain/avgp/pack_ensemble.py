#!/usr/bin/env python3
"""Ensemble inference + submission packing for Elder A-V-G+P.

Loads N seed-trained ckpts per task (binary, ternary), runs test inference,
averages softmax probabilities across seeds for classification, and averages
log1p PHQ predictions across seeds for regression. Source PHQ from the
task (binary vs ternary) with higher mean val_CCC across seeds.

Output: sub/<out_subdir>/submission_Track1_A-V-G+P.zip with id,binary_pred,
phq9_pred / id,ternary_pred,phq9_pred CSVs (no BOM, lowercase headers).

Usage:
  python avgp/pack_ensemble.py \
    --binary_ckpts ckpts_bin.txt \
    --ternary_ckpts ckpts_ter.txt \
    --out_subdir avgp_repro_v1
"""
from __future__ import annotations
import argparse, csv, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MPDDElderDataset, collate_batch, resolve_project_path, _strip_row_keys
from models import TorchcatBaseline


def real_test_labels(track: str, task: str):
    """Load real test labels from labels_{binary,3class}.csv (true class labels;
    PHQ column is class-conditional placeholder — only used for ID list)."""
    sub = "Elder" if track == "Track1" else "Young"
    fname = "labels_binary.csv" if task == "binary" else "labels_3class.csv"
    col = "label2" if task == "binary" else "label3"
    path = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub / fname
    with open(path, encoding="utf-8-sig") as fh:
        rows = _strip_row_keys(list(csv.DictReader(fh)))
    lmap, pmap = {}, {}
    for r in rows:
        pid = int(r["ID"])
        lmap[pid] = int(float(r[col]))
        for k in ("PHQ-9", "phq9_score", "PHQ9", "phq9"):
            if k in r and r[k]:
                pmap[pid] = float(r[k]); break
    return lmap, pmap


def run_one_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    track = ckpt["track"]; task = ckpt["task"]; sub = ckpt["subtrack"]
    sub_label = "Elder" if track == "Track1" else "Young"
    data_root = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub_label
    real_lmap, real_pmap = real_test_labels(track, task)
    source_split = {pid: "test" for pid in real_lmap}
    ds = MPDDElderDataset(
        data_root=data_root,
        label_map=real_lmap,
        source_split_map=source_split,
        subtrack=sub,
        task=task,
        audio_feature=ckpt["audio_feature"],
        video_feature=ckpt["video_feature"],
        personality_npy=resolve_project_path(ckpt["personality_npy"]),
        phq_map=real_pmap,
        target_t=int(ckpt["target_t"]),
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)
    model = TorchcatBaseline(**dict(ckpt["model_kwargs"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    has_reg = bool(ckpt["model_kwargs"].get("use_regression_head", False))
    model.eval()
    ids, probs, phq_log = [], [], []
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
                logits, reg = out
                phq_log.extend(reg.cpu().numpy().tolist())
            else:
                logits = out
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.extend(p.tolist())
            ids.extend(batch["pid"].cpu().numpy().tolist())
    return {
        "track": track, "task": task, "subtrack": sub,
        "ids": ids, "probs": np.array(probs), "phq_log": np.array(phq_log) if phq_log else None,
        "val_metrics": ckpt.get("best_val_metrics", {}),
    }


def ensemble(results: list[dict], phq_strategy: str = "mean") -> dict:
    """Average softmax probs and log1p PHQ across all results.
    All results must share same `ids` order.
    """
    assert results, "no ckpts"
    ids = results[0]["ids"]
    for r in results:
        if r["ids"] != ids:
            raise ValueError("ID order mismatch across ckpts")
    probs = np.mean([r["probs"] for r in results], axis=0)
    cls_pred = probs.argmax(-1)
    phqs = [r["phq_log"] for r in results if r["phq_log"] is not None]
    if phqs:
        if phq_strategy == "median":
            phq_log = np.median(phqs, axis=0)
        else:
            phq_log = np.mean(phqs, axis=0)
        phq_orig = np.clip(np.expm1(phq_log), 0.0, 27.0)
    else:
        phq_log = None
        phq_orig = None
    return {
        "ids": ids,
        "probs": probs,
        "cls_pred": cls_pred,
        "phq_log": phq_log,
        "phq_orig": phq_orig,
        "n_ckpts": len(results),
    }


def load_ckpt_list(path: Path) -> list[Path]:
    paths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = Path(line)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"WARN: missing {p}", file=sys.stderr)
            continue
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_ckpts", required=True, help="file with binary ckpt paths (one per line)")
    ap.add_argument("--ternary_ckpts", required=True, help="file with ternary ckpt paths")
    ap.add_argument("--out_subdir", required=True, help="output dir under sub/")
    ap.add_argument("--phq_strategy", default="mean", choices=["mean", "median"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    bin_paths = load_ckpt_list(Path(args.binary_ckpts))
    ter_paths = load_ckpt_list(Path(args.ternary_ckpts))
    print(f"[ensemble] binary ckpts: {len(bin_paths)}, ternary ckpts: {len(ter_paths)}", flush=True)

    bin_results = [run_one_ckpt(p, device) for p in bin_paths]
    ter_results = [run_one_ckpt(p, device) for p in ter_paths]
    print(f"[ensemble] all inferences done", flush=True)

    bin_ens = ensemble(bin_results, phq_strategy=args.phq_strategy)
    ter_ens = ensemble(ter_results, phq_strategy=args.phq_strategy)

    # PHQ source: pick task with higher MEAN val_CCC across its seeds.
    bin_val_ccc_mean = np.mean([r["val_metrics"].get("ccc", 0) for r in bin_results])
    ter_val_ccc_mean = np.mean([r["val_metrics"].get("ccc", 0) for r in ter_results])
    if bin_val_ccc_mean >= ter_val_ccc_mean:
        phq_src = bin_ens; phq_src_label = f"binary (mean val_ccc={bin_val_ccc_mean:.3f})"
    else:
        phq_src = ter_ens; phq_src_label = f"ternary (mean val_ccc={ter_val_ccc_mean:.3f})"

    out_root = PROJECT_ROOT / "sub" / args.out_subdir
    workdir = out_root / "_workdir_Track1_A-V-G+P"
    workdir.mkdir(parents=True, exist_ok=True)

    # Build PHQ map by ID (in case ids order differs)
    phq_map = {int(pid): float(phq_src["phq_orig"][i])
               for i, pid in enumerate(phq_src["ids"])} if phq_src["phq_orig"] is not None else {}

    for ens, csv_name, label_key in [(bin_ens, "binary.csv", "binary_pred"),
                                      (ter_ens, "ternary.csv", "ternary_pred")]:
        csv_path = workdir / csv_name
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", label_key, "phq9_pred"])
            for i, pid in enumerate(ens["ids"]):
                w.writerow([int(pid), int(ens["cls_pred"][i]),
                            f"{phq_map.get(int(pid), 0.0):.4f}"])

    zip_path = out_root / "submission_Track1_A-V-G+P.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(workdir / "binary.csv", arcname="binary.csv")
        zf.write(workdir / "ternary.csv", arcname="ternary.csv")

    # Summary
    summary = {
        "binary": {
            "n_ckpts": len(bin_results),
            "val_metrics_per_ckpt": [r["val_metrics"] for r in bin_results],
            "mean_val_f1": float(np.mean([r["val_metrics"].get("f1", 0) for r in bin_results])),
            "mean_val_ccc": float(bin_val_ccc_mean),
        },
        "ternary": {
            "n_ckpts": len(ter_results),
            "val_metrics_per_ckpt": [r["val_metrics"] for r in ter_results],
            "mean_val_f1": float(np.mean([r["val_metrics"].get("f1", 0) for r in ter_results])),
            "mean_val_ccc": float(ter_val_ccc_mean),
        },
        "phq_source": phq_src_label,
        "phq_strategy": args.phq_strategy,
        "phq_pred_stats": {
            "mean": float(np.mean(phq_src["phq_orig"])) if phq_src["phq_orig"] is not None else 0,
            "std": float(np.std(phq_src["phq_orig"])) if phq_src["phq_orig"] is not None else 0,
            "min": float(np.min(phq_src["phq_orig"])) if phq_src["phq_orig"] is not None else 0,
            "max": float(np.max(phq_src["phq_orig"])) if phq_src["phq_orig"] is not None else 0,
        },
        "binary_pred_dist": {int(c): int((bin_ens["cls_pred"] == c).sum()) for c in range(2)},
        "ternary_pred_dist": {int(c): int((ter_ens["cls_pred"] == c).sum()) for c in range(3)},
        "zip": str(zip_path.relative_to(PROJECT_ROOT)),
    }
    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== Ensemble done ===")
    print(f"  binary  : {len(bin_results)} ckpts, mean val_f1={summary['binary']['mean_val_f1']:.3f}, "
          f"mean val_ccc={summary['binary']['mean_val_ccc']:+.3f}")
    print(f"  ternary : {len(ter_results)} ckpts, mean val_f1={summary['ternary']['mean_val_f1']:.3f}, "
          f"mean val_ccc={summary['ternary']['mean_val_ccc']:+.3f}")
    print(f"  PHQ source: {phq_src_label}")
    print(f"  PHQ pred:  mean={summary['phq_pred_stats']['mean']:.2f} "
          f"std={summary['phq_pred_stats']['std']:.2f} "
          f"range=[{summary['phq_pred_stats']['min']:.2f}, {summary['phq_pred_stats']['max']:.2f}]")
    print(f"  Binary  pred dist: {summary['binary_pred_dist']}")
    print(f"  Ternary pred dist: {summary['ternary_pred_dist']}")
    print(f"  Zip: {zip_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
