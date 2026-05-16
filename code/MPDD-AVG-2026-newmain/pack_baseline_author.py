#!/usr/bin/env python3
"""Pack 6 submission zips using the original author's baseline checkpoints.

Track1 (Elder): checkpoints/Track1/<subtrack>/<task>/best_model_2026-04-30-*.pth
Track2 (Young): checkpoints/Track2/<subtrack>/<task>/baseline_Track2_*_wav2vec_resnet/best_model_*.pth

Also reports test-set metrics against real labels (offline grading).
"""
from __future__ import annotations
import argparse, csv, json, shutil, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MPDDElderDataset, collate_batch, resolve_project_path, _strip_row_keys
from metrics import classification_metrics, joint_regression_metrics
from models import TorchcatBaseline

PROJECT_ROOT = Path(__file__).resolve().parent
SUBTRACK_DIR = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P", "G+P": "G-P"}

SUBTRACKS = ["A-V+P", "A-V-G+P", "G+P"]


def find_baseline_ckpts():
    """Return {(track, subtrack, task): ckpt_path}."""
    ckpts = {}
    for sub in SUBTRACKS:
        sub_dir = SUBTRACK_DIR[sub]
        for task in ["binary", "ternary"]:
            # Track1: 2026-04-30 file directly in task dir
            t1_dir = PROJECT_ROOT / "checkpoints" / "Track1" / sub_dir / task
            t1 = sorted(t1_dir.glob("best_model_2026-04-30-*.pth"))
            if t1:
                ckpts[("Track1", sub, task)] = t1[0]

            # Track2: baseline_Track2_*_wav2vec_resnet/best_model_*.pth
            t2_dir = PROJECT_ROOT / "checkpoints" / "Track2" / sub_dir / task
            cands = list(t2_dir.glob("baseline_Track2_*_wav2vec_resnet/best_model_*.pth"))
            if cands:
                ckpts[("Track2", sub, task)] = cands[0]
    return ckpts


def real_test_labels(track, task):
    sub = "Elder" if track == "Track1" else "Young"
    fname = "labels_binary.csv" if task == "binary" else "labels_3class.csv"
    label_col = "label2" if task == "binary" else "label3"
    path = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub / fname
    with open(path, encoding="utf-8-sig") as fh:
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


def run_inference(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    track = ckpt["track"]; task = ckpt["task"]; subtrack = ckpt["subtrack"]
    sub = "Elder" if track == "Track1" else "Young"
    data_root = PROJECT_ROOT / "MPDD-AVG2026" / "MPDD-AVG2026-test" / sub
    real_lmap, real_pmap = real_test_labels(track, task)
    source_split = {pid: "test" for pid in real_lmap}
    ds = MPDDElderDataset(
        data_root=data_root,
        label_map=real_lmap,
        source_split_map=source_split,
        subtrack=subtrack,
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
    ids, y_true, y_pred, probs, phq_pred_log, phq_true_log = [], [], [], [], [], []
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
            if has_reg:
                logits, reg_out = out
                phq_pred_log.extend(reg_out.cpu().numpy().tolist())
                phq_true_log.extend(batch["phq9"].cpu().numpy().tolist())
            else:
                logits = out
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.extend(p.tolist())
            y_pred.extend(p.argmax(-1).tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            ids.extend(batch["pid"].cpu().numpy().tolist())
    # metrics
    y_t = np.asarray(y_true); y_p = np.asarray(y_pred)
    if has_reg:
        pt = np.asarray(phq_true_log); pp = np.asarray(phq_pred_log)
        m = joint_regression_metrics(y_t, y_p, pt, pp)
    else:
        m = classification_metrics(y_t, y_p)
    summary = {k: v for k, v in m.items()
               if k not in {"ids","y_true","y_pred","class_true","class_pred","phq_true","phq_pred"}}
    return {
        "ckpt": str(ckpt_path), "track": track, "task": task, "subtrack": subtrack,
        "audio_feature": ckpt["audio_feature"], "video_feature": ckpt["video_feature"],
        "ids": ids, "y_pred": y_pred, "probs": probs,
        "phq_pred_log": phq_pred_log, "metrics": summary,
        "val": ckpt.get("best_val_metrics", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="sub/baseline_author")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_root = PROJECT_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpts = find_baseline_ckpts()
    print(f"Found {len(ckpts)} baseline ckpts. Expected 12 (2 tracks * 3 subs * 2 tasks).")
    for k, p in sorted(ckpts.items()):
        print(f"  {k}: {p.relative_to(PROJECT_ROOT)}")

    # Run inference, collect results per (track, subtrack, task)
    results = {}
    for key, p in sorted(ckpts.items()):
        print(f"\n[infer] {key}")
        r = run_inference(p, device)
        results[key] = r
        m = r["metrics"]; v = r["val"]
        print(f"  val_f1={v.get('f1',0):.3f} val_ccc={v.get('ccc',0):.3f}  "
              f"|  test_f1={m.get('f1',0):.3f} test_kappa={m.get('kappa',0):+.3f} "
              f"test_ccc={m.get('ccc',0):+.3f} test_acc={m.get('acc',0):.3f}")

    # Build 6 submission zips: per (track, subtrack)
    summary_rows = []
    for track in ["Track1", "Track2"]:
        for sub in SUBTRACKS:
            bin_r = results.get((track, sub, "binary"))
            ter_r = results.get((track, sub, "ternary"))
            if not bin_r or not ter_r:
                print(f"  SKIP {track}/{sub}: missing ckpt")
                continue
            # PHQ source: pick the one with higher val_ccc
            bin_val_ccc = bin_r["val"].get("ccc", -99)
            ter_val_ccc = ter_r["val"].get("ccc", -99)
            phq_src = bin_r if bin_val_ccc >= ter_val_ccc else ter_r
            phq_src_label = f'{phq_src["task"]}(val_ccc={phq_src["val"].get("ccc",0):.3f})'
            # Build id -> phq_pred (expm1 of log1p prediction)
            phq_map = {}
            for i, pid in enumerate(phq_src["ids"]):
                phq_map[int(pid)] = float(np.clip(np.expm1(phq_src["phq_pred_log"][i]), 0.0, 27.0))

            workdir = out_root / f"_workdir_{track}_{sub}"
            workdir.mkdir(parents=True, exist_ok=True)
            for r, csv_name, label_key in [
                (bin_r, "binary.csv", "binary_pred"),
                (ter_r, "ternary.csv", "ternary_pred"),
            ]:
                csv_path = workdir / csv_name
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["id", label_key, "phq9_pred"])
                    for i, pid in enumerate(r["ids"]):
                        w.writerow([int(pid), int(r["y_pred"][i]), f'{phq_map.get(int(pid), 0.0):.4f}'])
            # zip
            zip_path = out_root / f"submission_{track}_{sub}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(workdir / "binary.csv", arcname="binary.csv")
                zf.write(workdir / "ternary.csv", arcname="ternary.csv")
            # offline grade for this cell
            bin_test_f1 = bin_r["metrics"].get("f1", 0)
            ter_test_f1 = ter_r["metrics"].get("f1", 0)
            bin_test_kappa = bin_r["metrics"].get("kappa", 0)
            ter_test_kappa = ter_r["metrics"].get("kappa", 0)
            # Offline grading needs real PHQ test labels (already class-conditional in dataset)
            # phq_src CCC is already computed in metrics
            phq_ccc = phq_src["metrics"].get("ccc", 0)
            # cls_f1 = mean(bin_f1, ter_f1); cls_kappa = mean(...)
            cls_f1 = (bin_test_f1 + ter_test_f1) / 2
            cls_kappa = (bin_test_kappa + ter_test_kappa) / 2
            score = (cls_f1 + phq_ccc + cls_kappa) / 3
            row = {
                "track": track, "subtrack": sub,
                "bin_val_f1": bin_r["val"].get("f1",0),
                "ter_val_f1": ter_r["val"].get("f1",0),
                "bin_test_f1": bin_test_f1, "ter_test_f1": ter_test_f1,
                "bin_test_kappa": bin_test_kappa, "ter_test_kappa": ter_test_kappa,
                "phq_source": phq_src_label, "phq_test_ccc": phq_ccc,
                "predicted_score": score,
                "zip": str(zip_path.relative_to(PROJECT_ROOT)),
            }
            summary_rows.append(row)
            print(f"\n  {track}/{sub}: predicted_score={score:+.4f}  "
                  f"(bin_f1={bin_test_f1:.3f}, ter_f1={ter_test_f1:.3f}, "
                  f"kappa~{cls_kappa:+.3f}, phq_ccc={phq_ccc:+.3f} via {phq_src_label})")
            print(f"  -> {zip_path.relative_to(PROJECT_ROOT)}")

    # Write summary
    with open(out_root / "submissions_summary.json", "w") as fh:
        json.dump({"submissions": summary_rows}, fh, indent=2)
    print("\n=== SUMMARY ===")
    for r in summary_rows:
        print(f"  {r['track']}/{r['subtrack']}: predicted_score={r['predicted_score']:+.4f}")
    avg_score = sum(r["predicted_score"] for r in summary_rows) / max(1, len(summary_rows))
    print(f"\n  Mean predicted Score across {len(summary_rows)} cells: {avg_score:+.4f}")
    print(f"\nZips:")
    for z in sorted(out_root.glob("*.zip")):
        print(f"  {z.relative_to(PROJECT_ROOT)}  ({z.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
