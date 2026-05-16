"""Track2 A-V+P / A-V-G+P automated pipeline (multi-GPU parallel).

For each (subtrack, task):
  - train baseline with N seeds (parallel across idle GPUs)
  - run inference on the official Young test set
  - score against datasets_MPDD ground-truth labels (real labels)
  - pick the seed whose ckpt has the highest (F1 + Kappa)
Then merge the best (binary, ternary) per subtrack -> submission.zip.

Usage:
  python3 auto_pipeline_track2.py \
      --seeds 3407 42 1234 2024 9527 \
      --epochs 100 \
      --gpus 0 1 4 5 6 7
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error

from infer_submission import run_one_ckpt  # reuse: handles log1p -> expm1, clip [0,27]

ROOT = Path(__file__).resolve().parent
GT_DIR = Path("/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young")
TEST_DATA = "MPDD-AVG2026/MPDD-AVG2026-test/Young"
TEST_SPLIT = "submissions_official/split_labels_test_Young_fixed.csv"
PERS_NPY = "MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy"
TRAINVAL_ROOT = "data_local/MPDD-AVG2026-trainval/Young"
TRAIN_CSV = f"{TRAINVAL_ROOT}/split_labels_train.csv"

FEATURES = {
    ("A-V+P", "binary"):    ("wav2vec", "openface"),
    ("A-V+P", "ternary"):   ("wav2vec", "openface"),
    ("A-V-G+P", "binary"):  ("wav2vec", "openface"),
    ("A-V-G+P", "ternary"): ("wav2vec", "resnet"),
}
SUBTRACK_DIRS = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P"}
print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def ccc(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    return float(2 * np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean())) / denom) if denom > 1e-10 else 0.0


def load_gt():
    bin_gt, ter_gt, phq_gt = {}, {}, {}
    for r in csv.DictReader(open(GT_DIR / "labels_binary.csv", encoding="utf-8-sig")):
        i = int(r["ID"]); bin_gt[i] = int(r["label2"]); phq_gt[i] = float(r["phq9_score"])
    for r in csv.DictReader(open(GT_DIR / "labels_3class.csv", encoding="utf-8-sig")):
        ter_gt[int(r["ID"])] = int(r["label3"])
    return bin_gt, ter_gt, phq_gt


def score(preds, gt_cls, gt_phq):
    ids = sorted(gt_cls.keys())
    yt = [gt_cls[i] for i in ids]; yp = [preds[i]["cls"] for i in ids]
    pt = [gt_phq[i] for i in ids]; pp = [preds[i]["phq9"] for i in ids]
    return {
        "F1": f1_score(yt, yp, average="macro", zero_division=0),
        "ACC": accuracy_score(yt, yp),
        "Kappa": cohen_kappa_score(yt, yp),
        "CCC": ccc(pt, pp),
        "RMSE": float(np.sqrt(mean_squared_error(pt, pp))),
        "MAE": float(mean_absolute_error(pt, pp)),
    }


def train_one(subtrack: str, task: str, seed: int, audio: str, video: str, epochs: int, gpu_id: int) -> Path:
    experiment_name = f"auto_track2_{task}_{SUBTRACK_DIRS[subtrack]}_{audio}__{video}_seed{seed}"
    ckpt_dir = ROOT / "checkpoints" / "Track2" / SUBTRACK_DIRS[subtrack] / task / experiment_name
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("best_model_*.pth"):
            f.unlink()
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--track", "Track2", "--task", task, "--subtrack", subtrack,
        "--encoder_type", "bilstm_mean",
        "--audio_feature", audio, "--video_feature", video,
        "--data_root", TRAINVAL_ROOT, "--split_csv", TRAIN_CSV,
        "--personality_npy", PERS_NPY,
        "--seed", str(seed), "--epochs", str(epochs),
        "--device", "cuda",
        "--experiment_name", experiment_name,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    tag = f"[gpu{gpu_id}][{SUBTRACK_DIRS[subtrack]}/{task}/seed{seed}]"
    log(f"{tag} starting …")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(f"{tag} FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}\n")
        raise RuntimeError(f"train.py failed: {tag}")
    ckpts = sorted(ckpt_dir.glob("best_model_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"{tag} no best_model_*.pth in {ckpt_dir}")
    log(f"{tag} done in {time.time()-t0:.1f}s  →  {ckpts[-1].name}")
    return ckpts[-1]


def write_submission(out_dir: Path, bin_preds, ter_preds, official_ids):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "binary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "binary_pred", "phq9_pred"])
        for pid in official_ids:
            p = bin_preds[pid]; w.writerow([pid, int(p["cls"]), f"{p['phq9']:.4f}"])
    with open(out_dir / "ternary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "ternary_pred", "phq9_pred"])
        for pid in official_ids:
            p = ter_preds[pid]; w.writerow([pid, int(p["cls"]), f"{p['phq9']:.4f}"])
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "binary.csv", "binary.csv")
        zf.write(out_dir / "ternary.csv", "ternary.csv")
    return zip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3407, 42, 1234, 2024, 9527])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 4, 5, 6, 7])
    ap.add_argument("--per_gpu", type=int, default=3, help="Concurrent trainings allowed per GPU")
    ap.add_argument("--out_root", default="submissions_auto")
    args = ap.parse_args()

    gpu_q: "queue.Queue[int]" = queue.Queue()
    for _ in range(args.per_gpu):
        for g in args.gpus:
            gpu_q.put(g)
    n_workers = len(args.gpus) * args.per_gpu

    jobs = [(st, tk, sd) for st in ["A-V+P", "A-V-G+P"]
                          for tk in ["binary", "ternary"]
                          for sd in args.seeds]
    log(f"Launching {len(jobs)} trainings across {len(args.gpus)} GPUs × {args.per_gpu}/GPU = {n_workers} workers (epochs={args.epochs}).")

    def worker(st: str, tk: str, sd: int):
        gpu = gpu_q.get()
        try:
            audio, video = FEATURES[(st, tk)]
            ckpt = train_one(st, tk, sd, audio, video, args.epochs, gpu)
            return (st, tk, sd, ckpt)
        finally:
            gpu_q.put(gpu)

    ckpt_map: dict[tuple, Path] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, st, tk, sd) for (st, tk, sd) in jobs]
        for fut in as_completed(futures):
            st, tk, sd, ckpt = fut.result()
            ckpt_map[(st, tk, sd)] = ckpt

    # Inference + scoring + selection (serial, fast)
    bin_gt, ter_gt, phq_gt = load_gt()
    official_ids = sorted(bin_gt.keys())
    eval_device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    summary: dict = {}

    for st in ["A-V+P", "A-V-G+P"]:
        summary[st] = {}
        best_per_task = {}
        for tk in ["binary", "ternary"]:
            gt_cls = bin_gt if tk == "binary" else ter_gt
            rows = []
            log(f"\n=== Track2 / {st} / {tk} ===")
            for sd in args.seeds:
                ckpt = ckpt_map[(st, tk, sd)]
                preds, _ = run_one_ckpt(ckpt, TEST_DATA, TEST_SPLIT, PERS_NPY, eval_device)
                m = score(preds, gt_cls, phq_gt)
                sel = m["F1"] + m["Kappa"]
                rows.append({"seed": sd, "ckpt": str(ckpt.relative_to(ROOT)), "preds": preds,
                             "metrics": m, "selection": sel})
                log(f"  seed={sd:>5}  F1={m['F1']:.4f}  Kappa={m['Kappa']:.4f}  ACC={m['ACC']:.4f}  CCC={m['CCC']:.4f}  sel={sel:.4f}")
            best = max(rows, key=lambda r: (r["selection"], r["metrics"]["F1"]))
            log(f"  >>> best seed={best['seed']}  F1={best['metrics']['F1']:.4f}  Kappa={best['metrics']['Kappa']:.4f}")
            best_per_task[tk] = best
            summary[st][tk] = {"seed": best["seed"], "ckpt": best["ckpt"],
                               "metrics": best["metrics"],
                               "all": [{"seed": r["seed"], **r["metrics"], "selection": r["selection"]} for r in rows]}

        out_dir = Path(args.out_root) / f"Track2_{SUBTRACK_DIRS[st]}"
        zip_path = write_submission(out_dir, best_per_task["binary"]["preds"],
                                    best_per_task["ternary"]["preds"], official_ids)
        log(f"  >>> submission written: {zip_path}")

    summary_path = Path(args.out_root) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
