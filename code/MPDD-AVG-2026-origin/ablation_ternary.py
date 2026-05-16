"""Hyperparameter ablation for ternary task (classification only).

4 configs x 3 seeds = 12 trainings per subtrack, parallel across idle GPUs.
After training, evaluate each ckpt on the test set and only report
classification metrics (F1 / ACC / Kappa).

Usage:
  python3 ablation_ternary.py --subtrack A-V+P
  python3 ablation_ternary.py --subtrack A-V-G+P
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from infer_submission import run_one_ckpt

ROOT = Path(__file__).resolve().parent
GT_DIR = Path("/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young")
TEST_DATA = "MPDD-AVG2026/MPDD-AVG2026-test/Young"
TEST_SPLIT = "submissions_official/split_labels_test_Young_fixed.csv"
PERS_NPY = "MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy"
TRAINVAL_ROOT = "data_local/MPDD-AVG2026-trainval/Young"
TRAIN_CSV = f"{TRAINVAL_ROOT}/split_labels_train.csv"

FEATURES = {"A-V+P": ("wav2vec", "openface"), "A-V-G+P": ("wav2vec", "resnet")}
SUBTRACK_DIRS = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P"}

CONFIGS = [
    {"tag": "baseline",  "hidden_dim": 64,  "dropout": 0.5},
    {"tag": "h128",      "hidden_dim": 128, "dropout": 0.5},
    {"tag": "drop03",    "hidden_dim": 64,  "dropout": 0.3},
    {"tag": "h128+drop03","hidden_dim": 128,"dropout": 0.3},
]
print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def load_ter_gt() -> dict[int, int]:
    return {int(r["ID"]): int(r["label3"]) for r in csv.DictReader(open(GT_DIR / "labels_3class.csv", encoding="utf-8-sig"))}


def cls_score(preds, gt_cls):
    ids = sorted(gt_cls.keys())
    yt = [gt_cls[i] for i in ids]; yp = [preds[i]["cls"] for i in ids]
    return {
        "F1": f1_score(yt, yp, average="macro", zero_division=0),
        "ACC": accuracy_score(yt, yp),
        "Kappa": cohen_kappa_score(yt, yp),
    }


def train_one(subtrack: str, cfg: dict, seed: int, gpu_id: int) -> Path:
    audio, video = FEATURES[subtrack]
    experiment_name = f"ablation_ter_{SUBTRACK_DIRS[subtrack]}_{cfg['tag']}_seed{seed}"
    ckpt_dir = ROOT / "checkpoints" / "Track2" / SUBTRACK_DIRS[subtrack] / "ternary" / experiment_name
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("best_model_*.pth"):
            f.unlink()
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--track", "Track2", "--task", "ternary", "--subtrack", subtrack,
        "--encoder_type", "bilstm_mean",
        "--audio_feature", audio, "--video_feature", video,
        "--data_root", TRAINVAL_ROOT, "--split_csv", TRAIN_CSV,
        "--personality_npy", PERS_NPY,
        "--seed", str(seed), "--epochs", "100",
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--dropout", str(cfg["dropout"]),
        "--device", "cuda",
        "--experiment_name", experiment_name,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    tag = f"[gpu{gpu_id}][{cfg['tag']}/seed{seed}]"
    log(f"{tag} starting …")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(f"{tag} FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}\n")
        raise RuntimeError(f"train.py failed: {tag}")
    ckpts = sorted(ckpt_dir.glob("best_model_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"{tag} no best_model_*.pth")
    log(f"{tag} done in {time.time()-t0:.1f}s")
    return ckpts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtrack", required=True, choices=["A-V+P", "A-V-G+P"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3407, 42, 1234])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 4, 5, 6, 7])
    ap.add_argument("--per_gpu", type=int, default=3)
    args = ap.parse_args()

    gpu_q: "queue.Queue[int]" = queue.Queue()
    for _ in range(args.per_gpu):
        for g in args.gpus:
            gpu_q.put(g)
    n_workers = len(args.gpus) * args.per_gpu

    jobs = [(cfg, sd) for cfg in CONFIGS for sd in args.seeds]
    log(f"\n=========================================================")
    log(f"  Ablation: Track2 / {args.subtrack} / ternary")
    log(f"  {len(CONFIGS)} configs × {len(args.seeds)} seeds = {len(jobs)} trainings")
    log(f"  GPUs: {args.gpus} × {args.per_gpu}/GPU = {n_workers} slots")
    log(f"=========================================================\n")

    def worker(cfg, sd):
        gpu = gpu_q.get()
        try:
            return (cfg, sd, train_one(args.subtrack, cfg, sd, gpu))
        finally:
            gpu_q.put(gpu)

    ckpt_map: dict[tuple, Path] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, cfg, sd) for (cfg, sd) in jobs]
        for fut in as_completed(futures):
            cfg, sd, ckpt = fut.result()
            ckpt_map[(cfg["tag"], sd)] = ckpt

    # Evaluation on test set (classification only)
    log("\n=== Evaluating on test set (n=22) ===\n")
    eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt_cls = load_ter_gt()
    rows: list[dict] = []
    for cfg in CONFIGS:
        for sd in args.seeds:
            ckpt = ckpt_map[(cfg["tag"], sd)]
            preds, _ = run_one_ckpt(ckpt, TEST_DATA, TEST_SPLIT, PERS_NPY, eval_device)
            m = cls_score(preds, gt_cls)
            rows.append({"config": cfg["tag"], "hidden_dim": cfg["hidden_dim"], "dropout": cfg["dropout"],
                         "seed": sd, **m, "ckpt": str(ckpt.relative_to(ROOT))})

    # Print per-seed table
    log(f"{'Config':<14} {'hd':>4} {'do':>5} {'seed':>5}    {'F1':>8} {'ACC':>8} {'Kappa':>8}")
    log("-" * 70)
    for r in rows:
        log(f"{r['config']:<14} {r['hidden_dim']:>4} {r['dropout']:>5.2f} {r['seed']:>5}    {r['F1']:>8.4f} {r['ACC']:>8.4f} {r['Kappa']:>8.4f}")

    # Aggregate per config (mean ± std across seeds)
    log("\n=== Per-config aggregate (mean ± std across 3 seeds) ===\n")
    log(f"{'Config':<14} {'hd':>4} {'do':>5}    {'F1':>16} {'ACC':>16} {'Kappa':>16}")
    log("-" * 80)
    summary: dict = {"subtrack": args.subtrack, "seeds": args.seeds, "rows": rows, "agg": {}}
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg["tag"]]
        f1 = np.array([r["F1"] for r in sub]); acc = np.array([r["ACC"] for r in sub]); kap = np.array([r["Kappa"] for r in sub])
        log(f"{cfg['tag']:<14} {cfg['hidden_dim']:>4} {cfg['dropout']:>5.2f}    {f1.mean():>7.4f}±{f1.std():>5.3f}   {acc.mean():>7.4f}±{acc.std():>5.3f}   {kap.mean():>7.4f}±{kap.std():>5.3f}")
        summary["agg"][cfg["tag"]] = {"F1_mean": float(f1.mean()), "F1_std": float(f1.std()),
                                       "ACC_mean": float(acc.mean()), "ACC_std": float(acc.std()),
                                       "Kappa_mean": float(kap.mean()), "Kappa_std": float(kap.std())}

    out_path = Path("ablation_logs") / f"ablation_ternary_{SUBTRACK_DIRS[args.subtrack]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"\n→ Saved: {out_path}")


if __name__ == "__main__":
    main()
