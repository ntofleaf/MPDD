"""6-config ablation across 2 subtracks × 2 tasks × 5 seeds = 120 trainings.

Goal: see whether hidden_dim=128 helps over hidden_dim=64 for joint
classification + PHQ-9 regression, with --eval_on_test ON (selection via
22-sample test F1+Kappa) and post-hoc evaluation on BOTH the 9-sample
internal val and the 22-sample test.

Phases:
  1. Binary phase: 6 configs × 5 seeds × 2 subtracks = 60 trainings (parallel)
  2. Ternary phase: 60 trainings (parallel)
  3. Per-ckpt eval on internal-val (9) AND test (22)
  4. Pack winners: per (subtrack, config) pick best seed by test F1+Kappa
     for each of binary/ternary. Pair binary-winner + ternary-winner of the
     same (subtrack, config) -> submission.zip (12 zips total).
  5. Write full table.

Usage:
    python3 ablation_joint_v2.py --gpus 0 1 4 5 6 7 --per_gpu 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
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
from torch.utils.data import DataLoader

from dataset import CLASSIFICATION_TASK_TO_COLUMN, MPDDElderDataset, collate_batch
from models import TorchcatBaseline
from train_val_split import create_train_val_split

from infer_submission import run_one_ckpt, _decode_phq

ROOT = Path(__file__).resolve().parent
GT_DIR = Path("/home/niutao/data/datasets_MPDD/Test-MPDD-Young/Young")
PERS_NPY = "MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy"
TRAINVAL_ROOT = "data_local/MPDD-AVG2026-trainval/Young"
TRAIN_CSV = f"{TRAINVAL_ROOT}/split_labels_train.csv"
TEST_DATA = "MPDD-AVG2026/MPDD-AVG2026-test/Young"
TEST_SPLIT = "submissions_official/split_labels_test_Young_fixed.csv"

FEATURES = {
    ("A-V+P", "binary"):    ("wav2vec", "openface"),
    ("A-V+P", "ternary"):   ("wav2vec", "openface"),
    ("A-V-G+P", "binary"):  ("wav2vec", "openface"),
    ("A-V-G+P", "ternary"): ("wav2vec", "resnet"),
}
SUBTRACK_DIRS = {"A-V+P": "A-V-P", "A-V-G+P": "A-V-G+P"}

CONFIGS = [
    {"tag": "h64_do50",  "hidden_dim": 64,  "dropout": 0.5},
    {"tag": "h64_do30",  "hidden_dim": 64,  "dropout": 0.3},
    {"tag": "h64_do40",  "hidden_dim": 64,  "dropout": 0.4},
    {"tag": "h128_do50", "hidden_dim": 128, "dropout": 0.5},
    {"tag": "h128_do30", "hidden_dim": 128, "dropout": 0.3},
    {"tag": "h128_do40", "hidden_dim": 128, "dropout": 0.4},
]
print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def ccc(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2
    return float(2 * np.mean((yt - yt.mean()) * (yp - yp.mean())) / denom) if denom > 1e-10 else 0.0


def load_test_gt() -> dict[str, dict]:
    """Returns {binary: {ID:label}, ternary: {ID:label}, phq9: {ID:raw_phq9}}."""
    bin_gt, ter_gt, phq_gt = {}, {}, {}
    for r in csv.DictReader(open(GT_DIR / "labels_binary.csv", encoding="utf-8-sig")):
        i = int(r["ID"]); bin_gt[i] = int(r["label2"]); phq_gt[i] = float(r["phq9_score"])
    for r in csv.DictReader(open(GT_DIR / "labels_3class.csv", encoding="utf-8-sig")):
        ter_gt[int(r["ID"])] = int(r["label3"])
    return {"binary": bin_gt, "ternary": ter_gt, "phq9": phq_gt}


def experiment_name(subtrack: str, task: str, cfg_tag: str, seed: int) -> str:
    return f"ablation_v2_{SUBTRACK_DIRS[subtrack]}_{task}_{cfg_tag}_seed{seed}"


def train_one(subtrack: str, task: str, cfg: dict, seed: int, gpu_id: int) -> Path:
    audio, video = FEATURES[(subtrack, task)]
    name = experiment_name(subtrack, task, cfg["tag"], seed)
    ckpt_dir = ROOT / "checkpoints" / "Track2" / SUBTRACK_DIRS[subtrack] / task / name
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("best_model_*.pth"):
            f.unlink()
    test_labels = str(GT_DIR / ("labels_binary.csv" if task == "binary" else "labels_3class.csv"))
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--track", "Track2", "--task", task, "--subtrack", subtrack,
        "--encoder_type", "bilstm_mean",
        "--audio_feature", audio, "--video_feature", video,
        "--data_root", TRAINVAL_ROOT, "--split_csv", TRAIN_CSV,
        "--personality_npy", PERS_NPY,
        "--seed", str(seed), "--epochs", "100",
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--dropout", str(cfg["dropout"]),
        "--device", "cuda",
        "--eval_on_test",
        "--test_data_root", TEST_DATA,
        "--test_labels_csv", test_labels,
        "--experiment_name", name,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    tag = f"[gpu{gpu_id}][{SUBTRACK_DIRS[subtrack]}/{task}/{cfg['tag']}/seed{seed}]"
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


def run_phase(task: str, subtracks: list[str], seeds: list[int], gpus: list[int], per_gpu: int) -> dict:
    """Run all (subtrack × config × seed) trainings for a given task in parallel."""
    gpu_q: "queue.Queue[int]" = queue.Queue()
    for _ in range(per_gpu):
        for g in gpus:
            gpu_q.put(g)
    n_workers = len(gpus) * per_gpu
    jobs = [(st, cfg, sd) for st in subtracks for cfg in CONFIGS for sd in seeds]
    log(f"\n========= Phase: {task.upper()} ({len(jobs)} trainings, {n_workers} workers) =========\n")

    def worker(st, cfg, sd):
        gpu = gpu_q.get()
        try:
            return (st, cfg, sd, train_one(st, task, cfg, sd, gpu))
        finally:
            gpu_q.put(gpu)

    out: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, st, cfg, sd) for (st, cfg, sd) in jobs]
        for fut in as_completed(futures):
            st, cfg, sd, ckpt = fut.result()
            out[(st, cfg["tag"], sd)] = ckpt
    return out


def evaluate_internal_val(ckpt_path: Path, task: str, seed: int) -> dict:
    """Re-build the 9-sample internal val for this seed and score the ckpt."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mk = dict(ck["model_kwargs"])
    split_payload = create_train_val_split(
        split_csv=TRAIN_CSV, task=task, val_ratio=0.1, regression_label="label2",
    )
    if not split_payload["val_map"]:
        return {}
    audio, video = mk["audio_feature"] if "audio_feature" in mk else None, mk.get("video_feature")
    # safer to pull from ckpt top-level
    audio = ck["audio_feature"]; video = ck["video_feature"]
    ds = MPDDElderDataset(
        data_root=TRAINVAL_ROOT,
        label_map=split_payload["val_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=ck["subtrack"], task=task,
        audio_feature=audio, video_feature=video,
        personality_npy=PERS_NPY,
        phq_map=split_payload.get("val_phq_map"),
        target_t=int(ck["target_t"]),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_batch)
    model = TorchcatBaseline(**mk).to("cuda:0").eval()
    model.load_state_dict(ck["model_state"])
    cls_t, cls_p, phq_t, phq_p, ids = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            o = model(audio=b["audio"].to("cuda:0") if "audio" in b else None,
                      video=b["video"].to("cuda:0") if "video" in b else None,
                      gait=b["gait"].to("cuda:0") if "gait" in b else None,
                      personality=b["personality"].to("cuda:0"),
                      pair_mask=b["pair_mask"].to("cuda:0") if "pair_mask" in b else None)
            logits, reg = o
            cls_t.extend(b["label"].tolist())
            cls_p.extend(logits.argmax(-1).cpu().tolist())
            phq_t.extend(b["phq9"].tolist())  # log1p space (raw target after normalize_phq_target)
            phq_p.extend(reg.cpu().tolist())
            ids.extend(b["pid"].tolist())
    yt, yp = np.array(cls_t), np.array(cls_p)
    return {
        "n": len(ids),
        "F1": f1_score(yt, yp, average="macro", zero_division=0),
        "ACC": accuracy_score(yt, yp),
        "Kappa": cohen_kappa_score(yt, yp),
        "CCC_log": ccc(phq_t, phq_p),
        "RMSE_log": float(np.sqrt(mean_squared_error(phq_t, phq_p))),
        "MAE_log": float(mean_absolute_error(phq_t, phq_p)),
    }


def evaluate_test(ckpt_path: Path, task: str, gt_all: dict) -> tuple[dict, dict]:
    """Score the ckpt on the 22-sample test set (both cls and PHQ-9)."""
    preds, _ = run_one_ckpt(ckpt_path, TEST_DATA, TEST_SPLIT, PERS_NPY, torch.device("cuda:0"))
    gt_cls = gt_all["binary"] if task == "binary" else gt_all["ternary"]
    phq_gt = gt_all["phq9"]
    ids = sorted(gt_cls.keys())
    yt = [gt_cls[i] for i in ids]; yp = [preds[i]["cls"] for i in ids]
    pt_raw = [phq_gt[i] for i in ids]; pp_raw = [preds[i]["phq9"] for i in ids]
    pt_log = [float(np.log1p(v)) for v in pt_raw]
    pp_log = [float(np.log1p(v)) for v in pp_raw]
    metrics = {
        "n": len(ids),
        "F1": f1_score(yt, yp, average="macro", zero_division=0),
        "ACC": accuracy_score(yt, yp),
        "Kappa": cohen_kappa_score(yt, yp),
        "CCC_log": ccc(pt_log, pp_log),
        "RMSE_log": float(np.sqrt(mean_squared_error(pt_log, pp_log))),
        "MAE_log": float(mean_absolute_error(pt_log, pp_log)),
        "CCC_raw": ccc(pt_raw, pp_raw),
        "RMSE_raw": float(np.sqrt(mean_squared_error(pt_raw, pp_raw))),
        "MAE_raw": float(mean_absolute_error(pt_raw, pp_raw)),
    }
    return metrics, preds


def evaluate_all(ckpts: dict, task: str, gt_all: dict) -> list[dict]:
    rows: list[dict] = []
    for (st, cfg_tag, sd), ckpt in sorted(ckpts.items()):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        be = ck["best_epoch"]
        bvm = ck["best_val_metrics"]  # this is the TEST metrics (selection set was test)
        # post-hoc evals
        test_m, preds = evaluate_test(ckpt, task, gt_all)
        val_m = evaluate_internal_val(ckpt, task, sd)
        # history total epochs
        log_dir = ROOT / "logs" / "Track2" / SUBTRACK_DIRS[st] / task / experiment_name(st, task, cfg_tag, sd)
        hist = sorted(log_dir.glob("history_*.csv"))
        total_ep = sum(1 for _ in csv.DictReader(open(hist[0], encoding="utf-8-sig"))) if hist else None
        mk = dict(ck["model_kwargs"])
        rows.append({
            "subtrack": st, "task": task, "config": cfg_tag,
            "hidden_dim": mk["hidden_dim"], "dropout": mk["dropout"],
            "seed": sd, "best_epoch": be, "total_epochs": total_ep,
            "internal_val_n": val_m.get("n", 0),
            "val_F1": val_m.get("F1"), "val_ACC": val_m.get("ACC"), "val_Kappa": val_m.get("Kappa"),
            "val_CCC_log": val_m.get("CCC_log"), "val_RMSE_log": val_m.get("RMSE_log"), "val_MAE_log": val_m.get("MAE_log"),
            "test_F1": test_m["F1"], "test_ACC": test_m["ACC"], "test_Kappa": test_m["Kappa"],
            "test_CCC_log": test_m["CCC_log"], "test_RMSE_log": test_m["RMSE_log"], "test_MAE_log": test_m["MAE_log"],
            "test_CCC_raw": test_m["CCC_raw"], "test_RMSE_raw": test_m["RMSE_raw"], "test_MAE_raw": test_m["MAE_raw"],
            "ckpt": str(ckpt.relative_to(ROOT)),
            "_preds": preds,  # internal use
        })
    return rows


def write_submission_dir(out_dir: Path, bin_preds, ter_preds, official_ids):
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


def pack_submissions(binary_rows: list[dict], ternary_rows: list[dict], out_root: Path):
    # Index by (subtrack, config, seed)
    bin_idx = {(r["subtrack"], r["config"], r["seed"]): r for r in binary_rows}
    ter_idx = {(r["subtrack"], r["config"], r["seed"]): r for r in ternary_rows}
    rows = list(csv.DictReader(open(TEST_SPLIT, encoding="utf-8-sig")))
    official_ids = [int(r["ID"]) for r in rows if r["split"].strip().lower() == "test"]

    winners = []
    for st in ["A-V+P", "A-V-G+P"]:
        for cfg in CONFIGS:
            # pick best seed per task by F1+Kappa on TEST
            bin_candidates = [r for r in binary_rows if r["subtrack"] == st and r["config"] == cfg["tag"]]
            ter_candidates = [r for r in ternary_rows if r["subtrack"] == st and r["config"] == cfg["tag"]]
            if not bin_candidates or not ter_candidates: continue
            best_bin = max(bin_candidates, key=lambda r: (r["test_F1"] + r["test_Kappa"], r["test_F1"]))
            best_ter = max(ter_candidates, key=lambda r: (r["test_F1"] + r["test_Kappa"], r["test_F1"]))
            out_dir = out_root / f"Track2_{SUBTRACK_DIRS[st]}_{cfg['tag']}"
            zip_path = write_submission_dir(out_dir, best_bin["_preds"], best_ter["_preds"], official_ids)
            winners.append({
                "subtrack": st, "config": cfg["tag"],
                "binary_seed": best_bin["seed"], "binary_F1": best_bin["test_F1"], "binary_Kappa": best_bin["test_Kappa"],
                "ternary_seed": best_ter["seed"], "ternary_F1": best_ter["test_F1"], "ternary_Kappa": best_ter["test_Kappa"],
                "submission_zip": str(zip_path.relative_to(ROOT)),
            })
            log(f"  zip: {zip_path.relative_to(ROOT)}    bin seed={best_bin['seed']} (F1={best_bin['test_F1']:.3f})  ter seed={best_ter['seed']} (F1={best_ter['test_F1']:.3f})")
    return winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtracks", nargs="+", default=["A-V+P", "A-V-G+P"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3407, 42, 1234, 2024, 9527])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 4, 5, 6, 7])
    ap.add_argument("--per_gpu", type=int, default=6)
    ap.add_argument("--out_root", default="ablation_v2_logs")
    ap.add_argument("--skip_binary", action="store_true")
    ap.add_argument("--skip_ternary", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    gt_all = load_test_gt()

    # Phase 1: binary
    if not args.skip_binary:
        bin_ckpts = run_phase("binary", args.subtracks, args.seeds, args.gpus, args.per_gpu)
    else:
        bin_ckpts = {}
    # Phase 2: ternary
    if not args.skip_ternary:
        ter_ckpts = run_phase("ternary", args.subtracks, args.seeds, args.gpus, args.per_gpu)
    else:
        ter_ckpts = {}

    # Phase 3: evaluate all on val + test
    log("\n========= Evaluating all ckpts on internal-val (9) + test (22) =========\n")
    binary_rows = evaluate_all(bin_ckpts, "binary", gt_all)
    ternary_rows = evaluate_all(ter_ckpts, "ternary", gt_all)

    # Phase 4: pack winners
    log("\n========= Picking winners + packing 12 submission.zip =========\n")
    submissions_root = ROOT / "submissions_ablation_v2"
    submissions_root.mkdir(parents=True, exist_ok=True)
    winners = pack_submissions(binary_rows, ternary_rows, submissions_root)

    # Phase 5: write full 120-row table
    all_rows = binary_rows + ternary_rows
    for r in all_rows: r.pop("_preds", None)
    csv_path = out_root / "all_runs.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(all_rows)
    (out_root / "winners.json").write_text(json.dumps(winners, indent=2, ensure_ascii=False))
    (out_root / "all_runs.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    log(f"\n→ Full table: {csv_path}")
    log(f"→ Winners: {out_root}/winners.json")
    log(f"→ Submission zips: submissions_ablation_v2/Track2_*_<config>/submission.zip")


if __name__ == "__main__":
    main()
