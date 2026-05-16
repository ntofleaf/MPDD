"""Run inference with official checkpoints and emit a CodaBench-format submission.

Produces, for one (track, subtrack) combination:
  submissions_official/<Track>_<Subtrack>/binary.csv
  submissions_official/<Track>_<Subtrack>/ternary.csv
  submissions_official/<Track>_<Subtrack>/submission.zip
"""
from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MPDDElderDataset, collate_batch, load_task_maps, resolve_project_path
from models import TorchcatBaseline

PROJECT_ROOT = Path(__file__).resolve().parent


def _decode_phq(value: float) -> float:
    # The training target was np.log1p(PHQ-9); invert with expm1 and clamp to [0, 27]
    return float(np.clip(np.expm1(float(value)), 0.0, 27.0))


def _find_ckpt(subdir: Path) -> Path:
    matches = sorted(subdir.rglob("best_model_*.pth"))
    if not matches:
        raise FileNotFoundError(f"No best_model_*.pth under {subdir}")
    return matches[0]


def run_one_ckpt(ckpt_path: Path, data_root: str, split_csv: str, personality_npy: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    task = ck["task"]
    subtrack = ck["subtrack"]
    audio_feature = ck["audio_feature"]
    video_feature = ck["video_feature"]
    target_t = int(ck["target_t"])
    regression_label = ck.get("regression_label") or "label2"

    task_maps = load_task_maps(split_csv, task, regression_label)
    dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack=subtrack,
        task=task,
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        phq_map=task_maps["test_phq_map"],
        target_t=target_t,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)
    model_kwargs = dict(ck["model_kwargs"])
    model = TorchcatBaseline(**model_kwargs).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    use_reg = bool(model_kwargs.get("use_regression_head", False))

    preds: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for batch in loader:
            out = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if use_reg:
                logits, reg_out = out
                reg_out_np = reg_out.cpu().numpy().tolist()
            else:
                logits, reg_out_np = out, None
            cls = logits.argmax(dim=-1).cpu().numpy().tolist()
            ids = batch["pid"].cpu().numpy().tolist()
            for i, pid in enumerate(ids):
                preds[int(pid)] = {
                    "cls": int(cls[i]),
                    "phq9": _decode_phq(reg_out_np[i]) if reg_out_np is not None else 0.0,
                }
    return preds, task


def write_submission(out_dir: Path, binary_preds: dict[int, dict], ternary_preds: dict[int, dict], official_ids: list[int]):
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "binary.csv"
    ter_path = out_dir / "ternary.csv"

    with open(bin_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "binary_pred", "phq9_pred"])
        for pid in official_ids:
            p = binary_preds.get(pid, {"cls": 0, "phq9": 0.0})
            w.writerow([pid, int(p["cls"]), f"{p['phq9']:.4f}"])

    with open(ter_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ternary_pred", "phq9_pred"])
        for pid in official_ids:
            p = ternary_preds.get(pid, {"cls": 0, "phq9": 0.0})
            w.writerow([pid, int(p["cls"]), f"{p['phq9']:.4f}"])

    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(bin_path, "binary.csv")
        zf.write(ter_path, "ternary.csv")
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["Track1", "Track2"])
    ap.add_argument("--subtrack", required=True, choices=["A-V-G+P", "A-V-P", "G-P"])
    ap.add_argument("--out_root", default="submissions_official")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    if args.track == "Track1":
        data_root = "MPDD-AVG2026/MPDD-AVG2026-test/Elder"
        split_csv = "MPDD-AVG2026/MPDD-AVG2026-test/Elder/split_labels_test.csv"
        personality_npy = "MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy"
    else:
        data_root = "MPDD-AVG2026/MPDD-AVG2026-test/Young"
        split_csv = "submissions_official/split_labels_test_Young_fixed.csv"
        personality_npy = "MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy"

    ck_dir = PROJECT_ROOT / "checkpoints" / args.track / args.subtrack
    bin_ck = _find_ckpt(ck_dir / "binary")
    ter_ck = _find_ckpt(ck_dir / "ternary")
    print(f"[{args.track}/{args.subtrack}] binary ckpt: {bin_ck.relative_to(PROJECT_ROOT)}")
    print(f"[{args.track}/{args.subtrack}] ternary ckpt: {ter_ck.relative_to(PROJECT_ROOT)}")

    bin_preds, _ = run_one_ckpt(bin_ck, data_root, split_csv, personality_npy, device)
    ter_preds, _ = run_one_ckpt(ter_ck, data_root, split_csv, personality_npy, device)

    # Official ID order from the split CSV
    rows = list(csv.DictReader(open(resolve_project_path(split_csv), encoding="utf-8-sig")))
    official_ids = [int(r["ID"]) for r in rows if r["split"].strip().lower() == "test"]

    out_dir = Path(args.out_root) / f"{args.track}_{args.subtrack.replace('+', '-')}"
    zip_path = write_submission(out_dir, bin_preds, ter_preds, official_ids)
    print(f"[{args.track}/{args.subtrack}] wrote: {zip_path}")


if __name__ == "__main__":
    main()
