"""Pick top-N combos per (track, task, subtrack) from grid summary CSVs.

Gates applied before ranking:
  - collapse_count == 0       (none of the 3 grid seeds produced a degenerate ckpt)
  - recall_min_class_mean >= MIN_RECALL_GATE

Among survivors, keep top N by f1_mean (descending). If a cell has fewer than N
survivors, we relax the recall gate progressively (0.3 -> 0.2 -> 0.0). If still
empty, fall back to "best by f1_mean ignoring all gates" — better to validate
*something* than nothing.

Output: JSON file with shape
  {
    "Track1": {
      "binary": {
        "G+P":     [{"audio": ..., "video": ..., "encoder": ..., "f1_mean": ...}, ...],
        "A-V+P":   [...],
        "A-V-G+P": [...]
      },
      "ternary": {...}
    },
    "Track2": {...}
  }
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_SUBTRACKS = ("G+P", "A-V+P", "A-V-G+P")


def _read_summary(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # cast numerics
            for k in ("n_runs", "collapse_count"):
                row[k] = int(float(row[k]))
            for k in ("f1_mean", "f1_std", "acc_mean", "kappa_mean", "ccc_mean",
                      "rmse_mean", "mae_mean",
                      "recall_min_class_mean", "recall_min_class_min"):
                try:
                    row[k] = float(row[k])
                except (KeyError, ValueError):
                    row[k] = 0.0
            out.append(row)
    return out


def _select_cell(rows: list[dict], top_n: int, recall_floor: float) -> list[dict]:
    """Apply gates then return top_n by f1_mean."""
    gates = [recall_floor, 0.2, 0.0]  # progressive relaxation
    for floor in gates:
        survivors = [
            r for r in rows
            if r["collapse_count"] == 0 and r["recall_min_class_mean"] >= floor
        ]
        if len(survivors) >= top_n:
            survivors.sort(key=lambda r: -r["f1_mean"])
            return survivors[:top_n]
    # last-resort fallback: just take top_n by f1_mean
    rows.sort(key=lambda r: -r["f1_mean"])
    return rows[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_dir", default="experiments/grid_baseline")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--recall_floor", type=float, default=0.3)
    parser.add_argument("--out", default="experiments/grid_baseline/top_combos_for_kfold.json")
    args = parser.parse_args()

    grid_dir = (PROJECT_ROOT / args.grid_dir).resolve()
    # Find grid CSVs
    csv_files = sorted(grid_dir.glob("grid_summary_*.csv"))
    if not csv_files:
        raise SystemExit(f"No grid_summary_*.csv under {grid_dir}")

    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for csv_file in csv_files:
        for row in _read_summary(csv_file):
            track = row["track"]
            task = row["task"]
            sub = row["subtrack"]
            by_cell[(track, task, sub)].append(row)

    selection: dict = {}
    print(f"[select] top_n={args.top_n}  recall_floor>={args.recall_floor}  "
          f"gate=collapse_count==0 + progressively-relaxed recall")
    print()
    for (track, task, sub), rows in sorted(by_cell.items()):
        chosen = _select_cell(rows, top_n=args.top_n, recall_floor=args.recall_floor)
        selection.setdefault(track, {}).setdefault(task, {})[sub] = [
            {
                "audio": c["audio"],
                "video": c["video"],
                "encoder": c["encoder"],
                "f1_mean": round(c["f1_mean"], 4),
                "kappa_mean": round(c["kappa_mean"], 4),
                "ccc_mean": round(c["ccc_mean"], 4),
                "recall_min_class_mean": round(c["recall_min_class_mean"], 4),
                "collapse_count": c["collapse_count"],
            }
            for c in chosen
        ]
        print(f"[{track}/{task:8s}/{sub:9s}] selected {len(chosen)} combo(s):")
        for c in chosen:
            print(f"   audio={c['audio']:9s} video={c['video']:9s}  "
                  f"f1={c['f1_mean']:.3f}  kappa={c['kappa_mean']:+.3f}  "
                  f"ccc={c['ccc_mean']:+.3f}  recall_min={c['recall_min_class_mean']:.2f}  "
                  f"collapse={c['collapse_count']}/3")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, ensure_ascii=False)
    print(f"\n[select] wrote {out_path}")


if __name__ == "__main__":
    main()
