#!/usr/bin/env python3
"""Analyze offline_eval/results/*.json — val→test patterns across selection strategies."""
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    n = len(xs)
    return 1 - 6 * sum((rx - ry) ** 2) / (n * (n * n - 1))


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    xs = np.array(xs, float); ys = np.array(ys, float)
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def ccc_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    if len(y_true) < 2:
        return float("nan")
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    var_t = y_true.var(); var_p = y_pred.var()
    cov = ((y_true - mu_t) * (y_pred - mu_p)).mean()
    den = var_t + var_p + (mu_t - mu_p) ** 2
    return 2 * cov / den if den > 0 else float("nan")


def f1_macro(y_true, y_pred, n_class):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_class):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
        else:
            prec = tp / (tp + fp); rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def kappa(y_true, y_pred, n_class):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.zeros((n_class, n_class), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    n = cm.sum()
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0


def load_all():
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(p) as fh:
                rows.append(json.load(fh))
        except Exception:
            pass
    return rows


def parse_variant(name):
    m = re.search(r"(frozen_val_seed\d+|fold\d+_seed\d+)$", name)
    if not m: return None
    return m.group(1)


def main():
    rows = load_all()
    # Augment each row with parsed metadata.
    for r in rows:
        v = parse_variant(r.get("experiment_name", "")) or parse_variant(r.get("ckpt", ""))
        # parse from ckpt path which always has variant in dirname
        r["variant"] = v
        if v and v.startswith("frozen_val_seed"):
            r["kind"] = "frozen"
            r["seed"] = int(v.split("seed")[-1])
            r["fold"] = None
        elif v and v.startswith("fold"):
            r["kind"] = "fold"
            mm = re.match(r"fold(\d+)_seed(\d+)", v)
            r["fold"] = int(mm.group(1))
            r["seed"] = int(mm.group(2))
        else:
            r["kind"] = "unknown"
        r["combo"] = (r["track"], r["task"], r["subtrack"], r["audio_feature"],
                      r["video_feature"], r["encoder_type"])

    print(f"Loaded {len(rows)} eval records")
    print(f"By kind: frozen={sum(1 for r in rows if r['kind']=='frozen')} "
          f"fold={sum(1 for r in rows if r['kind']=='fold')}")

    # ============================================================
    # ANALYSIS A: val→test rank correlation
    # ============================================================
    print("\n" + "=" * 80)
    print("A. val F1 → test F1 correlation (across all records)")
    print("=" * 80)
    cell_groups = defaultdict(list)
    for r in rows:
        cell = (r["track"], r["task"], r["subtrack"])
        cell_groups[cell].append(r)
    print(f"{'cell':<32} | {'n':>3} | {'spearman':>9} | {'pearson':>8} | {'val_f1':<15} | {'test_f1':<15} | val→test best")
    all_corrs = []
    for cell, lst in sorted(cell_groups.items()):
        vf1 = [r["val"].get("f1", 0.0) for r in lst]
        tf1 = [r["test"].get("f1", 0.0) for r in lst]
        sp = spearman(vf1, tf1)
        pe = pearson(vf1, tf1)
        if not np.isnan(sp): all_corrs.append(sp)
        best_val_idx = int(np.argmax(vf1))
        # rank of best_val in test_f1
        test_rank = sorted(tf1, reverse=True).index(tf1[best_val_idx]) + 1
        print(f"{str(cell):<32} | {len(lst):>3} | {sp:>9.3f} | {pe:>8.3f} | "
              f"min={min(vf1):.2f}/max={max(vf1):.2f} | min={min(tf1):.2f}/max={max(tf1):.2f} | "
              f"val_best→test_rank #{test_rank}/{len(lst)}")
    print(f"\nMean Spearman across cells: {np.nanmean(all_corrs):+.3f}")
    print(f"Cells with positive correlation: {sum(1 for c in all_corrs if c>0)}/{len(all_corrs)}")
    print(f"Cells with negative correlation: {sum(1 for c in all_corrs if c<0)}/{len(all_corrs)}")

    # Best val ckpt vs best test ckpt — gap analysis
    print("\n  Best-val vs Best-test test_F1 per cell:")
    gaps = []
    for cell, lst in sorted(cell_groups.items()):
        bv = max(lst, key=lambda r: r["val"].get("f1", 0))
        bt = max(lst, key=lambda r: r["test"].get("f1", 0))
        gap = bt["test"]["f1"] - bv["test"]["f1"]
        gaps.append(gap)
        print(f"    {str(cell):<32} val→pick test_f1={bv['test'].get('f1',0):.3f} | "
              f"oracle test_f1={bt['test'].get('f1',0):.3f} | gap={gap:+.3f}")
    print(f"\n  Mean gap (oracle - val-pick): {mean(gaps):+.3f}  (large = val unreliable)")

    # ============================================================
    # ANALYSIS B: Seed selection — frozen seeds 42/3407/2026
    # ============================================================
    print("\n" + "=" * 80)
    print("B. Seed selection (frozen-val 3 seeds: 42, 3407, 2026)")
    print("=" * 80)
    # Group by combo, look at frozen records only
    by_combo_seed = defaultdict(dict)
    for r in rows:
        if r["kind"] != "frozen": continue
        by_combo_seed[r["combo"]][r["seed"]] = r

    seed_stats = {42: [], 3407: [], 2026: []}
    avg_stats, best_val_seed_stats, prob_avg_stats = [], [], []
    for combo, by_seed in by_combo_seed.items():
        if len(by_seed) < 2: continue
        for s, r in by_seed.items():
            if s in seed_stats:
                seed_stats[s].append(r["test"].get("f1", 0))
        # avg across seeds (decision-level: pick seed with highest val_f1)
        best_val = max(by_seed.values(), key=lambda r: r["val"].get("f1", 0))
        best_val_seed_stats.append(best_val["test"].get("f1", 0))
        # logits-level avg
        ids_set = set(by_seed[list(by_seed.keys())[0]]["ids"])
        valid = [r for r in by_seed.values() if set(r["ids"]) == ids_set]
        if len(valid) < 2: continue
        probs = np.mean([np.array(r["probs"]) for r in valid], axis=0)
        y_pred = probs.argmax(-1).tolist()
        y_true = valid[0]["y_true"]
        n_cls = probs.shape[1]
        prob_avg_stats.append(f1_macro(y_true, y_pred, n_cls))
        # simple avg of test_f1 (i.e., expected single-seed performance)
        avg_stats.append(mean(r["test"].get("f1", 0) for r in valid))

    for s, vals in seed_stats.items():
        if vals:
            print(f"  seed={s}: n={len(vals)} mean_test_f1={mean(vals):.3f} std={stdev(vals) if len(vals)>1 else 0:.3f}")
    print(f"\n  Strategy comparison (across {len(best_val_seed_stats)} combos with multi-seed):")
    print(f"    Single seed (avg of 3): mean_test_f1 = {mean(avg_stats):.3f}")
    print(f"    Pick best-val seed:     mean_test_f1 = {mean(best_val_seed_stats):.3f}")
    print(f"    Avg probs across seeds: mean_test_f1 = {mean(prob_avg_stats):.3f}")

    # ============================================================
    # ANALYSIS C: 5-fold ensemble vs frozen single
    # ============================================================
    print("\n" + "=" * 80)
    print("C. 5-fold ensemble vs frozen-val single ckpt")
    print("=" * 80)
    # For each combo, gather: (a) frozen seed42 single, (b) per-fold test_f1, (c) avg fold test_f1, (d) ensemble of 5 folds
    rows_a, rows_b, rows_c, rows_d = [], [], [], []
    rows_e, rows_f = [], []  # frozen 3-seed avg (logits), all combined
    combo_summary = []
    for combo, by_seed in by_combo_seed.items():
        # Find folds for this combo
        folds = [r for r in rows if r["kind"] == "fold" and r["combo"] == combo]
        if len(folds) < 3 or 42 not in by_seed:
            continue
        frozen42 = by_seed[42].get("test", {}).get("f1", 0)
        fold_tf1 = [r["test"].get("f1", 0) for r in folds]
        fold_avg = mean(fold_tf1)
        # logits ensemble across folds (only if ids align)
        ids_ref = folds[0]["ids"]
        aligned = [r for r in folds if r["ids"] == ids_ref]
        if len(aligned) >= 3:
            probs = np.mean([np.array(r["probs"]) for r in aligned], axis=0)
            y_pred = probs.argmax(-1).tolist()
            y_true = aligned[0]["y_true"]
            n_cls = probs.shape[1]
            ens_f1 = f1_macro(y_true, y_pred, n_cls)
            ens_kappa = kappa(y_true, y_pred, n_cls)
        else:
            ens_f1 = float("nan"); ens_kappa = float("nan")
        # frozen 3-seed logits ensemble
        ids_f = by_seed[42]["ids"]
        valid_f = [r for r in by_seed.values() if r["ids"] == ids_f]
        if len(valid_f) >= 2:
            probs_f = np.mean([np.array(r["probs"]) for r in valid_f], axis=0)
            frozen_ens_f1 = f1_macro(valid_f[0]["y_true"], probs_f.argmax(-1).tolist(), probs_f.shape[1])
        else:
            frozen_ens_f1 = float("nan")
        # all-7-ensemble: 3 frozen + folds
        all_aligned = valid_f + aligned
        all_ids_match = all(r["ids"] == ids_ref for r in all_aligned) and ids_ref == ids_f
        if all_ids_match and len(all_aligned) >= 4:
            probs_all = np.mean([np.array(r["probs"]) for r in all_aligned], axis=0)
            all_f1 = f1_macro(all_aligned[0]["y_true"], probs_all.argmax(-1).tolist(), probs_all.shape[1])
        else:
            all_f1 = float("nan")
        combo_summary.append({
            "combo": combo, "frozen42": frozen42, "fold_avg": fold_avg,
            "ens_5fold": ens_f1, "frozen_3seed_ens": frozen_ens_f1,
            "all_ens": all_f1, "ens_5fold_kappa": ens_kappa,
        })

    print(f"  {'combo':<70} | frozen42 | fold_avg | 5fold_ens | 3seed_ens | all_ens")
    for s in combo_summary:
        c = s["combo"]
        label = f'{c[0][-1]}/{c[1][:3]}/{c[2][:5]}/{c[3][:6]}/{c[4][:7]}'
        print(f"  {label:<70} | {s['frozen42']:>8.3f} | {s['fold_avg']:>8.3f} | "
              f"{s['ens_5fold']:>9.3f} | {s['frozen_3seed_ens']:>9.3f} | {s['all_ens']:>7.3f}")
    print(f"\n  Strategy averages across {len(combo_summary)} combos:")
    for key in ["frozen42", "fold_avg", "ens_5fold", "frozen_3seed_ens", "all_ens"]:
        vals = [s[key] for s in combo_summary if not np.isnan(s[key])]
        if vals:
            print(f"    {key:<20} mean_test_f1 = {mean(vals):.3f}  (n={len(vals)})")

    # ============================================================
    # ANALYSIS D: PHQ ensemble strategies (binary task only — has reg head)
    # ============================================================
    print("\n" + "=" * 80)
    print("D. PHQ regression ensemble — strategies for sourcing phq9_pred")
    print("=" * 80)
    # For each combo (binary or ternary), look at all available ckpts with phq pred
    # Strategies: (1) frozen42 single, (2) best-val-CCC single, (3) frozen-3seed mean,
    # (4) 5fold mean, (5) 5fold median, (6) all-ens mean
    phq_summary = []
    for combo, by_seed in by_combo_seed.items():
        folds = [r for r in rows if r["kind"] == "fold" and r["combo"] == combo]
        frozen_ckpts = list(by_seed.values())
        all_ckpts = frozen_ckpts + folds
        # Filter to ckpts with phq_pred
        all_ckpts = [r for r in all_ckpts if r.get("phq_pred_log1p")]
        if len(all_ckpts) < 3:
            continue
        # ids alignment
        ids_ref = all_ckpts[0]["ids"]
        all_ckpts = [r for r in all_ckpts if r["ids"] == ids_ref]
        if not all_ckpts:
            continue
        phq_true_log1p = all_ckpts[0]["phq_true_log1p"]
        if not phq_true_log1p:
            continue
        # real PHQ in original space (from labels file via expm1)
        phq_true_orig = np.expm1(np.array(phq_true_log1p))

        # Strategy 1: frozen42 only
        ck42 = by_seed.get(42)
        s1 = ccc_from_arrays(phq_true_orig, np.expm1(np.array(ck42["phq_pred_log1p"]))) if ck42 and ck42["ids"]==ids_ref else float("nan")

        # Strategy 2: best-val-CCC single
        best_val_ccc = max(all_ckpts, key=lambda r: r["val"].get("ccc", -99))
        s2 = ccc_from_arrays(phq_true_orig, np.expm1(np.array(best_val_ccc["phq_pred_log1p"])))

        # Strategy 3: frozen 3-seed mean (log1p space)
        valid_frozen = [r for r in frozen_ckpts if r["ids"] == ids_ref and r.get("phq_pred_log1p")]
        s3 = float("nan")
        if len(valid_frozen) >= 2:
            mean_log = np.mean([np.array(r["phq_pred_log1p"]) for r in valid_frozen], axis=0)
            s3 = ccc_from_arrays(phq_true_orig, np.expm1(mean_log))

        # Strategy 4: 5fold mean
        valid_folds = [r for r in folds if r["ids"] == ids_ref and r.get("phq_pred_log1p")]
        s4 = float("nan")
        if len(valid_folds) >= 3:
            mean_log = np.mean([np.array(r["phq_pred_log1p"]) for r in valid_folds], axis=0)
            s4 = ccc_from_arrays(phq_true_orig, np.expm1(mean_log))

        # Strategy 5: 5fold median
        s5 = float("nan")
        if len(valid_folds) >= 3:
            med_log = np.median([np.array(r["phq_pred_log1p"]) for r in valid_folds], axis=0)
            s5 = ccc_from_arrays(phq_true_orig, np.expm1(med_log))

        # Strategy 6: all 8 ckpts mean
        s6 = float("nan")
        if len(all_ckpts) >= 4:
            mean_log = np.mean([np.array(r["phq_pred_log1p"]) for r in all_ckpts], axis=0)
            s6 = ccc_from_arrays(phq_true_orig, np.expm1(mean_log))

        # Strategy 7: oracle test-best single
        best_test_ccc = -99
        best_test_pred = None
        for r in all_ckpts:
            c = ccc_from_arrays(phq_true_orig, np.expm1(np.array(r["phq_pred_log1p"])))
            if c > best_test_ccc:
                best_test_ccc = c
                best_test_pred = r
        s7 = best_test_ccc

        phq_summary.append({
            "combo": combo, "n_ckpts": len(all_ckpts),
            "frozen42": s1, "best_val_ccc": s2, "frozen3_mean": s3,
            "fold5_mean": s4, "fold5_median": s5, "all_mean": s6,
            "oracle": s7,
        })

    print(f"  {'combo':<70} | n  | fr42  | val_ccc| fr3_m | f5_m  | f5_med| all_m | oracle")
    for s in phq_summary:
        c = s["combo"]
        label = f'{c[0][-1]}/{c[1][:3]}/{c[2][:5]}/{c[3][:6]}/{c[4][:7]}'
        print(f"  {label:<70} | {s['n_ckpts']:>2} |"
              f"{s['frozen42']:>6.3f} |{s['best_val_ccc']:>7.3f} |{s['frozen3_mean']:>6.3f} |"
              f"{s['fold5_mean']:>6.3f} |{s['fold5_median']:>6.3f} |{s['all_mean']:>6.3f} |{s['oracle']:>6.3f}")
    print(f"\n  Strategy averages across {len(phq_summary)} combos:")
    for key in ["frozen42", "best_val_ccc", "frozen3_mean", "fold5_mean", "fold5_median", "all_mean", "oracle"]:
        vals = [s[key] for s in phq_summary if not np.isnan(s[key])]
        if vals:
            print(f"    {key:<14} mean_test_ccc = {mean(vals):+.3f}  (n={len(vals)})")

    # Save aggregated CSV
    out_csv = RESULTS_DIR.parent / "summary.csv"
    with open(out_csv, "w") as fh:
        fh.write("track,task,subtrack,audio,video,encoder,variant,seed,fold,"
                 "val_f1,val_ccc,val_kappa,test_f1,test_ccc,test_kappa,test_acc\n")
        for r in rows:
            v = r["val"]; t = r["test"]
            fh.write(f"{r['track']},{r['task']},{r['subtrack']},"
                     f"{r['audio_feature']},{r['video_feature']},{r['encoder_type']},"
                     f"{r['variant']},{r.get('seed','')},{r.get('fold','')},"
                     f"{v.get('f1',0):.4f},{v.get('ccc',0):.4f},{v.get('kappa',0):.4f},"
                     f"{t.get('f1',0):.4f},{t.get('ccc',0):.4f},{t.get('kappa',0):.4f},"
                     f"{t.get('acc',0):.4f}\n")
    print(f"\nFull table written to {out_csv}")


if __name__ == "__main__":
    main()
