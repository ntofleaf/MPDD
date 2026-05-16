# MPDD-AVG Baseline

> A multimodal baseline framework for **MPDD-AVG 2026** — supporting depression detection across elderly and young populations via audio, video, gait, and personality modalities.

<div align="left">

[![Website](https://img.shields.io/badge/🌐%20Official%20Website-MPDD--AVG%202026-brightgreen?style=for-the-badge&logo=github&logoColor=white)](https://hacilab.github.io/MPDD-AVG-2026.github.io/index.html)
[![Baseline Report](https://img.shields.io/badge/📑%20Baseline%20Report-MPDD--AVG%202026-blue?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://github.com/hacilab/MPDD-AVG-2026/blob/main/MPDD_AVG_baseline.pdf)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-MPDD--AVG--2026-orange?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/chasonfff/MPDD-AVG-2026/tree/main)

</div>

---

## ✨ Supported Configuration

| Dimension            | Options                                                                   |
| -------------------- | ------------------------------------------------------------------------- |
| **Tracks**     | `Track1` (Elder) · `Track2` (Young)                                  |
| **Tasks**      | Binary classification (`label2`) · Ternary classification (`label3`) |
| **Sub-tracks** | `A-V+P` · `A-V-G+P` · `G+P`                                       |
| **Encoders**   | `bilstm_mean` · `hybrid_attn`                                        |

> Both binary and ternary classification jointly train a **PHQ-9 regression head**, so each experiment outputs both classification metrics and PHQ-9 regression metrics.

---

## The weights of our best model will be uploaded soon!

---

## 📋 Table of Contents

1. [Key Features](#1-key-features)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Modalities &amp; Features](#5-supported-modalities--features)
6. [Quick Start](#6-quick-start)
7. [Manual CLI Usage](#7-manual-cli-usage)
8. [Train/Val Split Strategy](#8-trainval-split-strategy)
9. [Output Files](#9-output-files)
10. [Metrics &amp; Model Selection](#10-metrics--model-selection)
11. [Reproducing Experiments](#11-reproducing-experiments)
12. [FAQ](#12-faq)

---

## 1. Key Features

- Unified `train / val / test` pipeline
- Official **test set** is evaluated only once after training completes
- Internal `val` split carved from the official `train` set
- **ID-based split** — no subject leaks across `train` and `val`
- Joint training of **classification head + PHQ-9 regression head** (both binary and ternary)
- Training logs, history curves, checkpoints, and result summaries all saved automatically

**Output metrics (6 total):**

| Metric       | Source                |
| ------------ | --------------------- |
| `Macro-F1` | Classification head   |
| `ACC`      | Classification head   |
| `Kappa`    | Classification head   |
| `CCC`      | PHQ-9 regression head |
| `RMSE`     | PHQ-9 regression head |
| `MAE`      | PHQ-9 regression head |

---

## 2. Project Structure

```text
MPDD_Avg/
├── config.json
├── dataset.py
├── metrics.py
├── train.py
├── test.py
├── train_val_split.py
├── models/
│   ├── __init__.py
│   ├── hybrid_temporal_encoder.py
│   └── torchcat_baseline.py
├── scripts/
│   ├── Track1/                  # Elder
│   │   ├── A-V-P/
│   │   │   ├── run_binary.sh
│   │   │   └── run_ternary.sh
│   │   ├── A-V-G+P/
│   │   └── G-P/
│   └── Track2/                  # Young
│       ├── A-V-P/
│       ├── A-V-G+P/
│       └── G-P/
├── checkpoints/(the checkpoints of baseline system have been uploaded!)
├── logs/
└── MPDD-AVG2026/
```

---

## 3. Environment Setup

**Recommended environment:** Linux · Python `3.10` or `3.11` · PyTorch matching your local CUDA version

```bash
# Create and activate conda environment
conda create -n mpddavg python=3.10 -y
conda activate mpddavg
pip install --upgrade pip

# Install PyTorch (choose the command matching your CUDA version)
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install numpy scikit-learn
```

> **CPU-only?** Set `DEVICE=cpu` when running any script — no code changes needed.

---

## 4. Dataset Preparation

Place the dataset under the repository root with the following structure:

```text
MPDD-AVG2026/
├── MPDD-AVG2026-trainval/
│   ├── Elder/
│   │   ├── Audio/
│   │   ├── Video/
│   │   ├── IMU/
│   │   ├── split_labels_train.csv
│   │   └── descriptions_embeddings_with_ids.npy
│   └── Young/
│       ├── Audio/
│       ├── Video/
│       ├── IMU/
│       ├── split_labels_train.csv
│       └── descriptions_embeddings_with_ids.npy
└── MPDD-AVG2026-test/
    ├── Elder/
    │   ├── Audio/
    │   ├── Video/
    │   ├── IMU/
    │   └── split_labels_test.csv
    └── Young/
        ├── Audio/
        ├── Video/
        ├── IMU/
        └── split_labels_test.csv
```

**Required label file columns:** `ID` · `split` · `label2` · `label3` · `PHQ-9`

**Personality feature file:** `descriptions_embeddings_with_ids.npy`

---

## 5. Supported Modalities & Features

**Sub-track definitions:**

| Sub-track   | Modalities                         |
| ----------- | ---------------------------------- |
| `A-V+P`   | Audio + Video + Personality        |
| `A-V-G+P` | Audio + Video + Gait + Personality |
| `G+P`     | Gait + Personality only            |

**Feature dimensions by track:**

| Modality | Feature                    | Track1 (Elder) | Track2 (Young) |
| -------- | -------------------------- | :------------: | :------------: |
| Audio    | `mfcc` / `mfcc64`      |       64       |       64       |
| Audio    | `opensmile`              |       65       |       65       |
| Audio    | `wav2vec` / `wav2vec2` |      768      |      1024      |
| Video    | `densenet`               |      1000      |      1000      |
| Video    | `resnet`                 |      1000      |      1000      |
| Video    | `openface`               |      710      |      710      |
| IMU      | `gait`                   |       12       |       12       |

**Temporal processing:** All time-series modalities (`audio / video / gait`) are read as raw `[T, C]` tensors, then **linearly interpolated** to a fixed length `target_t` (default: `128`). The `personality` feature is a fixed-length vector and is not interpolated. This is consistent across all tracks and sub-tracks.

---

## 6. Quick Start

### 6.1 Run Pre-built Scripts

```bash
# Track1 / Elder / A-V-G+P — binary classification (runs all 9 A/V feature combos)
bash scripts/Track1/A-V-G+P/run_binary.sh

# Track1 / Elder / G+P — ternary classification
bash scripts/Track1/G-P/run_ternary.sh

# Track2 / Young / A-V+P — binary classification (runs all 9 A/V feature combos)
bash scripts/Track2/A-V-P/run_binary.sh

# Track2 / Young / A-V-G+P — ternary classification
bash scripts/Track2/A-V-G+P/run_ternary.sh
```

> Scripts under `A-V-P` and `A-V-G+P` automatically iterate over **9 audio/video feature combinations** (3 audio × 3 video). Scripts under `G-P` only run the gait-only track.

### 6.2 Override Hyperparameters via Environment Variables

```bash
DEVICE=cpu EPOCHS=5 BATCH_SIZE=4 LR=1e-3 bash scripts/Track1/A-V-P/run_binary.sh
```

**All overridable parameters:**

| Variable         | Description                          |
| ---------------- | ------------------------------------ |
| `DEVICE`       | `cuda` or `cpu`                  |
| `SEED`         | Random seed                          |
| `EPOCHS`       | Number of training epochs            |
| `BATCH_SIZE`   | Batch size                           |
| `LR`           | Learning rate                        |
| `WEIGHT_DECAY` | Weight decay                         |
| `HIDDEN_DIM`   | Hidden dimension size                |
| `DROPOUT`      | Dropout rate                         |
| `PATIENCE`     | Early stopping patience              |
| `MIN_DELTA`    | Minimum delta for early stopping     |
| `TARGET_T`     | Temporal interpolation target length |
| `PYTHON_BIN`   | Python binary path                   |

---

## 7. Manual CLI Usage

### 7.1 Training

**Binary classification:**

```bash
python train.py \
  --track Track1 \
  --task binary \
  --subtrack A-V-G+P \
  --encoder_type bilstm_mean \
  --audio_feature wav2vec \
  --video_feature resnet \
  --device cuda
```

**Ternary classification:**

```bash
python train.py \
  --track Track2 \
  --task ternary \
  --subtrack A-V+P \
  --encoder_type bilstm_mean \
  --audio_feature wav2vec \
  --video_feature resnet \
  --data_root MPDD-AVG2026/MPDD-AVG2026-trainval/Young \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv \
  --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy \
  --device cuda
```

Available encoders: `bilstm_mean` · `hybrid_attn`

### 7.2 Evaluate a Specific Checkpoint （test_scripts/README.md）

**Test Example**

```
python test.py
   --checkpoint checkpoints/Track2/A-V-G+P/ternary/track2_ternary_A-V-G+P_bilstm_mean_wav2vec__resnet_log1p/best_model_*.pth
   --data_root MPDD-AVG2026/MPDD-AVG2026-test/Young
   --split_csv MPDD-AVG2026/MPDD-AVG2026-test/Young/split_labels_test.csv
   --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy
```

### 7.3 Preview Train/Val Split

```bash
python train_val_split.py \
  --task ternary \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/split_labels_train.csv \
  --save_path tmp/elder_ternary_split_preview.csv
```

---

## 8. Train/Val Split Strategy

| Rule              | Detail                                                         |
| ----------------- | -------------------------------------------------------------- |
| Scope             | Val is carved only from the official `train` split           |
| Subject isolation | Splits by `ID` — no subject appears in both train and val   |
| Stratification    | `binary` stratifies by `label2`; `ternary` by `label3` |
| Default ratio     | `val_ratio = 0.1`                                            |
| Reproducibility   | Fixed split as long as data and random seed are unchanged      |

> 💡 **Tip — Ensure class coverage in val:** When splitting, make sure the validation set samples **at least one example per class label**. With small datasets or imbalanced label distributions, a naive random split may leave certain classes entirely out of val. Use stratified sampling (e.g., `sklearn.model_selection.StratifiedShuffleSplit`) to guarantee that every class in `label2` / `label3` is represented in the validation set — otherwise your val metrics will be unreliable and early stopping may select suboptimal checkpoints.

---

## 9. Output Files

Training outputs are written to:

```text
checkpoints/{Track}/{SubtrackDir}/{task}/{experiment_name}/
logs/{Track}/{SubtrackDir}/{task}/{experiment_name}/
```

| File                                  | Description                               |
| ------------------------------------- | ----------------------------------------- |
| `best_model_{timestamp}.pth`        | Best checkpoint                           |
| `result_{timestamp}.log`            | Training log                              |
| `history_{timestamp}.csv`           | Metric history per epoch                  |
| `train_result_{timestamp}.json`     | Consolidated result summary               |
| `{experiment_name}.csv`             | Experiment result table                   |
| `test_result_only_{timestamp}.json` | Test-only result (when using `test.py`) |

---

## 10. Metrics & Model Selection

Classification tasks output 6 metrics at val and test time:

| Metric       | Head             | Task             |
| ------------ | ---------------- | ---------------- |
| `Macro-F1` | Classification   | binary / ternary |
| `ACC`      | Classification   | binary / ternary |
| `Kappa`    | Classification   | binary / ternary |
| `CCC`      | PHQ-9 regression | all              |
| `RMSE`     | PHQ-9 regression | all              |
| `MAE`      | PHQ-9 regression | all              |

**Best checkpoint selection criterion:**

- `binary` / `ternary`: highest `Macro-F1` on the val set
- `regression`: highest `CCC` on the val set

The best model is only overwritten when the val metric strictly improves.

---

## 11. Reproducing Experiments

For reproducibility, **run the scripts in `scripts/` directly** — they already fix:

- `track`, `task`, `subtrack`
- `audio_feature`, `video_feature`
- Common hyperparameters

Notes:

- `Track1` uses the Elder default paths in `config.json`
- `Track2` scripts explicitly override to the Young data paths
- Default `target_t = 128`

---

## 12. FAQ

**Q: Can I run without a GPU?**

Yes. Prefix your command with `DEVICE=cpu`:

```bash
DEVICE=cpu bash scripts/Track1/A-V-P/run_binary.sh
```

**Q: How do I switch audio/video features?**

Edit the script directly, or pass arguments on the command line:

```bash
python train.py --audio_feature mfcc --video_feature densenet ...
```

**Q: How do I switch the encoder?**

```bash
python train.py --encoder_type hybrid_attn ...
```

**Q: Why do Track2 commands have longer data paths?**

`config.json` defaults to `Track1 / Elder`. Track2 scripts explicitly pass the Young data paths, so you can just run the scripts as-is.

**Q: How is temporal sequence length handled?**

All time-series modalities are linearly interpolated to `target_t` (default: `128`) — not hard-truncated or randomly cropped. To change this:

```bash
TARGET_T=256 bash scripts/Track1/G-P/run_binary.sh
# or
python train.py --target_t 256 ...
```

**Q: Why should Track2 video test results be interpreted with caution?**

The following test files in the current dataset release are **empty (0 bytes)**:

- `MPDD-AVG2026-test/Young/Video/densenet`
- `MPDD-AVG2026-test/Young/Video/resnet`
- `MPDD-AVG2026-test/Young/Video/openface`

As a result, the video branch in `Track2 / A-V+P` and `Track2 / A-V-G+P` receives zero-padded input during testing. The pipeline will still complete, but video information is absent. `Track2 / G+P` is **not affected** by this issue.

Additionally, `Young trainval / Video / openface` is not fully valid either — always cross-reference the number of valid samples reported in the training logs.

---

<div align="center">

*For questions or issues, please open a GitHub Issue on the [project repository](https://github.com/hacilab/MPDD-AVG-2026).*

</div>
