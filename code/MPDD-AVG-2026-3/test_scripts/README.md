# Test Scripts

## Directory Layout

```text
test_scripts/
├── _common.sh
├── Track1/
│   ├── A-V-G+P/
│   │   ├── run_binary.sh
│   │   └── run_ternary.sh
│   ├── A-V-P/
│   │   ├── run_binary.sh
│   │   └── run_ternary.sh
│   └── G-P/
│       ├── run_binary.sh
│       └── run_ternary.sh
└── Track2/
    ├── A-V-G+P/
    │   ├── run_binary.sh
    │   └── run_ternary.sh
    ├── A-V-P/
    │   ├── run_binary.sh
    │   └── run_ternary.sh
    └── G-P/
        ├── run_binary.sh
        └── run_ternary.sh
```

## Related Repository Layout

These scripts are intended to work with the following repository structure at the project root:

```text
MPDD_AVG-github/
├── checkpoints/
│   ├── Track1/
│   └── Track2/
├── MPDD-AVG2026/
│   ├── MPDD-AVG2026-test/
│   │   ├── Elder/
│   │   └── Young/
│   └── MPDD-AVG2026-trainval/
│       ├── Elder/
│       └── Young/
├── test.py
└── test_scripts/
```

The default relative paths are:

- Test set root:
  - `Track1 -> MPDD-AVG2026/MPDD-AVG2026-test/Elder`
  - `Track2 -> MPDD-AVG2026/MPDD-AVG2026-test/Young`
- Test split CSV:
  - `Track1 -> MPDD-AVG2026/MPDD-AVG2026-test/Elder/split_labels_test.csv`
  - `Track2 -> MPDD-AVG2026/MPDD-AVG2026-test/Young/split_labels_test.csv`
- Personality embeddings:
  - `Track1 -> MPDD-AVG2026/MPDD-AVG2026-trainval/Elder/descriptions_embeddings_with_ids.npy`
  - `Track2 -> MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy`
- Log output:
  - `logs/test_scripts`

## Checkpoint Directory Convention

Each script scans its corresponding checkpoint directory with the following layout:

```text
checkpoints/
  TrackX/
    SubtrackDir/
      binary|ternary/
        experiment_name/
          best_model_*.pth
```

## How To Run

Examples:

```bash
bash test_scripts/Track1/A-V-G+P/run_binary.sh
bash test_scripts/Track1/A-V-P/run_ternary.sh
bash test_scripts/Track2/G-P/run_binary.sh
```

You can also explicitly set the Python interpreter or device:

```bash
PYTHON_BIN=python3 DEVICE=cpu bash test_scripts/Track2/A-V-G+P/run_ternary.sh
```

## Test Example

```bash
python test.py \
  --checkpoint checkpoints/Track2/A-V-G+P/ternary/track2_ternary_A-V-G+P_bilstm_mean_wav2vec__resnet_log1p/best_model_2026-04-13-17.32.11.pth \
  --data_root MPDD-AVG2026/MPDD-AVG2026-test/Young \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-test/Young/split_labels_test.csv \
  --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy
```

## Output

All scripts eventually call the repository-level `test.py`, and outputs are written to:

```text
logs/test_scripts/<Track>/<SubtrackDir>/<Task>/<experiment_name>/
```

Each run usually generates:

- `test_result_only_*.json`
- `<experiment_name>_test_only.csv`

Notes

- Each script only scans `best_model_*.pth` files under its own checkpoint directory
- All default paths are now repository-relative
- If the default `PERSONALITY_NPY` file does not exist, the script exits with an error
- `test.py` includes compatibility remapping for old absolute paths stored in legacy checkpoints, as long as the target files exist in the current repository structure
