# Make Submission for CodaBench

This folder provides sample submission files and a helper script for generating a valid CodaBench submission for the MPDD-AVG 2026 Young sub-tracks.

The official submission format is a ZIP file named `submission.zip`, which must contain the following two CSV files:

```text
submission.zip
├── binary.csv
└── ternary.csv
```

The helper script is optional. Participants may also manually create `binary.csv` and `ternary.csv` and compress them into `submission.zip`. However, we recommend using the script to check common formatting errors before submission.

## Files

```text
make_submission_forcodabench/
├── binary_sample.csv
├── ternary_sample.csv
├── submission_sample.zip
├── make_submission_sample.py
└── README.md
```

- `binary_sample.csv`: sample file for the binary classification and regression setting.
- `ternary_sample.csv`: sample file for the ternary classification and regression setting.
- `submission_sample.zip`: example ZIP structure for CodaBench submission.
- `make_submission_sample.py`: helper script for validating and packaging submission files.
- `README.md`: this instruction file.

## Submission Format

Participants should copy the sample files and fill in their own prediction results.

### binary.csv

Required columns:

```text
id,binary_pred,phq9_pred
```

Column definitions:

- `id`: official test sample ID.
- `binary_pred`: predicted binary label.
  - `0`: normal
  - `1`: depressed
- `phq9_pred`: predicted PHQ-9 score in the range `[0, 27]`.

Example:

```csv
id,binary_pred,phq9_pred
1,0,3.2
5,1,8.7
7,1,12.4
```

### ternary.csv

Required columns:

```text
id,ternary_pred,phq9_pred
```

Column definitions:

- `id`: official test sample ID.
- `ternary_pred`: predicted ternary label.
  - `0`: normal
  - `1`: mild
  - `2`: severe
- `phq9_pred`: predicted PHQ-9 score in the range `[0, 27]`.

Example:

```csv
id,ternary_pred,phq9_pred
1,0,3.1
5,2,8.9
7,1,12.2
```

## How to Generate submission.zip

### Step 1: Prepare prediction files

Copy the sample files:

```text
binary_sample.csv  ->  binary.csv
ternary_sample.csv ->  ternary.csv
```

Then fill in all prediction columns in `binary.csv` and `ternary.csv`.

Do not change the `id` column. Each CSV file must contain all official test sample IDs exactly once.

### Step 2: Run the helper script

If all files are in the same folder, run:

```bash
python make_submission_sample.py \
  --binary_csv binary.csv \
  --ternary_csv ternary.csv \
  --binary_sample binary_sample.csv \
  --ternary_sample ternary_sample.csv \
  --output_dir my_submission
```

On Windows PowerShell, you can run:

```powershell
python .\make_submission_sample.py `
  --binary_csv .\binary.csv `
  --ternary_csv .\ternary.csv `
  --binary_sample .\binary_sample.csv `
  --ternary_sample .\ternary_sample.csv `
  --output_dir .\my_submission
```

The paths can be adjusted according to where you store your files. For example:

```bash
python make_submission_sample.py \
  --binary_csv ./predictions/binary.csv \
  --ternary_csv ./predictions/ternary.csv \
  --binary_sample ./binary_sample.csv \
  --ternary_sample ./ternary_sample.csv \
  --output_dir ./my_submission
```

### Step 3: Submit to CodaBench

After running the script, the output folder will contain:

```text
my_submission/
├── binary.csv
├── ternary.csv
└── submission.zip
```

Please submit only the generated file below to CodaBench:

```text
my_submission/submission.zip
```

## Is the helper script required?

No. The helper script is optional.

If you already have valid `binary.csv` and `ternary.csv` files, you may manually compress them into `submission.zip` and submit it to CodaBench.

However, we recommend using `make_submission_sample.py` because it checks common submission errors before packaging, including:

- missing required columns
- duplicated sample IDs
- inconsistent IDs between `binary.csv` and `ternary.csv`
- missing or extra official test IDs
- invalid `binary_pred` values
- invalid `ternary_pred` values
- empty, non-numeric, or out-of-range `phq9_pred` values

The script also ensures that the generated `submission.zip` contains `binary.csv` and `ternary.csv` directly at the root level.

## Important Notes

- The final ZIP file must be named `submission.zip`.
- The ZIP file must directly contain `binary.csv` and `ternary.csv`.
- Do not place the CSV files inside an extra folder.
- Each CSV file must contain all official test sample IDs exactly once.
- Missing, duplicated, or extra IDs may cause the submission to be rejected.
- `binary_pred` must only contain `0` or `1`.
- `ternary_pred` must only contain `0`, `1`, or `2`.
- `phq9_pred` must be a raw PHQ-9 score prediction in the range `[0, 27]`.
- The helper script does not perform model inference and does not generate prediction values. Participants should fill in their own predictions first.

## Common Mistakes

Incorrect ZIP structure:

```text
submission.zip
└── my_submission/
    ├── binary.csv
    └── ternary.csv
```

Correct ZIP structure:

```text
submission.zip
├── binary.csv
└── ternary.csv
```

Incorrect prediction values:

```text
binary_pred = 2        # invalid
ternary_pred = 3       # invalid
phq9_pred = -1         # invalid
phq9_pred = 30         # invalid
```

Correct prediction values:

```text
binary_pred: 0 or 1
ternary_pred: 0, 1, or 2
phq9_pred: any numeric value in [0, 27]
```

## Contact

If you encounter submission format issues, please check the required columns, sample IDs, prediction value ranges, and ZIP structure first.